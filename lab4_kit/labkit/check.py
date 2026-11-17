"""The check engine: check("C1", squared_distance) and check_all().

What a check does, in order:
  1. finds the task in labkit/cases/labN.py and the object you passed;
  2. NOT STARTED if the body is still only `pass` (or `...`);
  3. runs the toy cases, then the real cases, then any extra checks of the task (custom hook);
  4. property checks: the type of every result, and your inputs left unchanged;
  5. prints PASS with a few of the tests it ran (TEST n: what it checks, the inputs, Expected, Your result),
     or NOT YET with the FIRST failing test in the same format, the likely cause and a next step;
  6. on PASS: what you practised, a quality line, and an optional speed note (never a failure).

The report looks like this (the labels of the inputs come from the task, e.g. Predictions / Ground truth):

    C3 accuracy: NOT YET (2 of 6 tests passed)
       TEST 3: Three of four predictions right
          Predictions:        [1, 2, 3, 4]
          Ground truth:       [1, 2, 8, 4]
          Expected accuracy:  0.75
          Your result:        1.0
          NOT YET
       Likely cause:  ...
       Next step:     ...

Your code always runs on copies of the inputs, with its printing hidden, and any error it raises
is caught and summarised. Nothing is graded or sent anywhere.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import dis
import inspect
import io
import os
import re
import sys
import sysconfig
import textwrap
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cases as lab_registry
from . import reference as reference_registry
from .env import current_lab
from .spec import Attempt, Compare, Described, Raises, Task, exact
from .text import INDENT, LABEL_WIDTH, changed_where, field as report_field, first_difference
from .text import hide_user_folder, is_number
from .text import preview_pair, say, short, type_name
from .timing import best_time, format_duration, machine_factor

PASS, NOT_YET, NOT_STARTED = "PASS", "NOT YET", "NOT STARTED"
TIME_LIMIT_S = 10.0                       # stop running more cases after this long
LABKIT_DIR = Path(__file__).resolve().parent
_ATOMS = (int, float, str, bool, type(None))


# ============================================================================= reading code
_SKIP_OPCODES = {"RESUME", "NOP", "CACHE", "EXTENDED_ARG", "COPY_FREE_VARS", "MAKE_CELL", "NOT_TAKEN"}
_NOT_PLAIN = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR


def _does_nothing(code) -> bool:
    """True if the compiled body only returns None (a body of pass, ..., or just a docstring)."""
    if code.co_flags & _NOT_PLAIN:
        return False
    steps = [ins for ins in dis.get_instructions(code) if ins.opname not in _SKIP_OPCODES]
    if len(steps) == 1:                                       # Python 3.12-3.13: RETURN_CONST None
        return steps[0].opname == "RETURN_CONST" and steps[0].argval is None
    if len(steps) == 2:                                       # Python 3.10, 3.11, 3.14
        return (steps[0].opname == "LOAD_CONST" and steps[0].argval is None
                and steps[1].opname == "RETURN_VALUE")
    return False


def _own_functions(cls: type) -> list:
    """The functions defined directly in a class body (methods, static and class methods)."""
    found = []
    for value in vars(cls).values():
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        elif isinstance(value, property):
            value = value.fget
        if inspect.isfunction(value):
            found.append(value)
    return found


def is_stub(obj: object) -> bool:
    """True if `obj` is still a stub: a function whose body is only pass / ... / a docstring,
    or a class whose own methods are all like that (or that has no methods yet)."""
    if isinstance(obj, type):
        return all(is_stub(method) for method in _own_functions(obj))
    function = getattr(obj, "__func__", obj)                 # bound methods
    code = getattr(function, "__code__", None)
    return code is not None and _does_nothing(code)


def _code_objects(obj: object) -> list:
    if isinstance(obj, type):
        functions = _own_functions(obj)
    else:
        functions = [getattr(obj, "__func__", obj)]
    todo = [f.__code__ for f in functions if hasattr(f, "__code__")]
    found = []
    while todo:
        code = todo.pop()
        found.append(code)
        todo += [const for const in code.co_consts if inspect.iscode(const)]
    return found


def names_used(obj: object) -> set[str]:
    """Global names (functions, variables) that the code of `obj` refers to."""
    names: set[str] = set()
    for code in _code_objects(obj):
        names.update(code.co_names)
    return names


def _globals_of(obj: object) -> dict:
    if isinstance(obj, type):
        functions = _own_functions(obj)
        return functions[0].__globals__ if functions else {}
    return getattr(getattr(obj, "__func__", obj), "__globals__", {})


def _parameter_names(fn: object) -> list[str]:
    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    if code is None:
        return []
    return list(code.co_varnames[: code.co_argcount + code.co_kwonlyargcount])


# ============================================================================= tracebacks
def _system_folders() -> list[str]:
    folders = {str(LABKIT_DIR)}
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        if paths.get(key):
            folders.add(paths[key])
    return [os.path.normcase(os.path.abspath(f)) for f in folders]


_SYSTEM_FOLDERS = _system_folders()


def is_student_file(filename: str) -> bool:
    """True for code the student wrote: a notebook cell, or a .py file outside labkit and Python itself."""
    if not filename or filename.startswith("<frozen"):
        return False
    if filename.startswith("<"):                     # "<string>", "<ipython-input-3-...>"
        return True
    path = os.path.normcase(os.path.abspath(filename))
    return not any(path.startswith(folder + os.sep) for folder in _SYSTEM_FOLDERS)


def _student_frame(error: BaseException) -> tuple[int, str, Any] | None:
    """(line number, line text, code object) of the deepest traceback frame in the student's own code."""
    codes = []
    tb = error.__traceback__
    while tb is not None:
        codes.append(tb.tb_frame.f_code)
        tb = tb.tb_next
    frames = traceback.extract_tb(error.__traceback__)
    for code, frame in reversed(list(zip(codes, frames))):
        if is_student_file(frame.filename):
            return frame.lineno, (frame.line or "").strip(), code
    return None


def student_line(error: BaseException) -> tuple[int, str] | None:
    """(line number, line text) of the deepest traceback frame in the student's own code."""
    found = _student_frame(error)
    return None if found is None else (found[0], found[1])


def your_line(error: BaseException, obj: object = None, *, always_name: bool = False) -> tuple[str, str] | None:
    """('Your line 6 (in squared_distance):', 'diff = a[i] - b[i]') for the deepest line of student code.

    The function's name is added when that line is not inside `obj` itself (e.g. the error happened in the
    student's C1 function while C2 was being checked), so the number isn't read against the wrong cell.
    always_name=True adds it whenever the line is inside a function (for messages printed far from the def).
    """
    found = _student_frame(error)
    if found is None:
        return None
    number, text, code = found
    label = f"Your line {number}"
    name = getattr(code, "co_name", "<module>")
    if not name.startswith("<") and (always_name or (obj is not None and code not in _code_objects(obj))):
        label += f" (in {name})"
    return label + ":", text or "(the line's text isn't available)"


def def_line(obj: object) -> str | None:
    """The first line of the student's definition, e.g. 'def squared_distance(a):'."""
    try:
        lines, _ = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    for line in lines:
        if line.strip().startswith(("def ", "class ", "async def ")):
            return line.strip()
    return lines[0].strip() if lines else None


def format_traceback(error: BaseException) -> list[str]:
    """The traceback from the first frame of student code onwards (labkit's own frames dropped).

    The user folder is shown as ~ (it is often the student's name), so the traceback can be pasted
    into an AI tutor, as the hints suggest.
    """
    tb = error.__traceback__
    while tb is not None and not is_student_file(tb.tb_frame.f_code.co_filename):
        tb = tb.tb_next
    text = hide_user_folder("".join(traceback.format_exception(type(error), error, tb)))
    return text.rstrip("\n").split("\n")


# ============================================================================= likely causes
def likely_cause(attempt: Attempt, expected: Any = None) -> str | None:
    """A generic explanation when no task-specific detector matched (lab-agnostic)."""
    error = attempt.error
    if error is not None:
        name, text = type(error).__name__, str(error)
        if isinstance(error, UnboundLocalError):
            return "a variable is used before it gets a value in this function. Give it a starting value before the loop."
        if isinstance(error, NameError):
            missing = getattr(error, "name", None) or (re.findall(r"'(\w+)'", text) or ["?"])[0]
            return (f"Python doesn't know the name {missing!r}. Check the spelling, or run the cell "
                    "that defines it first.")
        if isinstance(error, TypeError):
            if "positional argument" in text or "required keyword" in text or "unexpected keyword" in text:
                if student_line(error) is None:       # the check's own call didn't fit the def line
                    return ("your def line has different parameters from the task card. Compare them "
                            "with the signature in the card.")
                return ("a function is called with the wrong number of arguments. Compare the call on "
                        "your line with that function's def line.")
            if "NoneType" in text:
                return ("a value is None where Python needed a number or a list. A function you call "
                        "may be missing its return line, or a variable starts as None.")
            if "'str'" in text and ("operand" in text or "concatenate" in text):
                return "some values are text (str), not numbers. Did you forget int()?"
            if "not iterable" in text:
                return "you looped over a single value. A for loop needs a list (or another sequence)."
            if "not subscriptable" in text:
                return "you used [ ] on a value that isn't a list or dict. Check what that variable holds."
            if "not callable" in text:
                return "you put ( ) after something that isn't a function. Did a variable reuse a function's name?"
        if isinstance(error, IndexError):
            return ("an index goes past the end of a list. Check the limits of your range(...), "
                    "and remember that the last index is len(x) - 1.")
        if isinstance(error, KeyError):
            return ("you read a dict key that isn't there yet. Use counts.get(key, 0), or test "
                    "`if key in counts` first.")
        if isinstance(error, ZeroDivisionError):
            return "you divided by zero. Which input makes the bottom of the fraction 0? Handle that case first."
        if isinstance(error, AttributeError):
            if "NoneType" in text:
                return "a value is None, so it has no methods. A function you call may be missing its return line."
            return "that value doesn't have this method or attribute. Check the spelling and what type the value is."
        if isinstance(error, RecursionError):
            return "the function keeps calling itself and never stops."
        if isinstance(error, ValueError) and "invalid literal for int()" in text:
            return "int() got text that isn't a whole number."
        if isinstance(error, SystemExit):
            return "your code called exit() or sys.exit(), which stops the whole program. Return a value instead."
        if isinstance(error, AssertionError):
            return "an assert in your code failed."
        if isinstance(error, RuntimeError) and "input()" in text:
            return "your function calls input(). In this lab, functions get their values as parameters instead."
        if name == "MemoryError":
            return "your code built something too big for memory. Is a list growing inside a loop that never ends?"
        return None
    got = attempt.got
    if got is None and expected is not None:
        return ("your function returned None. Check that it ends with `return <value>`, and that "
                "the return isn't inside an if or a loop that didn't run.")
    if inspect.isgenerator(got):
        return "you returned a generator, not a list. Did you forget the square brackets around a comprehension?"
    if type(got).__name__ in ("map", "filter", "zip", "dict_keys", "dict_values", "dict_items", "range"):
        return f"you returned a {type(got).__name__} object. Wrap it in list(...) to get a list."
    if callable(got) and not callable(expected):
        return "you returned a function itself. Did you forget the ( ) when calling it?"
    if isinstance(got, str) and not isinstance(expected, str):
        return f"your function returned text (a str). The spec asks for {type_name(expected)}: return the value itself."
    if is_number(expected) and is_number(got) and type(got) is not type(expected):
        if isinstance(expected, int):
            return ("the spec asks for an int, but you returned a float. Look for /, math.pow, "
                    "math.sqrt or float() in your code.")
        return ("the spec asks for a float. In Python 3, a / b always gives a float, but a // b, round(x) and "
                "statistics.mean of True/False values can give an int: wrap the result in float(...).")
    if isinstance(expected, list) and isinstance(got, tuple):
        return "you returned a tuple; the spec asks for a list."
    if isinstance(expected, dict) and isinstance(got, dict) and expected and got:
        want, have = {type(k) for k in expected}, {type(k) for k in got}
        if want != have:
            mine, theirs = next(iter(got)), next(iter(expected))
            return (f"your dict keys are {type(mine).__name__} values such as {mine!r}, but the spec uses "
                    f"{type(theirs).__name__} keys such as {theirs!r}. Use the value itself as the key.")
    return None


# ============================================================================= one call
def _copy_argument(value: Any) -> Any:
    """A private copy of one argument (fast for images and lists of images)."""
    if type(value) is list:
        if all(type(item) in _ATOMS for item in value):
            return value[:]
        if all(type(item) is list and (not item or type(item[0]) in _ATOMS) for item in value):
            return [item[:] for item in value]
    if isinstance(value, _ATOMS) or callable(value):
        return value
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def _no_input(prompt: object = "") -> str:
    raise RuntimeError("input() waits for someone to type, so it can't run inside a check")


def call_text(name: str, args: tuple, kwargs: dict | None = None) -> str:
    """'squared_distance([0, 0], [3, 4])' with long arguments shortened."""
    parts = [short(a, 45) for a in args] + [f"{k}={short(v, 30)}" for k, v in (kwargs or {}).items()]
    return f"{name}({', '.join(parts)})"


def run_call(fn: Any, args: tuple = (), kwargs: dict | None = None, *, show: str = "",
             track: bool = True) -> Attempt:
    """Call fn(*args, **kwargs) safely: on copies, with printing captured, errors caught and timed.

    track=True gives the function copies of its arguments and records any argument it changed.
    """
    kwargs = kwargs or {}
    passed_args = tuple(_copy_argument(a) for a in args) if track else tuple(args)
    passed_kwargs = {k: _copy_argument(v) for k, v in kwargs.items()} if track else dict(kwargs)
    attempt = Attempt(args=passed_args, kwargs=passed_kwargs, show=show)
    out, err = io.StringIO(), io.StringIO()
    real_input = builtins.input
    builtins.input = _no_input                 # input() would wait forever inside a check
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            attempt.got = fn(*passed_args, **passed_kwargs)
    except KeyboardInterrupt as error:
        attempt.error, attempt.interrupted = error, True
    except BaseException as error:          # also SystemExit, if the student's code calls exit()
        attempt.error = error
    finally:
        builtins.input = real_input
    attempt.seconds = time.perf_counter() - start
    attempt.printed, attempt.stderr = out.getvalue(), err.getvalue()
    if track:
        names = _parameter_names(fn)
        pairs = [(names[i] if i < len(names) else f"argument {i + 1}", before, after)
                 for i, (before, after) in enumerate(zip(args, passed_args))]
        pairs += [(key, before, passed_kwargs[key]) for key, before in kwargs.items()]
        for name, before, after in pairs:
            if not isinstance(before, _ATOMS) and not _same(before, after):
                attempt.changed.append(name)
                if not attempt.change_note:
                    nested = changed_where(name, before, after)
                    where = first_difference(before, after)
                    attempt.change_note = nested or (f"{name} was {short(before, 40)}, after the call it is "
                                                     f"{short(after, 40)}" + (f" ({where})" if where else ""))
    return attempt


# ============================================================================= results
@dataclass
class Failure:
    """The first failing check of a run, ready to print."""
    case: str
    expected: str = ""
    got: str = ""
    error: BaseException | None = None
    cause: str | None = None
    next_step: str | None = None
    difference: str | None = None
    attempt: Attempt | None = None
    extra: list = field(default_factory=list)       # (label, text) lines printed after the TEST block
    number: int = 0                                  # TEST n (the n-th check of the run)
    title: str = ""                                  # TEST n: <title> (default: case)
    inputs: list = field(default_factory=list)      # (label, text) lines under the title
    expected_label: str = "Expected"


@dataclass
class TestRecord:
    """A passing test, kept so that a PASS can show what was checked."""
    number: int
    title: str
    inputs: list
    expected_label: str
    expected: str
    got: str
    highlight: bool = False                          # Case(highlight=True)
    titled: bool = False                             # the case has its own title


@dataclass
class Report:
    task: Task
    status: str = NOT_YET
    passed: int = 0
    total: int = 0
    failure: Failure | None = None
    message: str = ""                                # for NOT STARTED and setup problems
    not_run: int = 0                                 # planned tests that did not run (after a stop), for NOT YET


class _AuthorProblem(Exception):
    """Something is wrong with the task data (not with the student's code)."""


class CheckContext:
    """What a check run knows and records. Custom hooks receive it as `ctx`:

        attempt = ctx.call(fn, arg1, arg2, show="fn(arg1, arg2)")   # safe call -> Attempt
        ctx.expect(attempt, expected)                                 # one check: compare and record
        ctx.require(ok, case=..., expected=..., got=..., cause=...)    # one check from any condition
        ctx.note("Mutants caught: 3 of 4")                            # an extra line under the result
        ctx.not_started("Write the class first.")                      # report NOT STARTED and stop
        with ctx.temp_dir() as folder: ...                            # a scratch folder, deleted after
        ctx.reference("majority_vote")                                # a reference object of this lab
        ctx.judge_tests(test_fn, reference=..., mutants={...}, ...)   # judge a student's test function
    """

    def __init__(self, task: Task, obj: Any, lab_cases: Any = None, *, verbose: bool = False) -> None:
        self.task = task
        self.obj = obj
        self.lab_cases = lab_cases
        self.verbose = verbose
        self.passed = 0
        self.total = 0
        self.failure: Failure | None = None
        self.records: list[TestRecord] = []              # passing tests, for the PASS summary
        self.notes: list[str] = []
        self.results: list[tuple[Attempt, Any]] = []     # (attempt, expected) for property checks
        self.tracked: list[Attempt] = []
        self.stopped = False
        self.not_started_message: str | None = None
        self.started = time.perf_counter()
        self.typed_seen = False                          # a typed case ran: the return-type check is planned

    # ------------------------------------------------------------------ for hooks
    def call(self, fn: Any, *args: Any, show: str | None = None, track: bool = True, **kwargs: Any) -> Attempt:
        """Run fn(*args, **kwargs) safely and return the Attempt (got, error, printed, seconds)."""
        name = getattr(fn, "__name__", "your function")
        attempt = run_call(fn, args, kwargs, show=show or call_text(name, args, kwargs), track=track)
        if track:
            self.tracked.append(attempt)
        if attempt.interrupted:
            self.stopped = True
        return attempt

    def expect(self, attempt: Attempt, expected: Any, *, compare: Compare | None = None,
               detectors: list | None = None, next_step: str | None = None, typed: bool = False) -> bool:
        """One check: does the attempt give `expected`? Records PASS or the failure.

        typed=True: the result is the task's return value, so the return-type property applies to it.
        """
        self.total += 1
        number = self.total
        attempt.expected = expected
        title = attempt.title or attempt.show
        limit = attempt.preview or 70
        if typed and not isinstance(expected, (Raises, Described)):
            self.typed_seen = True

        def failure(**fields: Any) -> Failure:
            return Failure(case=attempt.show, attempt=attempt, number=number, title=title,
                           inputs=list(attempt.inputs), expected_label=attempt.expected_label, **fields)

        def passed(shown_expected: str, shown_got: str) -> bool:
            self.passed += 1
            self.records.append(TestRecord(number, title, list(attempt.inputs), attempt.expected_label,
                                           shown_expected, shown_got, attempt.headline, bool(attempt.title)))
            return True

        if attempt.interrupted:
            self._record(failure(error=attempt.error), detectors)
            return False
        if isinstance(expected, Raises):
            if expected.matches(attempt.error):
                return passed(expected.describe(), f"raised {attempt.error_text}")
            if attempt.error is not None:
                got = f"raised {attempt.error_text}"
            else:
                got = f"returned {short(attempt.got, limit)} (no error raised)"
            found = failure(expected=expected.describe(), got=got, next_step=next_step)
            if attempt.error is not None:
                place = your_line(attempt.error, self.obj)
                if place:
                    found.extra.append(place)
            self._record(found, detectors)
            return False
        if attempt.error is not None:
            self._record(failure(error=attempt.error, expected=_shown(expected, limit), next_step=next_step),
                         detectors)
            return False
        if isinstance(expected, Described):              # words, not a value: there is nothing to compare with
            self._record(failure(expected=expected.text, got=short(attempt.got, limit), next_step=next_step),
                         detectors)
            return False
        compare = compare or exact()
        try:
            same = compare.same(attempt.got, expected)
        except Exception:
            same = False
        if same:
            if typed:
                self.results.append((attempt, expected))
            return passed(short(expected, limit) + (f" {compare.note}" if compare.note else ""),
                          short(attempt.got, limit))
        shown_expected, shown_got = preview_pair(expected, attempt.got, limit)
        if compare.note:
            shown_expected += f" {compare.note}"
        self._record(failure(expected=shown_expected, got=shown_got,
                             difference=first_difference(expected, attempt.got), next_step=next_step),
                     detectors)
        return False

    def require(self, ok: bool, *, case: str, expected: str, got: str, cause: str | None = None,
                next_step: str | None = None, title: str | None = None, inputs: list | None = None,
                expected_label: str = "Expected", attempt: Attempt | None = None) -> bool:
        """One check from any condition; on failure, print these texts (title: "TEST n: <title>").

        inputs          (label, text) lines under the title, e.g. [("Command line:", "python digits.py inspect")]
                        (default: "Call: <case>" when there is a title)
        expected_label  the label of the expected line, e.g. "Expected output"
        attempt         the Attempt this test is about: if its call raised, the failure adds "Your line N" and,
                        when `cause` is None, the likely cause of that error
        """
        self.total += 1
        if inputs is None:
            inputs = [("Call:", case)] if title else []
        inputs = [(label if label.endswith(":") else label + ":", text) for label, text in inputs]
        if ok:
            self.passed += 1
            self.records.append(TestRecord(self.total, title or case, inputs, expected_label, str(expected), str(got),
                                           False, bool(title)))
            return True
        failure = Failure(case=case, expected=expected, got=got, cause=cause, next_step=next_step,
                          title=title or case, inputs=inputs, expected_label=expected_label)
        if attempt is not None and attempt.error is not None and not attempt.interrupted:
            place = your_line(attempt.error, self.obj)
            if place:
                failure.extra.append(place)
            if cause is None:
                failure.cause = likely_cause(attempt, attempt.expected)
        self._record(failure, None)
        return False

    def note(self, text: str) -> None:
        """An extra line printed under the result."""
        self.notes.append(text)

    def reference(self, name: str) -> Any:
        """The reference object `name` of this lab (e.g. the correct function a test is judged on)."""
        lab = self.lab_cases.lab if self.lab_cases is not None else current_lab()
        return reference_registry.get(lab, name) if lab else None

    def not_started(self, message: str | None = None) -> None:
        """Report NOT STARTED (with an optional message instead of the standard one) and stop."""
        self.not_started_message = message or ""
        self.stopped = True

    @contextlib.contextmanager
    def temp_dir(self, *, chdir: bool = False):
        """A fresh scratch folder (a Path), deleted afterwards; chdir=True also works inside it."""
        import shutil, tempfile          # imported here: only file tasks need them

        folder = Path(tempfile.mkdtemp(prefix="labkit_check_"))
        previous = Path.cwd()
        try:
            if chdir:
                os.chdir(folder)
            yield folder
        finally:
            if chdir:
                os.chdir(previous)
            shutil.rmtree(folder, ignore_errors=True)

    def judge_tests(self, test_fn: Any, *, reference: Any, mutants: dict, must_catch: int,
                    param: str, global_name: str | None = None, min_asserts: int = 3,
                    must_fail_on: dict | None = None, all_of: dict | None = None) -> None:
        """Judge a student's test function (e.g. test_majority_vote(vote_fn)).

        The test must pass on `reference` and fail on at least `must_catch` of `mutants`
        ({"a vote that returns the largest label": buggy_fn, ...}). It must call `param`,
        not the global function `global_name`, and contain at least `min_asserts` asserts.
        must_fail_on  {label: buggy_fn}: the test must fail on each of these (one titled test each)
        all_of        {description: buggy_fn}: one titled test, "catches all N of these buggy versions"
        Every step is a titled test ("TEST n: Your test passes on a correct ..."), on PASS too.
        """
        name = getattr(test_fn, "__name__", "your test")
        target = global_name or param
        asserts, tuple_line = read_asserts(test_fn)
        if asserts is not None:
            if tuple_line is not None:
                self.require(False, case=f"{name}, line {tuple_line}", title="Every assert can fail (no assert (a, b))",
                             expected="an assert that can fail", got="assert (...): a tuple, which is always true",
                             cause=("Python reads assert (a, b) as 'assert this tuple', and a tuple that isn't "
                                    "empty is always true. Write assert a == b, \"message\" with no brackets."))
            else:
                self.require(True, case=name, title="Every assert can fail (no assert (a, b))",
                             expected="asserts that can fail", got="no assert (a, b)")
            self.require(asserts >= min_asserts, case=f"{name}", title=f"Your test has at least {min_asserts} asserts",
                         expected=f"at least {min_asserts} asserts", got=f"{asserts} assert(s)",
                         cause=f"add asserts that call {param}(...) and compare with the right answer.")
        globals_ = getattr(test_fn, "__globals__", {})
        missing = object()
        saved = globals_.get(global_name, missing) if global_name else missing
        calls_to_global = []

        def recorder(*args, **kwargs):
            calls_to_global.append(args)
            return reference(*args, **kwargs)

        if global_name:
            globals_[global_name] = recorder
        try:
            attempt = self.call(test_fn, reference, show=f"{name}({param}=the correct version)", track=False)
        finally:
            if global_name:
                if saved is missing:
                    globals_.pop(global_name, None)
                else:
                    globals_[global_name] = saved
        if attempt.interrupted:
            self.expect(attempt, None)
            return
        if global_name:
            self.require(not calls_to_global, case=attempt.show, title=f"Your test calls {param}, not {global_name}",
                         expected=f"your test calls {param}(...)",
                         got=(f"your test calls {global_name}(...) directly" if calls_to_global
                              else f"it calls {param}(...)"),
                         cause=f"your test calls {global_name} directly; call {param} instead.")
            if calls_to_global:
                return
        title = f"Your test passes on a correct {target}"
        if attempt.error is not None:
            failure = Failure(case=attempt.show, expected="your test passes on the correct version",
                              got=f"it failed: {attempt.error_text or type(attempt.error).__name__}", attempt=attempt,
                              title=title, inputs=[("Call:", attempt.show)],
                              cause="one of your asserts expects a wrong answer. Work that input out by hand.")
            place = your_line(attempt.error, test_fn)
            if place:
                failure.extra.append(place)
            self.total += 1
            self._record(failure, None)
            return
        self.require(True, case=attempt.show, title=title, expected="no assert fails", got="no assert failed")

        def fails_on(buggy: Any) -> bool | None:
            result = self.call(test_fn, buggy, show=f"{name}(a buggy version)", track=False)
            return None if result.interrupted else result.error is not None

        for label, buggy in (must_fail_on or {}).items():
            caught = fails_on(buggy)
            if caught is None:
                return
            self.require(caught, case=f"{name}({param}={label})", title=f"Your test fails on {label}",
                         expected="an assert fails", got="an assert fails" if caught else "every assert passed",
                         cause=f"which input makes {label} give a wrong answer? Add an assert for it.")
        if all_of:
            survivors = []
            for description, buggy in all_of.items():
                caught = fails_on(buggy)
                if caught is None:
                    return
                if not caught:
                    survivors.append(description)
            self.require(not survivors, case=f"{name} against {len(all_of)} buggy versions",
                         title=f"Your test catches all {len(all_of)} of these buggy versions",
                         expected=f"catches {len(all_of)} of {len(all_of)}",
                         got=f"catches {len(all_of) - len(survivors)}" + (
                             f" (survived: {'; '.join(survivors)})" if survivors else ""),
                         cause="add an assert for each survivor: which input makes it give a wrong answer?")
        caught, survived = 0, []
        for description, mutant in mutants.items():
            result = fails_on(mutant)
            if result is None:
                return
            if result:
                caught += 1
            else:
                survived.append(description)
        summary = f"Mutants caught: {caught} of {len(mutants)}"
        if survived:
            summary += " · survived: " + "; ".join(survived)
        self.note(summary)
        self.require(caught >= must_catch, case=f"{name} against {len(mutants)} buggy versions",
                     title=f"Your test catches at least {must_catch} of {len(mutants)} buggy versions",
                     expected=f"catches at least {must_catch}",
                     got=f"catches {caught}" + (f" (survived: {'; '.join(survived)})" if survived else ""),
                     cause="add an assert for each survivor: which input makes that buggy version give a wrong answer?")

    # ------------------------------------------------------------------ inside the engine
    def _record(self, failure: Failure, detectors: list | None) -> None:
        if self.failure is not None:
            return                                   # only the first failure is shown
        failure.number = failure.number or self.total
        failure.title = failure.title or failure.case
        attempt = failure.attempt
        if failure.cause is None and attempt is not None and not attempt.interrupted:
            for detector in list(detectors or []) + list(self.task.detectors):
                try:
                    hit = bool(detector.test(attempt))
                except Exception:                     # a rule that doesn't apply to this input
                    hit = False
                if hit:
                    failure.cause = detector.message
                    failure.next_step = failure.next_step or detector.next_step
                    break
            if failure.cause is None:
                failure.cause = likely_cause(attempt, attempt.expected)
        self.failure = failure

    def out_of_time(self) -> bool:
        return time.perf_counter() - self.started > TIME_LIMIT_S


def parse_source(source: str) -> ast.Module:
    """ast.parse() of student source, without printing Python's SyntaxWarnings again.

    The student already saw them when the cell ran (e.g. "assertion is always true" for assert (a, b));
    a check that re-parses the same code must not print them a second time.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)      # invalid escape sequences on Python 3.10-3.11
        return ast.parse(source)


def read_asserts(test_fn: Any) -> tuple[int | None, int | None]:
    """(number of asserts, line of the first `assert (a, b)`) from the source, or (None, None)."""
    try:
        lines, first = inspect.getsourcelines(test_fn)
        tree = parse_source(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError):
        return None, None
    count, tuple_line = 0, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            count += 1
            if isinstance(node.test, ast.Tuple) and tuple_line is None:
                tuple_line = node.lineno + max(first, 1) - 1
    return count, tuple_line


_read_asserts = read_asserts          # the old private name (Lab 2's cases file imported it)


# ============================================================================= running a task
def _type_ok(value: Any, returns: Any) -> bool:
    kinds = returns if isinstance(returns, tuple) else (returns,)
    for kind in kinds:
        if kind is None:
            kind = type(None)
        if kind is int and isinstance(value, bool):
            continue
        if isinstance(value, kind):
            return True
    return False


def _type_words(returns: Any) -> str:
    kinds = returns if isinstance(returns, tuple) else (returns,)
    words = []
    for kind in kinds:
        name = "None" if kind in (None, type(None)) else kind.__name__
        words.append(name if name == "None" else ("an " if name[0] in "aeiou" else "a ") + name)
    return " or ".join(words)


def _shown(value: Any, limit: int = 70) -> str:
    """A one-line preview of an expected value ("" for None; a Described expectation as it is)."""
    if isinstance(value, Described):
        return value.text
    return "" if value is None else short(value, limit)


def case_inputs(task: Task, case: Any, args: tuple, show: str) -> list[tuple[str, str]]:
    """The (label, text) lines printed under "TEST n: <title>" for one case.

    Case.show_inputs if given; else the arguments labelled with Task.input_labels (toy cases whose call
    is not spelled out with `show`); else "Call: <show>" when the case has its own title (otherwise the
    title already is the call).
    """
    if case.show_inputs:
        return [(f"{label}:", str(text)) for label, text in case.show_inputs.items()]
    if case.call is None and case.show is None and task.input_labels:
        lines = []
        for i, value in enumerate(args):
            label = task.input_labels[i] if i < len(task.input_labels) else f"Argument {i + 1}"
            lines.append((f"{label}:", short(value, 60)))
        lines += [(f"{key}:", short(value, 40)) for key, value in case.kwargs.items()]
        return lines
    if case.title:
        return [("Call:", show)]
    return []


def _run_cases(ctx: CheckContext) -> None:
    task, obj = ctx.task, ctx.obj
    for index, case in enumerate(task.cases):
        if ctx.stopped:
            return
        if ctx.out_of_time():
            ctx.total += len(task.cases) - index
            ctx.note(f"Stopped after {TIME_LIMIT_S:.0f} s: the remaining cases were not run.")
            return
        try:
            args = case.resolve_args()
            expected = ctx.lab_cases.expected(task, case, index) if ctx.lab_cases else case.expected
            facts = ctx.lab_cases.facts(task, case, index) if ctx.lab_cases else dict(case.facts)
        except (LookupError, OSError, ValueError) as problem:
            raise _AuthorProblem(f"the data for case {index} is missing ({problem})") from None
        show = case.show or call_text(task.name, args, case.kwargs)
        inputs = case_inputs(task, case, args, show)
        if case.call is not None:
            attempt = ctx.call(case.call, obj, show=show, track=False)
        else:
            attempt = ctx.call(obj, *args, show=show, **case.kwargs)
        attempt.facts = facts
        attempt.title, attempt.inputs, attempt.headline = case.title or "", inputs, case.highlight
        attempt.expected_label = case.expected_label or task.expected_label
        attempt.preview = getattr(case, "preview", None)
        ctx.expect(attempt, expected, compare=case.compare, typed=case.call is None)


def _property_checks(ctx: CheckContext) -> None:
    task = ctx.task
    if ctx.stopped:
        return
    if task.returns is not None and ctx.results:
        wrong = [(a, e) for a, e in ctx.results if not isinstance(e, Raises) and not _type_ok(a.got, task.returns)]
        if wrong:
            attempt, expected = wrong[0]
            ctx.total += 1
            ctx._record(Failure(case=attempt.show, expected=f"{_type_words(task.returns)}, like {short(expected)}",
                                got=f"{short(attempt.got)} ({type_name(attempt.got)})", attempt=attempt,
                                title=f"Your result is {_type_words(task.returns)}",
                                inputs=[("Call:", attempt.show)]), None)
        else:
            ctx.total, ctx.passed = ctx.total + 1, ctx.passed + 1
    if task.inputs_unchanged and any(_has_containers(a) for a in ctx.tracked):
        changed = [a for a in ctx.tracked if a.changed and not a.interrupted]
        if changed:
            attempt = changed[0]
            ctx.total += 1
            ctx._record(Failure(
                case=attempt.show, expected="your inputs unchanged (the check compares them before and after)",
                got=f"your function changed {', '.join(attempt.changed)}", attempt=attempt,
                title="Your function leaves its inputs unchanged", inputs=[("Call:", attempt.show)],
                extra=[("Where:", attempt.change_note)] if attempt.change_note else [],
                cause=("something like .append(), .sort(), .pop() or x[i] = ... changed a list you were "
                       "given. Build a new list instead and leave the inputs as they were.")), None)
        else:
            ctx.total, ctx.passed = ctx.total + 1, ctx.passed + 1


def _has_containers(attempt: Attempt) -> bool:
    return any(not isinstance(a, _ATOMS) for a in list(attempt.args) + list(attempt.kwargs.values()))


def run_task(task: Task, obj: Any, lab_cases: Any = None, *, verbose: bool = False) -> tuple[Report, CheckContext]:
    """Run every check of `task` on `obj` without printing; returns (report, context)."""
    report = Report(task)
    ctx = CheckContext(task, obj, lab_cases, verbose=verbose)
    if task.kind in ("function", "class", "test-function") and is_stub(obj):
        report.status = NOT_STARTED
        return report, ctx
    if task.stub_fingerprint and untouched_stub(obj, task.stub_fingerprint):
        report.status, report.message = NOT_STARTED, _untouched_message(task)
        return report, ctx
    _run_cases(ctx)
    hook_tests = None
    if task.custom is not None and not ctx.stopped:
        before = ctx.total
        try:
            task.custom(obj, ctx)
        except Exception as error:
            if student_line(error) is None:
                raise                                 # a problem in the hook itself, not in student code
            ctx.total += 1                            # student code raised inside the hook
            attempt = Attempt(error=error, show=f"{task.name} (extra checks)")
            ctx._record(Failure(case=attempt.show, error=error, attempt=attempt), None)
        hook_tests = ctx.total - before
    _property_checks(ctx)
    report.passed, report.total, report.failure = ctx.passed, ctx.total, ctx.failure
    if not ctx.stopped:
        # The same total on every run: planned tests that did not run are reported as "not run".
        if task.returns is not None and ctx.typed_seen and not ctx.results:
            report.not_run += 1                       # the return-type check needs one passing result
        if hook_tests is not None and task.custom_tests > hook_tests:
            report.not_run += task.custom_tests - hook_tests
    if ctx.not_started_message is not None:
        report.status, report.message = NOT_STARTED, ctx.not_started_message
    elif ctx.failure is not None or ctx.total == 0 or ctx.passed < ctx.total:
        report.status = NOT_YET
        if ctx.failure is None:
            ctx.failure = report.failure = Failure(case=task.name, expected="every test run",
                                                   got="some tests did not run", number=ctx.total,
                                                   title="Every test runs")
    else:
        report.status = PASS
    return report, ctx


# ============================================================================= printing
def qualified(task: Task) -> str:
    """How the student names the task's object: 'build_parser', or 'digits.build_parser' when it lives in a module
    (Task.home): rebinding only the notebook name would leave the module's own function unchanged."""
    return task.name if task.home == "__main__" else f"{task.home}.{task.name}"


def _not_started_line(task: Task, message: str = "") -> str:
    if not message:
        message = ("Replace `pass` with your code, run the cell that defines it, then run this check."
                   if task.home == "__main__" else
                   f"Replace `pass` with your code in {task.home}.py, save the file, then run this check.")
    return f"{task.id} {task.name}: NOT STARTED. " + message


def _untouched_message(task: Task) -> str:
    if task.home == "__main__":
        return "The stub still has only the code it came with: complete it, run its cell, then run this check."
    return (f"The stub still has only the code it came with: complete {task.name} in {task.home}.py, save the "
            "file, then run this check.")


def fingerprint(source: str) -> str:
    """A short fingerprint of a function's code (the first def or class in `source`), the same whatever its comments
    and docstring: sha256 of ast.unparse(body without the docstring), first 16 hex digits (the same text on Python
    3.10 to 3.14). Used for Task(stub_fingerprint=...)."""
    import hashlib

    node = parse_source(textwrap.dedent(source)).body[0]
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    code = ast.unparse(ast.Module(body=body, type_ignores=[]))
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _current_source(obj: Any) -> str | None:
    """The source of a function or class: from its module's file as saved now (a module the student edits in an
    editor), or else from the notebook cell that defined it."""
    target = getattr(obj, "__func__", obj)
    name = getattr(target, "__name__", None)
    module = sys.modules.get(getattr(target, "__module__", "") or "")
    path = getattr(module, "__file__", None)
    if name and path and str(path).endswith(".py") and is_student_file(str(path)):
        try:
            text = Path(path).read_text(encoding="utf-8")
            for node in parse_source(text).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                    return ast.get_source_segment(text, node)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            return None
        return None
    try:
        return inspect.getsource(target)
    except (OSError, TypeError):
        return None


def untouched_stub(obj: Any, stub_fingerprint: str) -> bool:
    """True if obj's code still has the fingerprint of the stub as shipped (only comments may have changed)."""
    try:
        source = _current_source(obj)
        return source is not None and fingerprint(source) == stub_fingerprint
    except (SyntaxError, IndexError, AttributeError, ValueError):
        return False


def _quality_line(obj: Any, ctx: CheckContext) -> str | None:
    if obj is None or not ctx.task.quality:
        return None
    doc = getattr(obj, "__doc__", None)
    parts = [f"docstring {'yes' if isinstance(doc, str) and doc.strip() else 'no'}",
             f"type hints {_hints_status(obj)}"]
    if any(_has_containers(a) for a in ctx.tracked):
        changed = any(a.changed for a in ctx.tracked)
        parts.append(f"inputs unchanged {'no' if changed else 'yes'}")
    return "Quality: " + " · ".join(parts)


def _hints_status(obj: Any) -> str:
    functions = _own_functions(obj) if isinstance(obj, type) else [getattr(obj, "__func__", obj)]
    wanted = found = 0
    for function in functions:
        code = getattr(function, "__code__", None)
        if code is None:
            continue
        names = [n for n in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount] if n not in ("self", "cls")]
        try:
            annotations = dict(function.__annotations__)
        except Exception:               # Python 3.14 evaluates hints lazily; a broken hint still counts
            return "yes"
        wanted += len(names) + 1
        found += sum(1 for n in names if n in annotations) + (1 if "return" in annotations else 0)
    if wanted == 0 or found == 0:
        return "no"
    return "yes" if found == wanted else "partly"


def _speed_note(task: Task, obj: Any, lab: int | None) -> str | None:
    if task.speed is None or not callable(obj):
        return None
    try:
        args = task.speed.resolve_args()
        copies = tuple(_copy_argument(a) for a in args)
        with contextlib.redirect_stdout(io.StringIO()):
            seconds = best_time(obj, *copies, budget_s=0.1)
        limit = task.speed.budget_s * machine_factor()
        if seconds <= limit:
            return None
        text = f"Correct, but slow: {format_duration(seconds)} per call"
        ref = reference_registry.get(lab, task.reference_name) if lab else None
        if callable(ref) and ref is not obj:
            with contextlib.redirect_stdout(io.StringIO()):
                ref_seconds = best_time(ref, *copies, budget_s=0.1)
            if seconds <= 2.5 * ref_seconds:      # about as fast as the reference here: no note
                return None
            text += f" (reference ~{format_duration(ref_seconds)} here)"
        return f"Speed note: {text}. {task.speed.note}".rstrip()
    except Exception:
        return None


def style_notes(obj: Any) -> list[str]:
    """Optional style remarks found with ast (never affect PASS): range(len(...)), while counters, ..."""
    try:
        lines, first = inspect.getsourcelines(obj)
        tree = parse_source(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError):
        return []
    offset = max(first, 1) - 1
    notes = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0) + offset
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
                and node.args and isinstance(node.args[-1], ast.Call)
                and isinstance(node.args[-1].func, ast.Name) and node.args[-1].func.id == "len"):
            notes.append((line, "uses range(len(...)): zip() or enumerate() may read better."))
        elif isinstance(node, ast.While) and any(_is_counter_step(n) for n in ast.walk(node)):
            notes.append((line, "uses a while loop with a counter: a for loop may read better."))
        elif (isinstance(node, ast.Compare) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
              and any(isinstance(c, ast.Constant) and (c.value is None or c.value is True or c.value is False)
                      for c in node.comparators)):
            notes.append((line, "compares with == None, True or False: write `is None`, or test the value itself."))
    notes.sort()
    return [f"Style note (optional): line {line} {text}" for line, text in notes[:3]]


def _is_counter_step(node: ast.AST) -> bool:
    """`i += 1` (any augmented assignment), or `i = i + 1` / `i = i - 1`: a while-loop counter."""
    if isinstance(node, ast.AugAssign):
        return True
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return False
    value, name = node.value, node.targets[0].id
    return (isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub))
            and isinstance(value.left, ast.Name) and value.left.id == name
            and isinstance(value.right, ast.Constant) and is_number(value.right.value))


def test_block(number: int, title: str, rows: list, status: str, width: int = 0) -> list[str]:
    """One test, as printed under a check:

       TEST 2: Three of four predictions right
          Predictions:        [1, 2, 3, 4]
          Expected accuracy:  0.75
          Your result:        1.0
          NOT YET
    """
    rows = [(label, text) for label, text in rows if text is not None and str(text) != ""]
    width = max([LABEL_WIDTH, width] + [len(label) + 2 for label, _ in rows])
    lines = [f"{INDENT}TEST {number}: {title}"]
    lines += [INDENT * 2 + (label + " ").ljust(width) + str(text) for label, text in rows]
    lines.append(INDENT * 2 + status)
    return lines


def pass_summary(task: Task, ctx: CheckContext, total: int) -> list[str]:
    """Up to task.show_on_pass passing tests in full, then "... and N more tests passed."."""
    limit = max(0, int(task.show_on_pass or 0))
    records = ctx.records
    chosen = ([r for r in records if r.highlight] or [r for r in records if r.titled] or records[:2])[:limit]
    lines: list[str] = []
    blocks = [(record, list(record.inputs) + [(f"{record.expected_label}:", record.expected),
                                              ("Your result:", record.got)]) for record in chosen]
    width = max([len(label) + 2 for _, rows in blocks for label, _ in rows], default=0)
    for record, rows in blocks:
        lines += test_block(record.number, record.title, rows, PASS, width)
    more = total - len(chosen)
    if chosen and more > 0:
        lines.append(INDENT + f"… and {more} more test{'s' if more != 1 else ''} passed.")
    return lines


def format_report(report: Report, ctx: CheckContext, *, lab: int | None, verbose: bool = False,
                  style: bool = False, dependency: tuple[str, str] | None = None) -> list[str]:
    """Turn a report into the exact lines printed under a check cell."""
    task, obj = report.task, ctx.obj
    head = f"{task.id} {task.name}"
    if report.status == NOT_STARTED:
        return [_not_started_line(task, report.message)]
    lines: list[str] = []
    if report.status == PASS:
        lines.append(f"{head}: PASS ({report.passed}/{report.total})")
        lines += pass_summary(task, ctx, report.total)
        if task.practise:
            lines.append(INDENT + "You practised: " + task.practise.rstrip(".") + ".")
        quality = _quality_line(obj, ctx)
        if quality:
            lines.append(INDENT + quality)
        for note in ctx.notes:
            lines.append(INDENT + note)
        speed = _speed_note(task, obj, lab)
        if speed:
            lines.append(INDENT + speed)
        if obj is not None and lab and obj is reference_registry.get(lab, task.reference_name):
            lines.append(INDENT + f"Note: this is the reference from rescue(\"{task.id}\"). "
                                  f"Come back to {task.id} when you're ready.")
    else:
        failure = report.failure or Failure(case=task.name)
        attempt = failure.attempt
        number = failure.number or report.total or 1
        title = failure.title or failure.case
        inputs = list(failure.inputs)
        expected_row = (f"{failure.expected_label}:", failure.expected)
        if attempt is not None and attempt.interrupted:
            lines.append(f"{head}: NOT YET (you stopped the check)")
            lines += test_block(number, title, inputs + [("Stopped after:", format_duration(attempt.seconds))],
                                NOT_YET)
            lines.append(report_field("Likely cause:", "a check normally takes under 1 s, so your code may be "
                                      "stuck in a loop that never ends. Look at each while loop: does its "
                                      "condition ever become False?"))
            lines.append(report_field("Next step:", "fix the loop, run the cell with def again, then this check."))
            return lines
        cause, next_step = failure.cause, failure.next_step
        if dependency is not None:
            cause, next_step = dependency
        if failure.error is not None:
            lines.append(f"{head}: NOT YET (your code raised an error)")
            error_text = attempt.error_text if attempt is not None else f"{type(failure.error).__name__}: {failure.error}"
            if len(error_text) > 160:
                error_text = error_text[:157] + "…"
            lines += test_block(number, title, inputs + [expected_row, ("Your result:", f"raised {error_text}")],
                                NOT_YET)
            place = your_line(failure.error, obj)
            if place:
                lines.append(report_field(*place))
            else:
                definition = def_line(obj)
                if definition:
                    lines.append(report_field("Your def line:", definition))
            if cause:
                lines.append(report_field("Likely cause:", cause))
            if verbose:
                lines.append(report_field("Next step:", "read the traceback below from the bottom up, fix your "
                                          "line, run the cell with def again, then this check."))
            else:
                lines.append(report_field("Next step:", next_step or (
                    f"open Hint 2, or run check(\"{task.id}\", {qualified(task)}, verbose=True) for the full "
                    "traceback.")))
        else:
            total = report.total + report.not_run
            not_run = f", {report.not_run} not run" if report.not_run else ""
            lines.append(f"{head}: NOT YET ({report.passed} of {total} tests passed{not_run})")
            rows = inputs + [expected_row, ("Your result:", failure.got)]
            if failure.difference:
                rows.append(("Where:", failure.difference))
            lines += test_block(number, title, rows, NOT_YET)
            for label, text in failure.extra:
                lines.append(report_field(label, text))
            if cause:
                lines.append(report_field("Likely cause:", cause))
            lines.append(report_field("Next step:", next_step or (
                "fix it and run the cell with def again, then this check. Or open Hint 1 below.")))
        for note in ctx.notes:
            lines.append(INDENT + note)
        if attempt is not None and attempt.printed:
            printed = attempt.printed.rstrip("\n").split("\n")
            if verbose:
                lines.append(INDENT + f"Your function printed {len(printed)} line(s) during this test:")
                lines += [INDENT + "  " + line for line in printed[:20]]
            else:
                lines.append(report_field("Printed:", f"{len(printed)} line(s) from your function, hidden "
                                          "(verbose=True shows them)."))
        if verbose and failure.error is not None:
            lines.append("")
            lines.append("Full traceback (read it from the bottom up: the last line is the error,")
            lines.append("the lines above it show where it happened):")
            lines += format_traceback(failure.error)
    if style and obj is not None:
        lines += [INDENT + note for note in style_notes(obj)]
    return lines


# ============================================================================= finding things
def _lab_cases():
    """The cases of the current lab, or None after printing why there are none."""
    lab = current_lab()
    if lab is None:
        say("Run the setup cell first (the first code cell): it tells labkit which lab this is.")
        return None
    lab_cases = lab_registry.load(lab)
    if lab_cases is None or not lab_cases.tasks:
        say(f"This kit has no checks for Lab {lab}. Is LAB right in the setup cell?")
        return None
    return lab_cases


def _notebook_namespace() -> dict:
    main = sys.modules.get("__main__")
    return vars(main) if main is not None else {}


def find_object(task: Task) -> Any:
    """Look up the student's object for `task` by name: in the notebook, or in task.home."""
    if task.home != "__main__":
        module = sys.modules.get(task.home) or _notebook_namespace().get(task.home)
        if module is not None and hasattr(module, task.name):
            return getattr(module, task.name)
    return _notebook_namespace().get(task.name)


def _valid_ids(lab_cases) -> str:
    return ", ".join(lab_cases.tasks)


def _resolve_task(task_id: Any, obj: Any, lab_cases) -> tuple[Task | None, Any]:
    if not isinstance(task_id, str):
        if isinstance(obj, str) and lab_cases.find(obj.strip().upper()) is not None:
            task_id, obj = obj, task_id                       # check(squared_distance, "C1")
            say(f"Tip: the task id goes first: check(\"{task_id}\", {getattr(obj, '__name__', '...')}).")
            return lab_cases.find(task_id.strip().upper()), obj
        name = getattr(task_id, "__name__", None)
        task = lab_cases.by_name(name) if name else None
        if task is not None and obj is None:
            say(f"Tip: put the task id first: check(\"{task.id}\", {name}). Checking {task.id} now.")
            return task, task_id
        say(f"check() needs a task id first, e.g. check(\"C1\", squared_distance). "
            f"Tasks in Lab {lab_cases.lab}: {_valid_ids(lab_cases)}.")
        return None, obj
    key = task_id.strip().upper().replace(" ", "")
    task = lab_cases.find(key)
    if task is None:
        say(f"I don't know a task called \"{task_id}\" in Lab {lab_cases.lab}. "
            f"Valid ids: {_valid_ids(lab_cases)}.")
    return task, obj


def _object_problem(task: Task, obj: Any, lab_cases) -> str | None:
    """A friendly message if `obj` can't be the thing this task checks; None if it can."""
    if task.kind == "file" or task.kind == "custom":
        return None
    example = f"check(\"{task.id}\", {task.name})"
    if isinstance(obj, str):
        return f"Pass the {'class' if task.kind == 'class' else 'function'} itself, without quotes: {example}."
    if task.kind == "class":
        if isinstance(obj, type):
            return None
        if hasattr(type(obj), "__mro__") and type(obj).__name__ == task.name:
            return f"Pass the class itself, not an object made from it: {example} (no brackets after the name)."
        what = f"the function {obj.__name__}" if callable(obj) and hasattr(obj, "__name__") else type_name(obj)
        return f"{example} needs the class {task.name}, not {what}."
    if not callable(obj):
        return (f"{example} needs the function itself, not its result ({short(obj, 40)}). "
                "Write the name without brackets after it.")
    other = lab_cases.by_name(getattr(obj, "__name__", ""))
    if other is not None and other.id != task.id and getattr(obj, "__name__", "") != task.name:
        return (f"You passed {obj.__name__}, which is task {other.id}. "
                f"Did you mean check(\"{other.id}\", {obj.__name__})?")
    return None


def _dependency(task: Task, obj: Any, lab_cases) -> tuple[str, str] | None:
    """(cause, next step) if `obj` calls an earlier task's function that isn't passing."""
    if not task.depends_on or obj is None:
        return None
    used = names_used(obj)
    namespace = _globals_of(obj)
    for dep_id in task.depends_on:
        dep = lab_cases.find(dep_id)
        if dep is None or dep.name not in used:
            continue
        dep_obj = namespace.get(dep.name)
        dep_name = qualified(dep)
        rescue_text = f"`{dep_name} = rescue(\"{dep.id}\")`"
        if dep_obj is None:
            where = "Run the cell that defines it" if dep.home == "__main__" else f"Write it in {dep.home}.py and save"
            return (f"{task.name} uses {dep.name}, which isn't defined yet ({dep.id}). {where}, "
                    f"or run {rescue_text}.", f"run check(\"{dep.id}\", {dep_name}) when it's defined.")
        try:
            status = run_task(dep, dep_obj, lab_cases)[0].status
        except Exception:
            status = NOT_YET
        if status != PASS:
            return (f"{task.name} uses {dep.name}, which isn't passing yet ({dep.id}). "
                    f"Fix {dep.id} first or run {rescue_text}.",
                    f"run check(\"{dep.id}\", {dep_name}) to see what {dep.id} needs.")
    return None


# ============================================================================= public API
def check(task_id: str, obj: object = None, *, verbose: bool = False, style: bool = False) -> bool:
    """Check your work on one task, e.g. check("C1", squared_distance). Returns True on PASS.

    verbose=True also prints the full traceback of an error and anything your function printed.
    style=True adds optional style notes (they never change the result).
    """
    try:
        lab_cases = _lab_cases()
        if lab_cases is None:
            return False
        task, obj = _resolve_task(task_id, obj, lab_cases)
        if task is None:
            return False
        if obj is None and task.kind in ("function", "class", "test-function"):
            obj = find_object(task)
            if obj is None:
                say(_not_started_line(task, f"I can't find {task.name} yet: run the cell that defines it, "
                                            "then this check." if task.home == "__main__" else
                                            f"{task.home}.py has no {task.name} yet: write it, save the file, "
                                            "then run this check."))
                return False
        problem = _object_problem(task, obj, lab_cases)
        if problem:
            say(f"{task.id} {task.name}: {problem}")
            return False
        report, ctx = run_task(task, obj, lab_cases, verbose=verbose)
        dependency = _dependency(task, obj, lab_cases) if report.status == NOT_YET else None
        say(*format_report(report, ctx, lab=lab_cases.lab, verbose=verbose, style=style, dependency=dependency))
        return report.status == PASS
    except _AuthorProblem as problem:
        say(f"{task_id}: this check can't run: {problem}. Re-extract the kit; if it keeps happening, tell Javier.")
        return False
    except KeyboardInterrupt:
        say(f"{task_id}: check stopped (you pressed Interrupt).")
        return False
    except Exception as bug:                   # a bug in labkit must never look like the student's fault
        say(f"{task_id}: check() hit a problem inside labkit ({type(bug).__name__}: {bug}). "
            "This is not your fault: tell Javier, and carry on with the next task.")
        if verbose:
            say(hide_user_folder("".join(traceback.format_exception(type(bug), bug, bug.__traceback__))))
        return False


def quiet_status(task: Task, lab_cases=None) -> str:
    """PASS / NOT YET / NOT STARTED for the object the notebook currently defines (prints nothing)."""
    obj = find_object(task) if task.kind in ("function", "class", "test-function") else None
    if obj is None and task.kind in ("function", "class", "test-function"):
        return NOT_STARTED
    if _object_problem(task, obj, lab_cases or lab_registry.load(current_lab())):
        return NOT_YET
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return run_task(task, obj, lab_cases)[0].status
    except KeyboardInterrupt:
        raise
    except Exception:
        return NOT_YET


def is_rescued(obj: Any) -> bool:
    """True if obj is a reference borrowed with rescue() (it comes from labkit/reference/)."""
    module = getattr(obj, "__module__", None) or ""
    return module == "labkit.reference" or module.startswith("labkit.reference.")


def check_all(*, verbose: bool = False) -> dict[str, str]:
    """Check every CORE task you have defined in this notebook and print a status table, then the
    OPTIONAL tasks (extra practice) as a separate list.

    A task whose object is a reference borrowed with rescue() shows as RESCUED and is not counted as passing.
    Returns {task id: "PASS" / "NOT YET" / "NOT STARTED"} for both (a rescued task is "PASS": it works).
    verbose=True also runs the full check() for every CORE task that isn't passing.
    """
    lab_cases = _lab_cases()
    if lab_cases is None:
        return {}
    core = lab_cases.core()
    optional = [task for task in lab_cases.optional_part() if task not in core]
    statuses: dict[str, str] = {}
    try:
        for task in core + optional:
            statuses[task.id] = quiet_status(task, lab_cases)
    except KeyboardInterrupt:
        say("check_all stopped (you pressed Interrupt).")
        return statuses
    shown = core + optional
    id_width = max((len(t.id) for t in shown), default=2)
    name_width = max((len(t.name) for t in shown), default=10) + 3
    rescued = {task.id for task in shown if statuses[task.id] == PASS and is_rescued(find_object(task))}

    def row(task: Task) -> str:
        status = statuses[task.id]
        text = f"  {task.id.ljust(id_width)} {task.name.ljust(name_width)}"
        if task.id in rescued:
            where = "run its cell" if task.home == "__main__" else f"save {task.home}.py"
            text += f"RESCUED (reference)  → come back to it: write yours, {where}, then check_all()"
        elif status == PASS:
            text += PASS
        elif status == NOT_YET:
            text += NOT_YET.ljust(13) + f"→ check(\"{task.id}\", {qualified(task)}) shows why"
        else:
            text += NOT_STARTED
        if status != PASS and task.not_needed_later and not task.in_optional_part:
            text += "  (not needed later)"
        return text.rstrip()

    lines = [f"Lab {lab_cases.lab} · CORE status"] + [row(task) for task in core]
    passing = sum(1 for task in core if statuses[task.id] == PASS and task.id not in rescued)
    n_rescued = sum(1 for task in core if task.id in rescued)
    path = " → ".join(t.id for t in lab_cases.minimum_path())
    footer = f"CORE: {passing} of {len(core)} passing" + (f", {n_rescued} rescued." if n_rescued else ".")
    if path:
        goal = lab_cases.finale
        footer += f" Minimum path to {goal}: {path}." if goal else f" Minimum path: {path}."
    lines.append(footer)
    if optional:
        done = sum(1 for task in optional if statuses[task.id] == PASS and task.id not in rescued)
        lines.append(f"OPTIONAL (extra practice, any order): {done} of {len(optional)} passing.")
        lines += [row(task) for task in optional]
    say(*lines)
    if verbose:
        for task in core:
            if statuses[task.id] != PASS:
                say("")
                check(task.id, find_object(task), verbose=True)
    return statuses
