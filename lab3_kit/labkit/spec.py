"""The building blocks of a task specification: Task, Case, Detector, Speed, and comparisons.

A lab's checks are DATA, written in labkit/cases/labN.py with these classes (see AUTHORING.md).
The check engine (check.py) reads them; nothing here runs student code.

    Task("C1", "squared_distance",
         practise="the running-total loop (Theory S2)",
         cases=[Case(([0, 0], [3, 4]), 25), Case(lambda: (TEST.images[0], TRAIN.images[0]))],
         detectors=[Detector(lambda t: t.got == 5.0, "you returned the square root ...")],
         returns=int)

Expected values that come from the reference are not typed in by hand: leave `expected`
out (it defaults to REFERENCE) and generators/gen_cases.py stores them in labN_data.json.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .data import cached_digits

KINDS = ("function", "class", "test-function", "file", "custom")


class _Reference:
    """Marker: "the expected value is computed from the reference at build time"."""

    def __repr__(self) -> str:
        return "REFERENCE"


REFERENCE = _Reference()


@dataclass(frozen=True)
class Fact:
    """A build-time fact for detectors: gen_cases.py calls reference.<function>(*case args).

    Example: Case(args, facts={"index": Fact("nearest_index")}) lets a detector test
    `t.got == t.facts["index"]` ("you returned the index, not the label").
    """
    function: str


@dataclass(frozen=True)
class Raises:
    """Expected outcome "the call raises this error", optionally with an exact message.

    Raises(ValueError, "images have different lengths: 2 vs 1") or Raises(ValueError, contains="lengths").
    """
    error: type
    message: str | None = None
    contains: str | None = None

    def describe(self) -> str:
        if self.message is not None:
            return f"raises {self.error.__name__}({self.message!r})"
        if self.contains is not None:
            return f"raises {self.error.__name__} with {self.contains!r} in its message"
        return f"raises {self.error.__name__}"

    def matches(self, error: BaseException | None) -> bool:
        if error is None or not isinstance(error, self.error):
            return False
        text = str(error)
        if self.message is not None and text != self.message:
            return False
        return self.contains is None or self.contains in text


@dataclass(frozen=True)
class Described:
    """An expectation in words, printed as it is (no quotes): ctx.expect(attempt, Described("two files in results/")).

    For a custom check whose call raised before anything could be compared: the TEST block then reads
    `Expected:      two files in results/`, not `Expected:      'two files in results/'` (which looks like a
    string the function had to return).
    """
    text: str

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


# ----------------------------------------------------------------------------- comparisons
@dataclass(frozen=True)
class Compare:
    """How a result is compared with the expected value. Build one with exact(), approx() or custom()."""
    same: Callable[[Any, Any], bool]
    note: str = ""              # shown after the expected value, e.g. "(within ±0.001)"


def _equal(got: Any, expected: Any) -> bool:
    try:
        return bool(got == expected)
    except Exception:           # e.g. comparing numpy arrays, or a broken __eq__
        return False


def exact() -> Compare:
    """got == expected (the default)."""
    return Compare(_equal)


def _close(got: Any, expected: Any, tol: float) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return got is expected
    if isinstance(expected, (int, float)):
        if not isinstance(got, (int, float)):
            return False
        if math.isinf(expected) or math.isinf(got):
            return got == expected
        return abs(got - expected) <= tol
    if isinstance(expected, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(expected)
                and all(_close(g, e, tol) for g, e in zip(got, expected)))
    if isinstance(expected, dict):
        return (isinstance(got, dict) and set(got) == set(expected)
                and all(_close(got[k], expected[k], tol) for k in expected))
    return _equal(got, expected)


def approx(tol: float = 1e-6) -> Compare:
    """Numbers (also inside lists and dicts) may differ by at most `tol`.

    The report says "(within ±0.001)" next to the expected value; a tolerance below 1e-6 is only there
    for floating-point rounding, so it is not mentioned.
    """
    return Compare(lambda got, expected: _close(got, expected, tol), f"(within ±{tol:g})" if tol >= 1e-6 else "")


def custom(same: Callable[[Any, Any], bool], note: str = "") -> Compare:
    """Your own rule: same(got, expected) -> bool."""
    return Compare(same, note)


# ----------------------------------------------------------------------------- cases and rules
@dataclass
class Case:
    """One call of the student's object and the answer it should give.

    args      a tuple of arguments, or a function returning one (for real data, loaded lazily)
    kwargs    keyword arguments (a dict)
    expected  the answer; REFERENCE (default) = stored at build time by gen_cases.py;
              Raises(...) = the call must raise that error
    compare   exact() (default), approx(tol) or custom(fn)
    show      how the call is printed under "Failing case:", e.g. "accuracy([1, 2, 3], [1, 2, 4])";
              default: built from the arguments. Write it for real data so students can re-run it.
    call      instead of obj(*args): a function call(obj) -> result (for classes, symmetry, ...)
    key       a stable name for this case in labN_data.json (default: its position, "0", "1", ...)
    facts     {name: value or Fact("reference_function")} for detectors

    How the case is printed (all optional; the defaults work for every lab):
    title          a short human description, printed as "TEST 3: <title>" (default: the call itself)
    show_inputs    {label: text} lines printed under the title, e.g. {"Predictions": "[1, 2, 3, 4]"}
                   (default: the arguments labelled with Task.input_labels, or "Call: <show>")
    expected_label replaces the task's expected_label for this case, e.g. "Expected"
    highlight      True = one of the (at most Task.show_on_pass) tests printed on PASS; if no case of a
                   task is highlighted, the first titled cases are printed
    preview        how many characters of Expected / Your result to show (default 70); e.g. 160 for a result
                   that is one readable line of 80-120 characters (a long list is still shortened)
    """
    args: Any = ()
    expected: Any = REFERENCE
    kwargs: dict = field(default_factory=dict)
    compare: Compare | None = None
    show: str | None = None
    call: Callable[[Any], Any] | None = None
    key: str | None = None
    facts: dict = field(default_factory=dict)
    title: str | None = None
    show_inputs: dict | None = None
    expected_label: str | None = None
    highlight: bool = False
    preview: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.args, list):
            raise TypeError("Case args must be a tuple of arguments, e.g. Case(([3, 1, 3],), ...) for one "
                            "list argument (note the comma), not Case([3, 1, 3], ...)")

    def resolve_args(self) -> tuple:
        args = self.args() if callable(self.args) else self.args
        return tuple(args)


@dataclass
class Detector:
    """A misconception rule: if test(attempt) is True on the failing case, print `message`.

    `test` receives an Attempt: t.args, t.kwargs, t.got, t.expected, t.error, t.facts, t.raised(KeyError).
    `next_step` replaces the default "Next step" line (for example to point at a hint).
    """
    test: Callable[[Any], bool]
    message: str
    next_step: str | None = None


@dataclass
class Speed:
    """A speed note that WARNS but never fails.

    args       the arguments to time (a tuple, or a function returning one)
    budget_s   seconds per call on the build laptop; scaled by machine_factor() on other laptops
    note       what to say when the call is slower, e.g. "C2 calls this 100,000 times. ..."
    """
    args: Any
    budget_s: float
    note: str = ""

    def resolve_args(self) -> tuple:
        args = self.args() if callable(self.args) else self.args
        return tuple(args)


@dataclass
class Task:
    """Everything the check engine needs to know about one task.

    id            "W", "C1" ... "C6", "ENG1", "SCI2", "BLD1", "CH", "AI" ...
    name          the student's function or class name, e.g. "squared_distance"
    kind          "function" | "class" | "test-function" | "file" | "custom"
    practise      "the running-total loop (Theory S2)", printed as "You practised: ..."
    cases         toy cases first, then real ones
    detectors     2-4 misconception rules, tried in order on the failing case
    returns       a type (or tuple of types) every result must have; None = no type check
    inputs_unchanged  True = changing an argument in place is a failure (a property check)
    speed         a Speed note, or None
    depends_on    ids of earlier tasks whose functions this one may call (dependency hints)
    minimum_path  on the minimum path (the Core tasks the lab's final run needs)
    not_needed_later  a skippable Core task
    optional      a C-numbered task that lives in the OPTIONAL part (check_all lists it there, not in CORE)
    custom        an extra hook custom(obj, ctx) for anything cases cannot express
    reference     name in reference/labN.py (default: name); rescue=False = no rescue for this task
    home          module where check_all() finds the object (default "__main__", i.e. the notebook)
    quality       print the "Quality:" line on PASS
    input_labels  labels for the positional arguments in the TEST lines, e.g. ("Predictions", "Ground truth")
    expected_label  the label of the expected value, e.g. "Expected accuracy" (default "Expected")
    show_on_pass  how many tests a PASS prints in full (the rest are counted: "... and 3 more tests passed")
    custom_tests  how many tests the custom hook records when it runs to the end; a NOT YET then reports the
                  same total every time ("2 of 7 tests passed, 3 not run"), even when the hook stops early
    stub_fingerprint  for a stub that ships with code already in it (so it is not pass-only): the
                  check.fingerprint() of the stub as shipped; while the student's code still has it, the task
                  is NOT STARTED, not NOT YET (write it from the student version at build time)
    """
    id: str
    name: str
    kind: str = "function"
    title: str = ""
    practise: str = ""
    cases: list = field(default_factory=list)
    detectors: list = field(default_factory=list)
    returns: Any = None
    inputs_unchanged: bool = True
    speed: Speed | None = None
    depends_on: tuple = ()
    minimum_path: bool = False
    not_needed_later: bool = False
    custom: Callable[[Any, Any], None] | None = None
    reference: str | None = None
    rescue: bool = True
    home: str = "__main__"
    quality: bool = True
    optional: bool = False
    input_labels: tuple = ()
    expected_label: str = "Expected"
    show_on_pass: int = 3
    custom_tests: int = 0
    stub_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"task {self.id}: kind must be one of {KINDS}, not {self.kind!r}")
        self.depends_on = tuple(self.depends_on)
        # detectors may also be written as plain (predicate, message) pairs
        self.detectors = [d if isinstance(d, Detector) else Detector(*d) for d in self.detectors]

    @property
    def core(self) -> bool:
        """CORE tasks are C1, C2, ... (task ids from the blueprint), unless marked optional."""
        return self.id[:1] == "C" and self.id[1:].isdigit() and not self.optional

    @property
    def in_optional_part(self) -> bool:
        """Tasks of the OPTIONAL part: an optional C task, or ENG1, SCI1, BLD1."""
        if self.optional:
            return True
        return self.id[:3] in ("ENG", "SCI", "BLD") and self.id[3:] == "1"

    @property
    def reference_name(self) -> str:
        return self.reference or self.name


@dataclass
class Attempt:
    """What happened when the engine called the student's object once (given to detectors)."""
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    expected: Any = None
    got: Any = None
    error: BaseException | None = None
    printed: str = ""                                # what the call printed (hidden from the report)
    stderr: str = ""                                 # what it wrote to sys.stderr
    seconds: float = 0.0
    changed: list = field(default_factory=list)     # arguments the call changed in place
    change_note: str = ""                            # e.g. "truth was [1, 2], after the call it is [2, 1]"
    facts: dict = field(default_factory=dict)
    show: str = ""
    interrupted: bool = False
    title: str = ""                                  # "TEST n: <title>" (default: show)
    inputs: list = field(default_factory=list)      # (label, text) lines printed under the title
    expected_label: str = "Expected"
    headline: bool = False                           # printed in full on PASS
    preview: int | None = None                       # Case.preview: characters shown for Expected / Your result

    def raised(self, *errors: type) -> bool:
        """t.raised(KeyError) -> True if the call raised a KeyError (or a subclass)."""
        return self.error is not None and isinstance(self.error, errors or (BaseException,))

    @property
    def error_text(self) -> str:
        """'TypeError: ...' (empty if no error)."""
        if self.error is None:
            return ""
        message = str(self.error)
        return f"{type(self.error).__name__}: {message}" if message else type(self.error).__name__


# ----------------------------------------------------------------------------- data for authors
class Digits:
    """Lazy access to a kit digit file for real cases: loaded the first time it is used.

        TRAIN = Digits("mnist_train_10k.csv.gz", limit=1000)
        Case(lambda: (TEST.images[3], TRAIN.images[:200], TRAIN.labels[:200]), show="...")
    """

    def __init__(self, name: str, limit: int | None = None) -> None:
        self.name = name
        self.limit = limit

    @property
    def images(self) -> list[list[int]]:
        return cached_digits(self.name, self.limit)[0]

    @property
    def labels(self) -> list[int]:
        return cached_digits(self.name, self.limit)[1]


# ----------------------------------------------------------------------------- JSON with types
def encode(value: Any) -> Any:
    """Turn a Python value into JSON-safe data that keeps int dict keys, tuples and sets.

    decode(encode(x)) == x for None, bool, int, float (also inf), str, list, tuple, set, dict.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": [encode(v) for v in sorted(value, key=repr)]}
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith("__") for k in value):
            return {k: encode(v) for k, v in value.items()}
        return {"__dict__": [[encode(k), encode(v)] for k, v in value.items()]}
    raise TypeError(f"cannot store a {type(value).__name__} as case data")


def decode(data: Any) -> Any:
    """The inverse of encode()."""
    if isinstance(data, list):
        return [decode(v) for v in data]
    if isinstance(data, dict):
        if set(data) == {"__float__"}:
            return float(data["__float__"])
        if set(data) == {"__tuple__"}:
            return tuple(decode(v) for v in data["__tuple__"])
        if set(data) == {"__set__"}:
            return {decode(v) for v in data["__set__"]}
        if set(data) == {"__dict__"}:
            return {_hashable(decode(k)): decode(v) for k, v in data["__dict__"]}
        return {k: decode(v) for k, v in data.items()}
    return data


def _hashable(key: Any) -> Any:
    return tuple(key) if isinstance(key, list) else key
