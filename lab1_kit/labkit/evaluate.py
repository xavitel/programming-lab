"""Measuring a model: evaluate() (the final run of a lab), compare(), stopwatch and count_code_lines().

    evaluate(predict_nearest, n_train=1000, n_test=100, accuracy_fn=accuracy)
    Reading 100 test images with 1,000 training images: about 6 s on this laptop.
    ..........
    84 of 100 correct (84.0%) · 5.7 s · 57 ms per digit

try_predict(predict_nearest, image, train_images, train_labels) reads ONE image for a provided cell: it
returns the label, or None after printing what to do next (never a traceback).

evaluate() first checks that your function is ready (one timed call), prints an estimate, refuses
runs that would take more than 30 s (unless allow_slow=True), shows one dot per 10 images, and
returns a dict with the numbers. Interrupt (the square button) stops a run cleanly.
The 30 s limit is for the build laptop; slower laptops get a proportionally higher one
(see machine_factor()), so correct code is not refused just for running on an older machine.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import sys
import textwrap
import time
import tokenize
from pathlib import Path

from . import cases
from .check import _dependency, is_stub, run_call, your_line
from .data import cached_digits, digit_files
from .env import current_lab
from .text import is_number, say, short, thousands, type_name
from .timing import format_duration, format_eta, machine_factor

MAX_ETA_S = 30.0
_last: dict = {}


# ----------------------------------------------------------------------------- helpers
def _task_list(ids: list[str]) -> str:
    """['C1', 'C2', 'C3'] -> 'C1–C3'; ['C1', 'C2'] -> 'C1 and C2'; ['C1', 'C3', 'C4'] -> 'C1, C3 and C4'."""
    if len(ids) == 2:
        return f"{ids[0]} and {ids[1]}"
    numbers = [int(i[1:]) for i in ids if i[:1] == "C" and i[1:].isdigit()]
    if len(ids) > 1 and len(numbers) == len(ids) and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"{ids[0]}–{ids[-1]}"
    return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " and " + ids[-1]


def moment_message(predict_fn: object = None) -> str:
    """'Putting It All Together needs C1–C3. Not there yet? Run predict_nearest = rescue("C2")'

    The name of the run comes from the lab's cases file (FINALE); without one it says 'This run'.
    """
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    path = lab_cases.minimum_path() if lab_cases else []
    name = getattr(predict_fn, "__name__", None)
    task = lab_cases.by_name(name) if lab_cases and name else None
    if task is None and path:
        task = next((t for t in reversed(path) if "predict" in t.name), path[-1])
    needs = _task_list([t.id for t in path]) if path else "your predict function"
    run = (lab_cases.finale if lab_cases else None) or "This run"
    if task is None:
        return f"{run} needs {needs}. Not there yet? Use rescue() for the task you're stuck on."
    return f"{run} needs {needs}. Not there yet? Run {task.name} = rescue(\"{task.id}\")"


def _task_id_for(fn: object, fallback: str) -> str:
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    task = lab_cases.by_name(getattr(fn, "__name__", "")) if lab_cases else None
    return task.id if task else fallback


def _pixel_problem(images: list) -> str | None:
    """A friendly message if the images aren't lists of int pixels (checked on a few images)."""
    for image in images[:3]:
        if not isinstance(image, list) or not image:
            return f"each image should be a list of numbers, but one is {type_name(image)}."
        for pixel in image[:50]:
            if isinstance(pixel, str):
                return (f"the pixels are text such as {pixel!r}, not numbers. Convert each one with int() "
                        "when you read the file.")
            if not is_number(pixel):
                return f"a pixel is {type_name(pixel)} ({short(pixel, 20)}), not a number."
    return None


def _count_problem(n_train: object, n_test: object, size: object) -> str | None:
    if size not in (28, 14):
        return f"size must be 28 or 14 (got {size!r})."
    for name, value, top in (("n_train", n_train, 10_000), ("n_test", n_test, 2_000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= top:
            return f"{name} must be a whole number from 1 to {top:,} (got {value!r})."
    return None


def _empty_result(n_train: object, n_test: object, size: object) -> dict:
    return {"status": "not run", "correct": None, "total": n_test, "accuracy": None, "seconds": None,
            "ms_per_digit": None, "n_train": n_train, "n_test": n_test, "size": size,
            "predictions": [], "labels": [], "mistakes": []}


def last_run(predict_fn: object) -> dict | None:
    """The last finished evaluate() run of this exact function, or None."""
    run = _last.get("run")
    if run and run["fn"] is predict_fn and run["status"] == "done":
        return run
    return None


def _error_lines(fn_name: str, error: BaseException, where: str, fn: object = None) -> list[str]:
    text = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
    lines = [f"{fn_name} raised {text} on {where}."]
    place = your_line(error, fn, always_name=True)       # printed far from the def cell: name the function
    if place:
        lines.append(f"{place[0]} {place[1]}")
    return lines


# ----------------------------------------------------------------------------- one image, safely
def _is_label(value: object) -> bool:
    """An int label (also a numpy integer), not a bool, a float or text."""
    return not isinstance(value, (bool, str, bytes, float)) and hasattr(value, "__index__")


def try_predict(predict_fn, image: list[int], train_images: list[list[int]], train_labels: list[int]):
    """Return predict_fn(image, train_images, train_labels), or None after printing what to do next.

    Provided cells (the C2 view, Draw your own digit) use it, so that an unfinished task shows the next step
    instead of a traceback, even when the problem is in an earlier task (e.g. C1 still returns None):

        guess = try_predict(predict_nearest, test_images[0], train_images, train_labels)
        if guess is not None:
            print("Your machine says:", guess)
    """
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    name = getattr(predict_fn, "__name__", "your function")
    task = lab_cases.by_name(name) if lab_cases is not None else None
    if task is None and lab_cases is not None:
        task = next((t for t in reversed(lab_cases.minimum_path()) if "predict" in t.name), None)
    rescue_text = (f"run {task.name} = rescue(\"{task.id}\") in a new cell, then run this cell again"
                   if task is not None and task.rescue else "finish it, then run this cell again")
    if not callable(predict_fn) or is_stub(predict_fn):
        needs = _task_list(list(task.depends_on) + [task.id]) if task is not None else name
        say(f"This cell needs {needs}. Not there yet? {rescue_text[0].upper()}{rescue_text[1:]}.")
        return None
    train_copy = [list(train) for train in train_images]        # your function gets its own copy
    attempt = run_call(predict_fn, (list(image), train_copy, list(train_labels)), track=False)
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return None
    if attempt.error is None and _is_label(attempt.got):
        return attempt.got
    if attempt.error is not None:
        lines = _error_lines(name, attempt.error, "this image", predict_fn)
    elif attempt.got is None:
        lines = [f"{name} returned None instead of a label."]
    else:
        lines = [f"{name} returned {short(attempt.got, 40)} ({type_name(attempt.got)}); it should return a "
                 "label, an int such as 7."]
    dependency = _dependency(task, predict_fn, lab_cases) if task is not None else None
    if dependency is not None:
        lines += [f"Likely cause: {dependency[0]}", f"Next step: {dependency[1]}"]
    elif task is not None:
        lines.append(f"Next step: run check(\"{task.id}\", {name}) to see why, or {rescue_text}.")
    say(*lines)
    return None


# ----------------------------------------------------------------------------- evaluate
def evaluate(predict_fn, *, n_train: int = 1000, n_test: int = 100, size: int = 28,
             accuracy_fn=None, allow_slow: bool = False) -> dict:
    """Run predict_fn(image, train_images, train_labels) on the first n_test test images.

    Returns {"correct", "total", "accuracy", "seconds", "ms_per_digit", "predictions", "labels",
    "mistakes", "status", ...}; "status" is "done", "stopped", "refused" or "not ready".
    """
    result = _empty_result(n_train, n_test, size)
    problem = _count_problem(n_train, n_test, size)
    if problem:
        say(f"evaluate: {problem}")
        result["status"] = "refused"
        return result
    name = getattr(predict_fn, "__name__", "your function")
    if not callable(predict_fn) or is_stub(predict_fn):
        say(moment_message(predict_fn))
        result["status"] = "not ready"
        return result

    train_file, test_file = digit_files(size)
    train_images, train_labels = cached_digits(train_file, n_train)
    test_images, test_labels = cached_digits(test_file, n_test)
    problem = _pixel_problem(train_images) or _pixel_problem(test_images)
    if problem:
        say(f"evaluate: {problem}")
        result["status"] = "refused"
        return result
    train_copy = [image[:] for image in train_images]        # your function gets its own copy
    labels_copy = train_labels[:]

    # 1. one timed call: is the function ready, and how long will the run take?
    first = run_call(predict_fn, (test_images[0][:], train_copy, labels_copy), track=False)
    if first.interrupted:
        say("Stopped (you pressed Interrupt).")
        result["status"] = "stopped"
        return result
    if first.error is not None:
        say(*_error_lines(name, first.error, "the first test image", predict_fn), moment_message(predict_fn))
        result["status"] = "not ready"
        return result
    if first.got is None:
        say(moment_message(predict_fn))
        result["status"] = "not ready"
        return result
    if isinstance(first.got, (str, bool)) or not (is_number(first.got) or hasattr(first.got, "__index__")):
        say(f"{name} returned {short(first.got, 40)} ({type_name(first.got)}); it should return a label, "
            f"an int such as 7. check(\"{_task_id_for(predict_fn, 'C2')}\", {name}) shows why.")
        result["status"] = "not ready"
        return result
    per_image = first.seconds
    eta = per_image * n_test
    if eta > MAX_ETA_S and not allow_slow:
        # Slow laptops (and Python 3.10) get a proportionally higher limit: 30 s on the build laptop.
        limit = MAX_ETA_S * max(1.0, machine_factor())
        if eta > limit:                           # time two more calls before refusing
            per_image = min(per_image, min(run_call(predict_fn, (test_images[0][:], train_copy, labels_copy),
                                                    track=False).seconds for _ in range(2)))
            eta = per_image * n_test
        if eta > limit:
            hint = (f" Do you need all {n_train:,} training images? 1,000 is enough here." if n_train > 1000
                    else " Is something slow inside the loop?")
            say(f"That would take {format_eta(eta)} on this laptop ({n_test:,} images × "
                f"{format_duration(per_image)} each), more than the {limit:.0f} s limit.{hint} "
                "To run it anyway: evaluate(..., allow_slow=True). The Interrupt button stops a long run.")
            result["status"] = "refused"
            return result
    say(f"Reading {thousands(n_test)} test images with {thousands(n_train)} training images: "
        f"{format_eta(eta)} on this laptop.")

    # 2. the run, with a dot every 10 images
    screen = sys.stdout
    predictions = [first.got]
    hidden = io.StringIO()
    start = time.perf_counter()
    status, crash = "done", None
    try:
        with contextlib.redirect_stdout(hidden):
            for index in range(1, n_test):
                predictions.append(predict_fn(test_images[index], train_copy, labels_copy))
                if (index + 1) % 10 == 0:
                    screen.write(".")
                    screen.flush()
    except KeyboardInterrupt:
        status = "stopped"
    except Exception as error:   # the function crashed on a later image
        status, crash = "stopped", error
    seconds = first.seconds + (time.perf_counter() - start)
    screen.write("\n")
    screen.flush()
    done = len(predictions)
    truth = test_labels[:done]
    correct = sum(1 for p, t in zip(predictions, truth) if p == t)
    result.update(predictions=predictions, labels=truth, correct=correct, total=done,
                  mistakes=[i for i, (p, t) in enumerate(zip(predictions, truth)) if p != t],
                  seconds=round(seconds, 3), ms_per_digit=round(1000 * seconds / done, 2),
                  accuracy=correct / done)
    if status == "stopped":
        result["status"] = "stopped"
        if crash is not None:
            say(*_error_lines(name, crash, f"test image #{done}", predict_fn))
        say(f"Stopped after {done} of {n_test} images: {correct} correct so far.")
        return result

    # 3. accuracy: the harness counts, and cross-checks your accuracy() if you pass one
    shown = correct / n_test
    if accuracy_fn is not None:
        accuracy_name = getattr(accuracy_fn, "__name__", "accuracy")
        accuracy_id = _task_id_for(accuracy_fn, "C3")
        if not callable(accuracy_fn) or is_stub(accuracy_fn):
            say(f"({accuracy_name} isn't ready yet ({accuracy_id}), so the harness counted for you.)")
        else:
            attempt = run_call(accuracy_fn, (predictions, truth))
            value = attempt.got
            if attempt.error is not None:
                say(f"(Your {accuracy_name} raised {attempt.error_text}, so the harness counted for you. "
                    f"check(\"{accuracy_id}\", {accuracy_name}) shows why.)")
            elif is_number(value) and abs(value - shown) < 1e-9:
                shown = value
            elif is_number(value) and abs(value - 100 * shown) < 1e-6:
                say(f"(Your {accuracy_name} returned {value!r}, a percentage. The spec asks for a fraction "
                    f"between 0 and 1 ({shown!r}). check(\"{accuracy_id}\", {accuracy_name}) shows why.)")
            else:
                say(f"(Your {accuracy_name} returned {short(value, 30)}, but the harness counted {shown!r}. "
                    f"check(\"{accuracy_id}\", {accuracy_name}) shows why.)")
    if train_copy != train_images or labels_copy != train_labels:
        say(f"Note: {name} changed the training lists during the run. Build new lists instead.")
    printed = hidden.getvalue().count("\n")
    if printed:
        say(f"({name} printed {printed} lines during the run; they were hidden.)")
    result["status"] = "done"
    say(f"{correct} of {n_test} correct ({100 * shown:.1f}%) · {format_duration(seconds)} · "
        f"{format_duration(seconds / n_test)} per digit")
    _last["run"] = dict(result, fn=predict_fn)
    return result


# ----------------------------------------------------------------------------- compare (Lab 2)
def _required_positional(fn: object) -> int:
    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    if code is None:
        try:
            return sum(1 for p in inspect.signature(fn).parameters.values()
                       if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        except (TypeError, ValueError, NameError):
            return 3
    defaults = getattr(fn, "__defaults__", None) or ()
    count = code.co_argcount - len(defaults)
    return count - 1 if inspect.ismethod(fn) else count


def compare(old_fn, new_fn, *, n: int = 20, n_train: int = 1000, size: int = 28,
            labels: tuple[str, str] = ("legacy", "yours")) -> dict:
    """Golden check: do old_fn and new_fn give identical answers on the first n test images?

    Works for f(image), f(image_a, image_b) and f(image, train_images, train_labels).
    Prints e.g. "20/20 identical · legacy 137 ms/digit · yours 66 ms/digit".
    """
    result = {"identical": 0, "n": n, "old_ms": None, "new_ms": None, "differences": [], "status": "not run"}
    problem = _count_problem(n_train, n, size)
    if problem:
        say(f"compare: {problem}")
        return result
    for fn in (old_fn, new_fn):
        if not callable(fn):
            say(f"compare: pass two functions, without brackets after their names (got {type_name(fn)}).")
            return result
    if is_stub(new_fn):
        name = getattr(new_fn, "__name__", "your function")
        say(f"compare: {name} is still a stub (only `pass`). Write it first, then compare.")
        result["status"] = "not ready"
        return result
    arity = _required_positional(old_fn)
    train_file, test_file = digit_files(size)
    train_images, train_labels = cached_digits(train_file, max(n_train, n))
    test_images, _ = cached_digits(test_file, n)
    train_images, train_labels = train_images[:n_train], train_labels[:n_train]

    def arguments(i: int) -> tuple:
        if arity <= 1:
            return (test_images[i][:],)
        if arity == 2:
            return (test_images[i][:], train_images[i][:])
        return (test_images[i][:], [im[:] for im in train_images], train_labels[:])

    answers, times = {}, {}
    for label, fn in zip(labels, (old_fn, new_fn)):
        out, total = [], 0.0
        for i in range(n):
            attempt = run_call(fn, arguments(i), track=False)
            if attempt.interrupted:
                say("compare stopped (you pressed Interrupt).")
                return result
            if attempt.error is not None:
                fn_name = getattr(fn, "__name__", label)
                say(*_error_lines(fn_name, attempt.error, f"test image #{i}"))
                result["status"] = "not ready"
                return result
            out.append(attempt.got)
            total += attempt.seconds
        answers[label], times[label] = out, total / n
    old_out, new_out = answers[labels[0]], answers[labels[1]]
    same = [_equal(a, b) for a, b in zip(old_out, new_out)]
    differences = [{"image": i, labels[0]: a, labels[1]: b}
                   for i, (a, b, ok) in enumerate(zip(old_out, new_out, same)) if not ok]
    unit = "/digit" if arity >= 3 else " per call"
    result.update(identical=sum(same), old_ms=round(1000 * times[labels[0]], 3),
                  new_ms=round(1000 * times[labels[1]], 3), differences=differences, status="done")
    say(f"{sum(same)}/{n} identical · {labels[0]} {format_duration(times[labels[0]])}{unit} · "
        f"{labels[1]} {format_duration(times[labels[1]])}{unit}")
    if differences:
        first = differences[0]
        say(f"First difference: test image #{first['image']}: {labels[0]} says {short(first[labels[0]], 30)}, "
            f"{labels[1]} says {short(first[labels[1]], 30)}. Same answer first, prettier second: fix this first.")
    return result


def _equal(a: object, b: object) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


# ----------------------------------------------------------------------------- stopwatch
class stopwatch:
    """Time a block of code and print how long it took.

        with stopwatch("plain loop"):
            predictions = [predict_nearest(img, train_images, train_labels) for img in test_images]
        Timed plain loop: 10.2 s

    stopwatch.table() prints every timing of this session; stopwatch.clear() forgets them.
    """
    history: list = []

    def __init__(self, label: str = "timed block") -> None:
        self.label = str(label)
        self.seconds: float | None = None
        self._start = 0.0

    def __enter__(self) -> "stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, kind, error, tb) -> bool:
        self.seconds = time.perf_counter() - self._start
        stopwatch.history.append((self.label, self.seconds))
        ending = "" if kind is None else " (stopped early)"
        say(f"Timed {self.label}: {format_duration(self.seconds)}{ending}")
        return False                     # never hide an error

    @classmethod
    def table(cls) -> None:
        """Print every timing so far, compared with the first one."""
        if not cls.history:
            say("No timings yet: use `with stopwatch(\"label\"): ...` first.")
            return
        width = max(len(label) for label, _ in cls.history) + 2
        base = cls.history[0][1]
        lines = ["Timings so far"]
        for label, seconds in cls.history:
            line = f"  {label.ljust(width)}{format_duration(seconds):>9}"
            if seconds > 0 and label != cls.history[0][0]:
                line += f"   ({base / seconds:.1f}× the speed of {cls.history[0][0]})"
            lines.append(line)
        say(*lines)

    @classmethod
    def clear(cls) -> None:
        cls.history.clear()


# ----------------------------------------------------------------------------- code size
def _source_of(obj: object) -> str | None:
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if path.suffix == ".py" and path.is_file():
            with open(path, encoding="utf-8") as f:
                return f.read()
        return None
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return None


def _code_lines(source: str) -> int:
    """Lines with code: not blank, not only a comment, not part of a docstring."""
    source = textwrap.dedent(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstring_lines.update(range(first.lineno, first.end_lineno + 1))
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENDMARKER, tokenize.ENCODING}
    code: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in skip:
            code.update(range(token.start[0], token.end[0] + 1))
    return len(code - docstring_lines)


def count_code_lines(*objs: object) -> int:
    """Count the lines of code in functions, classes, modules or .py files.

    Blank lines, comment-only lines and docstrings are not counted.
    """
    total = 0
    for obj in objs:
        source = _source_of(obj)
        if source is None:
            say(f"count_code_lines: I can't read the source of {short(obj, 40)}; it was not counted.")
            continue
        try:
            total += _code_lines(source)
        except (SyntaxError, tokenize.TokenError):
            say(f"count_code_lines: the source of {short(obj, 40)} doesn't parse; it was not counted.")
    return total
