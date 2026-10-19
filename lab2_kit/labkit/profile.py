"""profile_call(): where does the time go? A small, friendly profiler (Lab 2, ENG1).

    from labkit.profile import profile_call
    profile_call(predict_knn, test_images[0], train_images, train_labels)

It runs the call once with the profiler switched on and prints one line per function, with how many times
it ran and the time spent inside it: its own lines plus the built-ins, generator expressions, lambdas and
library code they run (sum, zip, sorted, ...), but not the other named functions it calls, which get their
own line. The top line is the hotspot: the function whose work takes the most time. That is where making
code faster actually pays off. Measure, don't guess.

The profiler slows code down (it records every call), so compare the shares, not the seconds.

(How it works: sys.setprofile() asks Python to call a small recorder function at every function call and
return in this thread. Python's own cProfile records every thread of the kernel since Python 3.12, so
Jupyter's background threads could blur its numbers; this one only watches your call.)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time

from .text import say, short

__all__ = ["profile_call"]

_INNER = ("<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<lambda>")


def _where(filename: str) -> str:
    """'your notebook', 'labkit/check.py', 'legacy_reader.py', or '' for built-ins."""
    if filename in ("~", ""):
        return ""
    if filename.startswith("<") or "ipykernel_" in filename or "ipython-input" in filename:
        return "your notebook"
    parts = filename.replace("\\", "/").split("/")
    if "labkit" in parts:
        return "/".join(parts[parts.index("labkit"):])
    return os.path.basename(filename)


def _name(key: tuple) -> str:
    """A readable name for a key (filename, first line, function name)."""
    filename, line, function = key
    if filename == "~":
        return f"{function} (built-in)"
    where = _where(filename)
    if function in _INNER:
        what = {"<genexpr>": "a generator expression", "<lambda>": "a lambda"}.get(function, "a comprehension")
        return f"{what} ({where}, line {line})" if where else what
    return f"{function} ({where})" if where else function


def _builtin_name(fn: object) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or "a built-in"
    owner = getattr(fn, "__self__", None)
    if owner is not None and not isinstance(owner, type(sys)) and name.count(".") == 0:
        name = f"{type(owner).__name__}.{name}"
    return name


def _is_python_library(filename: str) -> bool:
    if filename.startswith("<") or "ipykernel_" in filename:
        return False
    path = os.path.normcase(os.path.abspath(filename))
    return any(path.startswith(os.path.normcase(os.path.abspath(folder))) for folder in (sys.prefix, sys.base_prefix))


class _Recorder:
    """Counts calls and own time per function, for the calls made in THIS thread only."""

    def __init__(self) -> None:
        self.stats: dict = {}          # key -> [calls, own seconds]
        self.homes: dict = {}          # home key -> seconds spent inside it (its own work + what it runs)
        self.stack: list = []          # [key, start, child seconds, home key]
        self.codes: dict = {}          # code object -> (key, is it a named function of the user's code?)

    def _push(self, key: tuple, now: float, user_function: bool) -> None:
        parent_home = self.stack[-1][3] if self.stack else None
        home = key if user_function else (parent_home or key)
        self.stack.append([key, now, 0.0, home])

    def _pop(self, now: float) -> None:
        if not self.stack:
            return
        key, start, child, home = self.stack.pop()
        elapsed = now - start
        entry = self.stats.setdefault(key, [0, 0.0])
        entry[0] += 1
        entry[1] += elapsed - child
        self.homes[home] = self.homes.get(home, 0.0) + elapsed - child
        if self.stack:
            self.stack[-1][2] += elapsed

    def __call__(self, frame, event, arg) -> None:
        now = time.perf_counter()
        if event == "call":
            code = frame.f_code
            known = self.codes.get(code)
            if known is None:
                key = (code.co_filename, code.co_firstlineno, code.co_name)
                known = self.codes[code] = (key, code.co_name not in _INNER
                                            and not _is_python_library(code.co_filename))
            self._push(known[0], now, known[1])
        elif event == "c_call":
            self._push(("~", 0, _builtin_name(arg)), now, False)
        elif event in ("return", "c_return", "c_exception"):
            self._pop(now)


def profile_call(fn, *args, top: int = 5, **kwargs) -> None:
    """Run fn(*args, **kwargs) once under the profiler and print the top functions by own time, then the
    hotspot. Prints one friendly line instead if fn isn't written yet or raises an error."""
    from .check import is_stub, your_line

    name = getattr(fn, "__name__", "your function")
    if not callable(fn):
        say(f"profile_call needs a function, without brackets after its name (got {short(fn, 30)}).")
        return
    if is_stub(fn):
        say(f"profile_call: {name} is still a stub (only `pass`). Write it first, or borrow it with rescue().")
        return
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        top = 5
    recorder = _Recorder()
    error = None
    previous = sys.getprofile()
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sys.setprofile(recorder)
            try:
                fn(*args, **kwargs)
            finally:
                sys.setprofile(previous)
    except KeyboardInterrupt:
        say("profile_call stopped (you pressed Interrupt).")
        return
    except Exception as problem:                    # report it like a check would, never a traceback
        error = problem
    seconds = time.perf_counter() - start
    if error is not None:
        lines = [f"{name} raised {type(error).__name__}: {error}"]
        place = your_line(error, fn, always_name=True)
        if place:
            lines.append(f"{place[0]} {place[1]}")
        say(*lines, "Fix it first: the profile of a call that fails tells you nothing.")
        return
    stats = {key: value for key, value in recorder.stats.items() if not key[2].startswith("setprofile")}
    total = sum(own for _, own in stats.values()) or 1e-12
    # One row per "home": a named function, with the built-ins, generator expressions and lambdas it runs folded
    # in (so the top row and the hotspot are the same function, and its calls are the function's own calls).
    homes = {key: value for key, value in recorder.homes.items()
             if key[0] != "~" and not key[2].startswith("setprofile")}
    rows = sorted(((inside, stats.get(key, [0, 0.0])[0], _name(key)) for key, inside in homes.items()),
                  key=lambda row: -row[0])
    width = max([len(text) for _, _, text in rows[:top]] + [len("Function")]) + 2
    lines = [f"Profile of one call of {name}: {seconds:.2f} s with the profiler on (it slows code down: compare "
             "the shares, not the seconds).",
             "Time inside a function = its own lines + the built-ins and generator expressions they run "
             "(sum, zip, sorted, ...);",
             "the other functions of yours that it calls have their own line.",
             f"  {'Function'.ljust(width)}{'Calls':>11}{'Time inside':>14}{'Share':>8}"]
    for own, calls, text in rows[:top]:
        lines.append(f"  {text.ljust(width)}{calls:>11,}{own:>12.3f} s{own / total:>8.0%}")
    if rows:
        own, calls, text = rows[0]
        lines.append(f"Hotspot (the top line): {text}, {calls:,} calls, {own / total:.0%} of the time. Speed THAT "
                     "up first.")
    say(*lines)
