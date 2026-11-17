"""The notebook route for a command-line tool (Lab 4): run_cli(digits.main, "...") is `python digits.py ...`.

    run_cli(digits.main, "predict models/centroids.json samples/parcel_021.txt --show")

does what this does in a terminal, in the kit folder:

    python digits.py predict models/centroids.json samples/parcel_021.txt --show

It splits the command into words the way a terminal does (shlex.split), calls main(words), and shows what a
terminal would show: the output (stdout), every error line marked "stderr |", and the exit code. A SystemExit
(from argparse's --help or a usage error) ends the run as in a terminal, without stopping the notebook.

It always runs digits.py as you last SAVED it: before each run it reads the file again (fresh(), below), so
you never need to restart the kernel after an edit. You never need to change this file; open it if you're
curious how it works.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import io
import os
import re
import shlex
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Callable

from .text import hide_user_folder, say

__all__ = ["run_cli", "fresh", "split_command", "plain", "say_problem", "EXIT_MEANINGS"]

# Python 3.14's argparse colours --help when the environment asks for colour, and Jupyter kernels do
# (ipykernel sets FORCE_COLOR=1). PYTHON_COLORS=0 wins over FORCE_COLOR, so --help prints the same plain text
# on Python 3.10 to 3.14, in the notebook and in the terminal runs started from it. A real terminal is untouched.
os.environ.setdefault("PYTHON_COLORS", "0")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")   # ESC [ ... m: a colour code


def plain(text: str) -> str:
    """The text without terminal colour codes (ESC[...m), in case a tool coloured it anyway."""
    return _ANSI.sub("", text)

EXIT_MEANINGS = {0: "it worked", 1: "an error the user can fix", 2: "a usage error: the command line was wrong"}
_LABKIT = Path(__file__).resolve().parent

# (module name, function name) -> [the reference function put there by rescue(), the source of the file's own
# function when we first saw the rescue]. A rescue stays until you edit that function in the file.
_RESCUED: dict[tuple[str, str], list] = {}


# ============================================================================= reading the file again
def _is_reference(value: object) -> bool:
    return str(getattr(value, "__module__", "")).startswith("labkit.reference.")


def _function_sources(path: Path) -> dict[str, str]:
    """{name: source text} of every top-level function in a .py file ({} if it can't be read or parsed)."""
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return {}
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = ast.get_source_segment(text, node) or ""
    return found


def _student_module(module: object) -> bool:
    """True for a module loaded from a .py file of yours (not labkit, not Python's own, not the notebook)."""
    if not isinstance(module, types.ModuleType) or module.__name__ == "__main__":
        return False
    file = getattr(module, "__file__", None)
    if not file or not str(file).endswith(".py"):
        return False
    path = Path(file).resolve()
    if _LABKIT in path.parents:
        return False
    return not any(part in ("site-packages", "dist-packages") for part in path.parts) and \
        Path(sys.base_prefix).resolve() not in path.parents


def _problem_text(file: Path, error: BaseException) -> str:
    """One readable paragraph about an error raised while Python reads the file."""
    if isinstance(error, SyntaxError):
        where = f"line {error.lineno}" if error.lineno else "a line"
        text = (error.text or "").strip()
        detail = f"{type(error).__name__} on {where}: {error.msg}" + (f"\n   {where}: {text}" if text else "")
    else:
        line = None
        for frame in traceback.extract_tb(error.__traceback__):
            if Path(frame.filename).name == file.name:
                line = frame.lineno
        detail = f"{type(error).__name__}: {error}" + (f" (line {line})" if line else "")
    return (f"{file.name} can't be read: {detail}\n"
            f"Fix that line in {file.name}, save the file, then run this cell again. Until then, the notebook "
            "keeps the last version that worked.")


def fresh(module: types.ModuleType) -> tuple[types.ModuleType, str | None]:
    """Read a module of yours again (e.g. digits) as it is saved now. Returns (module, problem).

    problem is None, or a message when the file has an error (then the old version stays in use).
    A function you borrowed with rescue() (digits.save_model = rescue("C2")) is kept until you edit that
    function in the file; then your own version takes over, and a line says so.
    """
    if not _student_module(module):
        return module, None
    name = module.__name__
    file = Path(module.__file__)
    sources = None
    for attr, value in list(vars(module).items()):
        if callable(value) and _is_reference(value) and getattr(value, "__name__", None) == attr:
            if (name, attr) not in _RESCUED or _RESCUED[(name, attr)][0] is not value:
                sources = sources if sources is not None else _function_sources(file)
                _RESCUED[(name, attr)] = [value, sources.get(attr)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):          # a print() at the top of the file stays quiet
            module = importlib.reload(module)
    except Exception as error:                                     # noqa: BLE001 (an error in the student's file)
        return module, _problem_text(file, error)
    sources = _function_sources(file)
    edited = []
    for (mod_name, attr), (value, before) in list(_RESCUED.items()):
        if mod_name != name:
            continue
        now = sources.get(attr)
        if before is None or now == before:
            setattr(module, attr, value)                           # still the rescued one
        else:
            del _RESCUED[(mod_name, attr)]
            edited.append(attr)
    if edited:
        say(f"You edited {', '.join(edited)} in {file.name}: your own version is used again "
            "(instead of the rescued one).")
    return module, None


# ============================================================================= one message per cell
_LAST_PROBLEM: list = [None, None]          # [the last "can't be read" message, the cell run it was printed in]


def _cell_run() -> int | None:
    """The number of the cell run now (IPython's execution count), or None outside a notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return None if shell is None else int(shell.execution_count)
    except Exception:                        # noqa: BLE001 (no IPython)
        return None


def say_problem(problem: str) -> bool:
    """Print a "digits.py can't be read" message, but only once per cell run. Returns True if it printed it."""
    run = _cell_run()
    if run is not None and _LAST_PROBLEM == [problem, run]:
        return False
    _LAST_PROBLEM[:] = [problem, run]
    say(problem)
    return True


# ============================================================================= run_cli
def split_command(command: str | list[str] | tuple[str, ...]) -> list[str]:
    """The words of a command line, split the way a terminal splits them ("a b" stays one word)."""
    if isinstance(command, (list, tuple)):
        return [str(word) for word in command]
    if not isinstance(command, str):
        raise TypeError(f"the command must be text such as \"--help\", not {type(command).__name__}")
    return shlex.split(command, posix=True)


def _show(stdout: str, stderr: str) -> None:
    stdout, stderr = plain(stdout), plain(stderr)
    lines = stdout.rstrip("\n").split("\n") if stdout.strip() else []
    lines += [f"stderr | {line}" for line in stderr.rstrip("\n").split("\n")] if stderr.strip() else []
    if lines:
        say(*lines)


def _exit_code(value: Any, stderr: io.StringIO) -> int:
    """What `raise SystemExit(value)` does at the end of a script: None -> 0, an int -> itself, text -> 1."""
    if value is None:
        return 0
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    print(value, file=stderr)                    # Python prints a non-int exit value on stderr
    return 1


def run_cli(main_fn: Callable[..., Any], command: str | list[str] = "", *, prog: str | None = None) -> int:
    """Run main_fn(words) as if `python <file> <command>` was typed in a terminal; return the exit code.

    Prints "$ python digits.py <command>", then the output, the stderr lines (marked "stderr |") and the exit
    code with its meaning. A traceback (an error main() didn't catch) is shown as a terminal would show it,
    with your user folder as ~.
    """
    if not callable(main_fn):
        say(f"run_cli needs the function main itself, e.g. run_cli(digits.main, \"--help\") (got {main_fn!r}).")
        return 1
    try:
        words = split_command(command)
    except (TypeError, ValueError) as problem:
        say(f"run_cli: I can't split that command into words ({problem}). Check the quotes.")
        return 2
    module = sys.modules.get(getattr(main_fn, "__module__", "") or "")
    if module is not None and _student_module(module):
        module, problem = fresh(module)
        if problem:
            if not say_problem(problem):
                say(f"(not run: {Path(module.__file__).name} can't be read, see above)")
            return 1
        main_fn = getattr(module, getattr(main_fn, "__name__", "main"), main_fn)
    if prog is None:                              # your file's name; a rescued main() still stands for digits.py
        file = getattr(module, "__file__", None) if module is not None and _student_module(module) else None
        prog = Path(file).name if file else "digits.py"
    say("$ python " + " ".join([prog] + [shlex.quote(word) for word in words]))
    out, err = io.StringIO(), io.StringIO()
    note = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = main_fn(words)
        code = _exit_code(result, err)
        if result is None:
            note = ("Note: main() returned None. At the end of digits.py, raise SystemExit(main()) turns None "
                    "into exit code 0, so a script running it at 3 a.m. would think it worked. "
                    "main() should return 0 or 1 (C5).")
    except SystemExit as stop:                    # argparse (--help, a usage error) or sys.exit() somewhere
        code = _exit_code(stop.code, err)
    except KeyboardInterrupt:
        _show(out.getvalue(), err.getvalue())
        raise
    except Exception as error:                    # noqa: BLE001 (main() didn't catch it: a terminal shows it)
        tb = error.__traceback__
        while tb is not None and Path(tb.tb_frame.f_code.co_filename).resolve().parent == _LABKIT:
            tb = tb.tb_next                       # start at the first frame of your own code
        print(hide_user_folder("".join(traceback.format_exception(type(error), error, tb))).rstrip("\n"),
              file=err)
        code = 1
        note = ("Note: main() didn't catch this error, so the user sees a traceback (and exit code 1). For the "
                "errors you expect, print one line, error: ..., on stderr and return 1 (C5).")
    _show(out.getvalue(), err.getvalue())
    meaning = EXIT_MEANINGS.get(code, "an error")
    say(f"exit code {code} ({meaning})")
    if note:
        say(note)
    return code
