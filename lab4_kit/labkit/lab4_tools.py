"""Helpers for the provided cells of Lab 4 · Ship It to the Depot (standard library only; the BLD2 helpers
load_web_app and serve_app_in_background import FastAPI and uvicorn only when they are called).

    from labkit.lab4_tools import check, check_all, reload_digits, call_command, show_model_file, terminal

    digits = reload_digits()        # import digits.py, or read it again after you saved it
    check("C1", digits.build_parser)    # labkit's check(), worded for digits.py (digits.<name>, "save the file")
    call_command(digits.cmd_train, "train data/mnist_train_10k.csv.gz --out models/centroids.json --force")
    show_model_file("models/centroids.json")
    terminal("python -m labkit.minitest tests")

Every helper here is already implemented: you only run it. None of them crashes on a task you haven't done
yet: it says what is missing and how to borrow the reference with rescue(). Open the file if you're curious.
"""
from __future__ import annotations

import ast
import contextlib
import csv
import importlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import cases
from .check import is_stub, run_call, your_line
from .cli import _function_sources, _problem_text, fresh, plain, say_problem, split_command
from .data import KIT_DIR, read_image_grid
from .drawing import prepare_drawing
from .env import current_lab
from .text import hide_user_folder, say, short

__all__ = ["check", "check_all", "for_digits", "reload_digits", "ready", "ready_command", "call_command", "try_model_file",
           "show_model_file", "terminal", "have_model", "demo_model", "read_samples", "copy_inbox",
           "show_inbox_result", "hygiene_report", "top3_scorer", "web_installed", "load_web_app",
           "serve_app_in_background", "handle_by_hand", "serve_in_background", "post_json", "ask",
           "score_strips", "show_strip"]

CORE_NAMES = ("cmd_inspect", "build_parser", "save_model", "load_model", "cmd_train", "cmd_predict", "main")


# ============================================================================= digits.py in the notebook
def _no_autoreload(name: str) -> None:
    """Leave `name` out of IPython's autoreload: reload_digits() and run_cli read the file again themselves
    (so line numbers in error reports stay right, and rescued functions stay in place)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("aimport", f"-{name}")
    except Exception:                 # noqa: BLE001 (no IPython, or no autoreload: nothing to switch off)
        pass


# ============================================================================= check() and check_all(), Lab 4 wording
def _digits_names() -> list[str]:
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    names = [task.name for task in lab_cases.tasks.values() if task.home == "digits"] if lab_cases else []
    return sorted(set(names + (["load_model"] if names else [])), key=len, reverse=True)


_NOT_STARTED = "Replace `pass` with your code, run the cell that defines it, then run this check."
_NOT_FOUND = "I can't find {name} yet: run the cell that defines it, then this check."


def for_digits(text: str) -> str:
    """The engine's wording, for functions that live in digits.py: check("C1", digits.build_parser),
    digits.build_parser = rescue("C1"), and "save digits.py" instead of "run the cell that defines it"."""
    names = _digits_names()
    lines = []
    for line in text.split("\n"):
        for name in names:
            line = re.sub(rf'check\("([A-Z0-9]+)", {name}([,)])', rf'check("\1", digits.{name}\2', line)
            line = re.sub(rf'(?<![\w.]){name} = rescue\(', f"digits.{name} = rescue(", line)
            if re.match(rf"^\S+ {name}: NOT STARTED\. ", line):
                line = line.replace(_NOT_STARTED, "Replace `pass` with your code in digits.py, save the file, then "
                                                  "run this check.")
                line = line.replace(_NOT_FOUND.format(name=name), f"digits.py has no {name} yet: write it, save the "
                                                                  "file, then run this check.")
        lines.append(line)
    return "\n".join(lines)


# C1 and C5 start with code already in them (inspect; the first two lines of main), so the engine can't see that
# they are untouched. A fingerprint of each stub as shipped (its code without comments and docstring) tells.
# build_lab4_master.py checks these against the student digits.py it builds, so they never drift.
STUB_FINGERPRINTS = {"build_parser": "38efe1c047d05610", "main": "f3930531159b10e2"}
_STARTED_TEXT = {"build_parser": "Add train and predict to build_parser in digits.py",
                 "main": "Complete main in digits.py (the dispatch dict, try/except, the exit codes)"}


def fingerprint(source: str) -> str:
    """A short fingerprint of a function's code: the same for the same code, whatever its comments."""
    import hashlib
    import textwrap

    node = ast.parse(textwrap.dedent(source)).body[0]
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    code = ast.unparse(ast.Module(body=body, type_ignores=[]))      # the same text on Python 3.10 to 3.14
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def untouched(fn: object) -> bool:
    """True if fn is build_parser or main of digits.py exactly as shipped (only comments may have changed)."""
    name = getattr(fn, "__name__", "")
    module = sys.modules.get(getattr(fn, "__module__", "") or "")
    if name not in STUB_FINGERPRINTS or module is None or not getattr(module, "__file__", None):
        return False
    source = _function_sources(Path(module.__file__)).get(name)
    try:
        return source is not None and fingerprint(source) == STUB_FINGERPRINTS[name]
    except (SyntaxError, IndexError, AttributeError):
        return False


def _home_task(task_id: str):
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    return lab_cases.find(str(task_id).strip().upper()) if lab_cases is not None else None


def check(task_id: str, obj: object = None, *, verbose: bool = False, style: bool = False) -> bool:
    """labkit's check(), worded for Lab 4 (your functions live in digits.py). Returns True on PASS."""
    from .check import check as engine_check

    task = _home_task(task_id)
    if task is not None and obj is not None and untouched(obj):
        say(f"{task.id} {task.name}: NOT STARTED. {_STARTED_TEXT[obj.__name__]}, save the file, then run this "
            "check.")
        return False
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ok = engine_check(task_id, obj, verbose=verbose, style=style)
    text = buffer.getvalue().rstrip("\n")
    if task is not None and task.home in ("digits", "web_app"):    # the code lives in a file, not in a cell
        file = f"{task.home}.py"
        text = text.replace("fix it and run the cell with def again, then this check",
                            f"fix it in {file}, save the file, then run this check again")
        text = text.replace("run the cell with def again, then this check", f"save {file}, then run this check again")
    if text.strip():
        say(for_digits(text))
    if task is not None and task.home == "digits" and getattr(obj, "__module__", "") == "__main__":
        name = getattr(obj, "__name__", task.name)
        say(f"Note: this {name} lives in the notebook. digits.py (main() and the terminal) still has its own: "
            f"paste yours into digits.py, save, then run check(\"{task.id}\", digits.{name}).")
    return ok


def check_all(*, verbose: bool = False) -> dict[str, str]:
    """labkit's check_all(), after reading digits.py again (as saved), worded for Lab 4."""
    from .check import check_all as engine_check_all

    digits = reload_digits()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        statuses = engine_check_all(verbose=verbose)
    text = buffer.getvalue().rstrip("\n")
    for task_id, name in (("C1", "build_parser"), ("C5", "main")):
        if statuses.get(task_id) == "NOT YET" and untouched(getattr(digits, name, None)):
            statuses[task_id] = "NOT STARTED"
            text = re.sub(rf"^(\s*{task_id} +{name} +)NOT YET +.*$", r"\g<1>NOT STARTED", text, flags=re.M)
    if text.strip():
        say(for_digits(text))
    return statuses


def _placeholder(name: str) -> types.ModuleType:
    """A stand-in module whose CORE functions are all still `pass`, so checks say NOT STARTED, not crash."""
    module = types.ModuleType(name)
    source = "\n".join(f"def {fn}(*args, **kwargs):\n    pass\n" for fn in CORE_NAMES)
    exec(compile(source, f"<{name}.py is missing>", "exec"), module.__dict__)
    return module


def reload_digits(name: str = "digits") -> types.ModuleType:
    """Import digits.py from the kit folder, or read it again as you last saved it, and return the module.

    Run it after every save (each check cell does it for you). If the file has an error, it prints the line
    to fix and keeps the last version that worked. Functions you borrowed with rescue() stay until you edit
    them in the file.
    """
    file = KIT_DIR / f"{name}.py"
    _no_autoreload(name)
    module = sys.modules.get(name)
    same_file = module is not None and Path(getattr(module, "__file__", "") or "").resolve() == file.resolve()
    if not same_file:
        if not file.is_file():
            say(f"I can't find {name}.py in the kit folder ({hide_user_folder(str(KIT_DIR))}). Re-extract the "
                "kit, or copy digits.py back from the zip, then run this cell again.")
            return _placeholder(name)
        if str(KIT_DIR) not in sys.path:
            sys.path.insert(0, str(KIT_DIR))
        sys.modules.pop(name, None)
        try:
            return importlib.import_module(name)
        except Exception as error:                     # noqa: BLE001 (an error in the student's file)
            say_problem(_problem_text(file, error))
            return _placeholder(name)
    module, problem = fresh(module)
    if problem:
        say_problem(problem)
    return module


def _task_of(fn: object):
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    if lab_cases is None:
        return None
    return lab_cases.by_name(getattr(fn, "__name__", ""))


def _needs(fn: object) -> str:
    """'cmd_train (C3). Not there yet? Run digits.cmd_train = rescue("C3") in a new cell, ...'"""
    name = getattr(fn, "__name__", "this function")
    task = _task_of(fn)
    if task is None:
        return f"{name} isn't written yet: write it in digits.py, save the file, then run this cell again."
    if task.rescue:
        borrow = f'digits.{name} = rescue("{task.id}")'
    else:
        return f"This cell needs {name} ({task.id}): write it, save the file, then run this cell again."
    return (f"This cell needs {name} ({task.id}). Not there yet? Run {borrow} in a new cell, then run this "
            "cell again.")


def ready(*functions: object) -> bool:
    """True if every function is written (not a stub); otherwise print what's missing and return False.

    Provided cells use it so that they never crash on a task you haven't done yet:  if ready(predict_with): ...
    """
    missing = []
    for fn in functions:
        if not callable(fn):
            missing.append(f"{short(fn, 30)} (not a function)")
        elif is_stub(fn):
            if getattr(fn, "__module__", "") == "__main__":
                task = _task_of(fn)
                label = f"{fn.__name__} ({task.id})" if task is not None else fn.__name__
                missing.append(f"{label}: write it in its cell above and run that cell")
            else:
                missing.append(_needs(fn))
    if not missing:
        return True
    say("This cell needs:", *[f"   - {item}" for item in missing], "Then run this cell again.")
    return False


def _fresh_function(fn: Callable) -> Callable:
    """The same function, read again from the file it lives in (if that is a file of yours)."""
    module = sys.modules.get(getattr(fn, "__module__", "") or "")
    if module is None:
        return fn
    module, problem = fresh(module)
    if problem:
        say_problem(problem)
    return getattr(module, getattr(fn, "__name__", ""), fn)


def call_command(command_fn: Callable, command: str) -> Any:
    """Turn `command` into args with the FINISHED parser (so this works before C1 passes), call YOUR command
    function with them, and print what it returns. Returns that value (None if it couldn't run)."""
    from .reference import lab4 as finished           # the reference parser only: your function does the work

    if not callable(command_fn):
        say(f"call_command needs a command function, e.g. call_command(digits.cmd_train, \"...\"), not "
            f"{short(command_fn, 40)}.")
        return None
    command_fn = _fresh_function(command_fn)
    name = getattr(command_fn, "__name__", "your command")
    if is_stub(command_fn):
        say(_needs(command_fn))
        return None
    try:
        args = finished.build_parser().parse_args(split_command(command))
    except SystemExit:
        say(f"The command line \"{command}\" isn't valid for digits.py: check the words and options.")
        return None
    say(f"{name}(args) with args from: {command}")
    attempt = run_call(command_fn, (args,), show=f"{name}(args)", track=False)
    if attempt.printed.strip():
        say(*attempt.printed.rstrip("\n").split("\n"))
        say(f"(Your {name} printed the lines above itself. It should return its text instead: main() prints it.)")
    if attempt.error is not None:
        say(f"{name} raised {attempt.error_text}")
        place = your_line(attempt.error, command_fn, always_name=True)
        if place:
            say(f"{place[0]} {place[1]}")
        partner = {"cmd_train": "save_model", "cmd_predict": "load_model"}.get(name)
        helper = getattr(command_fn, "__globals__", {}).get(partner) if partner else None
        if helper is not None and is_stub(helper):
            say(f"{name} uses {partner} (C2), which is still `pass`. " + _needs(helper))
        elif name == "cmd_predict" and isinstance(attempt.error, FileNotFoundError):
            say("Is the model there? models/centroids.json is made by the train cell (C3) above: run it first.")
        task = _task_of(command_fn)
        if task is not None:
            say(f"Run check(\"{task.id}\", digits.{name}) to see what {task.id} needs.")
        return None
    if isinstance(attempt.got, str):
        say(attempt.got)
    else:
        say(f"{name} returned {short(attempt.got, 60)}: a command function returns its text (a str) for main() "
            "to print.")
    return attempt.got


# ============================================================================= model files (C2, C3)
def _brief(value: Any) -> str:
    if isinstance(value, dict):
        inner = ", ".join(f"{json.dumps(k)}: {short(v, 32)}" for k, v in list(value.items())[:2])
        return "{" + inner + (f", …}} ({len(value)} keys)" if len(value) > 2 else "}")
    return short(value, 60)


def show_model_file(path: str | Path = "models/centroids.json") -> None:
    """Show a saved model file: its size, how it starts as text, and every key with a short value."""
    file = Path(path)
    if not file.is_file():
        say(f"There is no file {file.as_posix()} yet: the train step (C3) writes it.")
        return
    data = file.read_bytes()
    text = data.decode("utf-8", errors="replace")
    say(f"{file.as_posix()}: {len(data) / 1024:.0f} KB · {len(data):,} bytes · one line of JSON text",
        f"It starts with: {text[:110]}…")
    try:
        model = json.loads(text)
    except json.JSONDecodeError as error:
        say(f"It isn't valid JSON ({error}): json.dump(model, f) writes JSON; str(model) or repr(model) don't.")
        return
    if not isinstance(model, dict):
        say(f"It holds {type(model).__name__}, not a dict.")
        return
    say("Keys and values:", *[f"   {json.dumps(key):<14} {_brief(value)}" for key, value in model.items()])


def try_model_file(digits: types.ModuleType, path: str | Path = "output/demo_model.json") -> None:
    """C2's view: save a real model with YOUR save_model, show the file, and load it back with YOUR load_model."""
    from .reference import lab4 as finished
    from .reference.lab1 import fit_centroids

    save, load = getattr(digits, "save_model", None), getattr(digits, "load_model", None)
    if not callable(save) or is_stub(save):
        say(_needs(save) if callable(save) else "This cell needs digits.save_model (C2).")
        return
    if not callable(load) or is_stub(load):
        say("digits.py has lost the provided load_model: copy it back from labkit/reference/lab4.py, save, then run "
            "this cell again.")
        return
    from .data import cached_digits
    images, labels = cached_digits("mnist_train_10k.csv.gz", 1000)
    model = finished.new_model(fit_centroids(images, labels), n_train=len(images))
    target = Path(path)
    attempt = run_call(save, (model, target), {"force": True}, show="save_model(...)", track=False)
    if attempt.error is not None:
        say(f"save_model(model, Path(\"{target.as_posix()}\"), force=True) raised {attempt.error_text}")
        place = your_line(attempt.error, save, always_name=True)
        if place:
            say(f"{place[0]} {place[1]}")
        say("Run check(\"C2\", digits.save_model) to see what C2 needs.")
        return
    say(f"Your save_model wrote a model learnt from {len(images):,} training images:")
    show_model_file(target)
    back = run_call(load, (target,), show="load_model(...)", track=False)
    if back.error is not None:
        say(f"The provided load_model raised {back.error_text} on that file. Run check(\"C2\", digits.save_model).")
        return
    same = back.got == model
    say("The provided load_model read it back: "
        + ("the same dict (a round trip: save, then load, changes nothing)." if same else
           "a different value: " + short(back.got, 50) + "."))


# ============================================================================= a terminal, from the notebook
def terminal(command: str, *, timeout: float = 120.0) -> int:
    """Run `python ...` in the kit folder as a terminal would, and show its output and exit code.

    Only commands that start with python (or py, python3) are accepted: it runs them with this notebook's
    Python, so it works on Windows, macOS, Linux and Colab alike.
    """
    try:
        words = shlex.split(command, posix=True)
    except ValueError as problem:
        say(f"terminal: I can't split that command into words ({problem}).")
        return 2
    if not words or words[0] not in ("python", "python3", "py"):
        say("terminal() only runs Python commands, e.g. terminal(\"python digits.py --help\").")
        return 2
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHON_COLORS="0")   # plain --help on 3.14
    say(f"$ {command}")
    try:
        proc = subprocess.run([sys.executable] + words[1:], cwd=str(KIT_DIR), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        say(f"(stopped after {timeout:.0f} s)")
        return 1
    proc.stdout, proc.stderr = plain(proc.stdout), plain(proc.stderr)
    lines = proc.stdout.rstrip("\n").split("\n") if proc.stdout.strip() else []
    lines += [f"stderr | {line}" for line in proc.stderr.rstrip("\n").split("\n")] if proc.stderr.strip() else []
    if lines:
        say(*[hide_user_folder(line) for line in lines])
    say(f"exit code {proc.returncode}")
    return proc.returncode


def have_model(path: str | Path = "models/centroids.json") -> bool:
    """True if the model file is there; otherwise say which cell makes it and return False."""
    if Path(path).is_file():
        return True
    say(f"This cell needs {Path(path).as_posix()}: run the train cell of Putting It All Together first (it saves it).")
    return False


def demo_model(path: str | Path = "output/ai_demo_model.json") -> str:
    """A model file for a demonstration (made by the finished tool on 1,000 images if it isn't there yet)."""
    from .data import cached_digits
    from .reference import lab4 as finished
    from .reference.lab1 import fit_centroids

    file = Path(path)
    if not file.is_file():
        images, labels = cached_digits("mnist_train_10k.csv.gz", 1000)
        finished.save_model(finished.new_model(fit_centroids(images, labels), n_train=len(images)), file, force=True)
    return file.as_posix()


def read_samples(main_fn: Callable, model: str = "models/centroids.json", folder: str = "samples") -> None:
    """Run `predict MODEL FILE` through YOUR main() for every parcel in samples/, one line per parcel."""
    import contextlib

    main_fn = _fresh_function(main_fn)
    if is_stub(main_fn):
        say(_needs(main_fn))
        return
    files = sorted(p for p in Path(folder).glob("parcel_*") if p.suffix in (".csv", ".txt"))
    if not have_model(model):
        return
    say(f"python digits.py predict {model} <each file in {folder}/>:")
    for number, path in enumerate(files):
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main_fn(["predict", model, path.as_posix()])
        except SystemExit as stop:
            code = stop.code
        except Exception as error:                    # noqa: BLE001 (main didn't catch it)
            code, err = 1, io.StringIO(f"{type(error).__name__}: {error}")
        text = (out.getvalue().strip() or err.getvalue().strip() or "(nothing printed)").split("\n")[-1]
        if number == 0 and code not in (0, 1):
            say(f"   {path.name}: exit code {code}: {text[:90]}",
                "Your main() can't run predict yet (C1 and C5): run check(\"C5\", digits.main) to see why.")
            return
        say(f"   {path.name:<16} {text[:60]:<30} exit code {code}")


# ============================================================================= SCI1, BLD1: a new subcommand
_NEW_COMMANDS = {"evaluate": ("cmd_evaluate", "SCI1"), "batch": ("cmd_batch", "BLD1")}


def _subcommands(parser: Any) -> set[str]:
    import argparse
    found: set[str] = set()
    for action in getattr(parser, "_actions", []):
        if isinstance(action, argparse._SubParsersAction):          # noqa: SLF001 (argparse keeps them here)
            found |= set(action.choices)
    return found


def ready_command(digits: types.ModuleType, subcommand: str) -> bool:
    """True if digits.py has all three parts of an OPTIONAL subcommand (evaluate: SCI1, batch: BLD1): the
    function, its subparser in build_parser, and its entry in main's dispatch dict. Otherwise one line says
    which parts are missing, and it returns False."""
    name, task_id = _NEW_COMMANDS[subcommand]
    fn = getattr(digits, name, None)
    missing = []
    if not callable(fn) or is_stub(fn):
        missing.append(f"write {name}")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            choices = _subcommands(digits.build_parser())
    except Exception:                                  # noqa: BLE001 (build_parser has a bug: C1 says which)
        choices = set()
    if subcommand not in choices:
        missing.append(f"add the {subcommand} subparser to build_parser")
    main = getattr(digits, "main", None)
    if not str(getattr(main, "__module__", "")).startswith("labkit.reference."):
        source = _function_sources(Path(getattr(digits, "__file__", "") or "digits.py")).get("main", "")
        if f'"{subcommand}"' not in source and f"'{subcommand}'" not in source:
            missing.append(f'add "{subcommand}": {name} to the dispatch dict in main')
    if missing:
        say(f"This cell runs `{subcommand}` ({task_id}). Still to do in digits.py: " + "; ".join(missing)
            + ". Save the file, then run this cell again.")
        return False
    return True


# ============================================================================= BLD1: the inbox
def copy_inbox(target: str | Path = "output/inbox_tonight") -> Path:
    """A fresh copy of the kit's inbox/ (30 files), so the batch can move files without emptying the kit's."""
    source = KIT_DIR / "inbox"
    folder = Path(target)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    count = 0
    for file in sorted(source.glob("*")):
        if file.is_file():
            shutil.copy2(file, folder / file.name)
            count += 1
    say(f"Copied {count} files from inbox/ to {folder.as_posix()}/ (the kit's inbox/ stays as it is).")
    return folder


def show_inbox_result(folder: str | Path, out: str | Path) -> None:
    """After a batch run: what is left in the folder, what was archived, and the first rows of the CSV."""
    folder, out = Path(folder), Path(out)
    left = sorted(p.name for p in folder.glob("*") if p.is_file())
    archives = sorted(p for p in (folder / "archive").glob("*") if p.is_dir()) if (folder / "archive").is_dir() else []
    say(f"Files still in {folder.as_posix()}/: {len(left)}" + (f" ({', '.join(left[:5])}…)" if left else ""))
    for archive in archives:
        say(f"Archived in {archive.relative_to(folder).as_posix()}/: {sum(1 for p in archive.iterdir())} files")
    backups = sorted(out.parent.glob(f"{out.stem}_*{out.suffix}"))
    if backups:
        say("Backups: " + ", ".join(b.name for b in backups))
    if not out.is_file():
        say(f"There is no {out.as_posix()} yet.")
        return
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    say(f"{out.as_posix()}: {max(len(rows) - 1, 0)} rows. The first 6 lines:", *["   " + ",".join(r) for r in rows[:6]])


# ============================================================================= ENG2: hygiene
def hygiene_report() -> dict[str, bool]:
    """A checklist of the ENG2 hygiene steps in the kit folder (it only reads; it changes nothing)."""
    def lines_of(name: str) -> list[str]:
        path = KIT_DIR / name
        return path.read_text(encoding="utf-8", errors="replace").split("\n") if path.is_file() else []

    ignore = {line.strip() for line in lines_of(".gitignore")}
    readme = "\n".join(lines_of("README.md"))
    usage = readme.split("## Usage", 1)[1].split("\n## ", 1)[0] if "## Usage" in readme else ""
    status = {
        ".gitignore lists data/": "data/" in ignore,
        ".gitignore lists models/": "models/" in ignore,
        "a git repository (.git/)": (KIT_DIR / ".git").is_dir(),
        "README.md: the Usage section is written (no TODO left)": bool(usage.strip()) and "TODO" not in usage,
        "README.md: an AI-use declaration section": "## AI-use declaration" in readme,
        "tests/test_digits.py exists": (KIT_DIR / "tests" / "test_digits.py").is_file(),
    }
    width = max(len(item) for item in status)
    say(*[f"   {item:<{width}}  {'yes' if ok else 'not yet'}" for item, ok in status.items()])
    return status


# ============================================================================= BLD2: a web API for the canvas
# The FastAPI route: web_app.py (the student writes the Drawing model and the predict endpoint), served by uvicorn.
# The fallback without the web group: Python's own http.server with handle_by_hand, the same answers, checked by hand.
WEB_PACKAGES = ("fastapi", "uvicorn")
MAX_BODY = 20_000                     # the by-hand server: bytes; a drawing is about 3,000


def top3_scorer(model_path: str | Path = "models/centroids.json") -> Callable[[list[int]], list]:
    """A function pixels -> [[label, score], ...] for the 3 closest centroids of a saved model.

    The pixels are a raw drawing from tools/draw_digit.html: prepare_drawing() crops, scales and centres them
    first. score = (distance of the best) / (distance of this one): 1.0 for the best, smaller for the others.
    The model file is read at the first call (so web_app.py can be imported before the model exists).
    """
    from .reference import lab4 as finished
    from .reference.lab1 import squared_distance

    cache: dict[str, dict] = {}

    def top3(pixels: list[int]) -> list:
        if "centroids" not in cache:
            cache["centroids"] = finished.centroids_of(finished.load_model(Path(model_path)))
        image = prepare_drawing(list(pixels))
        distances = sorted((squared_distance(image, centroid), label)
                           for label, centroid in cache["centroids"].items())
        best = max(distances[0][0], 1e-9)
        return [[label, round(best / max(d, 1e-9), 2)] for d, label in distances[:3]]

    return top3


def web_installed() -> bool:
    """True if FastAPI and uvicorn are installed; otherwise print one line saying how to get them."""
    import importlib.util

    missing = [name for name in WEB_PACKAGES if importlib.util.find_spec(name) is None]
    if not missing:
        return True
    say(f"FastAPI isn't installed in this Python (missing: {', '.join(missing)}). It's optional: only BLD2 uses "
        "it. To install it: `uv sync --group web` in the kit folder (it needs internet; add `--group nn` if you "
        "installed that for Lab 5: uv keeps only the groups you name), or `pip install fastapi uvicorn`; then "
        "restart the kernel. Meanwhile, the by-hand server further down works without it.")
    return False


def load_web_app(name: str = "web_app") -> types.ModuleType | None:
    """Import web_app.py (BLD2) from the kit folder, or read it again as you last saved it.

    Returns None, with one line saying why, if FastAPI is not installed or the file has an error.
    """
    if not web_installed():
        return None
    if not (KIT_DIR / f"{name}.py").is_file():
        say(f"I can't find {name}.py in the kit folder ({hide_user_folder(str(KIT_DIR))}). Re-extract the kit, or "
            f"copy {name}.py back from the zip, then run this cell again.")
        return None
    module = reload_digits(name)
    return module if hasattr(module, "app") else None


def serve_app_in_background(app: Any, *, port: int = 0):
    """Run a FastAPI app with uvicorn on 127.0.0.1 (this computer only), in a thread of this notebook.

    Returns (url, stop): open the url in a browser on this computer, and call stop() when you're done.
    port=0 picks a free port. In a terminal, `python web_app.py` does the same on port 8000.
    """
    import time

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("the web server did not start (is another program using the port?)")
    bound = server.servers[0].sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{bound}/"

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=10)
        say(f"Stopped the server at {url}")

    say(f"Serving web_app at {url} (only this computer can reach it)")
    return url, stop


def handle_by_hand(body: bytes, top3_fn: Callable) -> tuple[int, dict]:
    """POST /predict without FastAPI: every check that the Drawing model of web_app.py declares, written by hand.

    413 if the body is over MAX_BODY bytes (checked first: never parse something that big); 422 if it isn't UTF-8
    JSON, isn't an object, or has no "pixels" list of 784 whole numbers from 0 to 255 (FastAPI's status for a body
    that breaks the contract); else 200 and {"top3": top3_fn(pixels)}.
    """
    if len(body) > MAX_BODY:
        return 413, {"detail": f"the request has {len(body):,} bytes; the limit is {MAX_BODY:,}"}
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 422, {"detail": "the body must be JSON (UTF-8)"}
    pixels = data.get("pixels") if isinstance(data, dict) else None
    if not isinstance(pixels, list) or len(pixels) != 784:
        return 422, {"detail": "\"pixels\" must be a list of 784 numbers"}
    for value in pixels:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            return 422, {"detail": "every pixel must be a whole number from 0 to 255"}
    return 200, {"top3": top3_fn(pixels)}


def serve_in_background(handler_fn: Callable | None = None, top3_fn: Callable | None = None, *, port: int = 0):
    """The fallback without FastAPI: Python's own http.server for tools/draw_digit.html, on 127.0.0.1, in a thread.

    GET / sends the drawing page; POST /predict passes the request body (at most 20,001 bytes are read) to
    handler_fn(body, top3_fn) (default: handle_by_hand, with the model in models/centroids.json), which returns
    (status, dict); the dict goes back as JSON. Returns (url, stop): call stop() when you're done.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    handler_fn = handler_fn or handle_by_hand
    top3_fn = top3_fn or top3_scorer(KIT_DIR / "models" / "centroids.json")
    page = (KIT_DIR / "tools" / "draw_digit.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "digits"
        sys_version = ""

        def log_message(self, *args: Any) -> None:     # quiet: no line per request in the notebook
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:                      # noqa: N802 (the name http.server expects)
            if self.path in ("/", "/draw_digit.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b'{"detail": "Not Found"}', "application/json")

        def do_POST(self) -> None:                     # noqa: N802
            if self.path != "/predict":
                self._send(404, b'{"detail": "Not Found"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            body = self.rfile.read(min(max(length, 0), MAX_BODY + 1))
            try:
                status, payload = handler_fn(body, top3_fn)
                data = json.dumps(payload).encode("utf-8")
            except Exception as error:                 # noqa: BLE001 (the handler raised)
                status, data = 500, json.dumps({"detail": f"the handler raised {type(error).__name__}"}).encode()
            self._send(int(status), data, "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    def stop() -> None:
        server.shutdown()
        server.server_close()
        say(f"Stopped the server at {url}")

    say(f"Serving tools/draw_digit.html at {url} with http.server (only this computer can reach it)")
    return url, stop


def post_json(url: str, body: bytes | dict | list) -> tuple[int, Any]:
    """POST a body to a URL of the local server; return (status, the JSON answer or the text)."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return status, raw.decode("utf-8", errors="replace")


def _why(answer: Any) -> str:
    """The first reason in an error answer: FastAPI's {"detail": [{"loc": [...], "msg": ...}]} or {"detail": "..."}."""
    detail = answer.get("detail") if isinstance(answer, dict) else answer
    if isinstance(detail, list) and detail and isinstance(detail[0], dict):
        first = detail[0]
        loc = [part for part in first.get("loc", []) if part != "body"]
        where = ""
        if loc and isinstance(loc[0], str):           # ("body", "pixels", 0) -> pixels[0]; ("body", 0) -> nothing
            where = loc[0] + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc[1:])
        more = f" (and {len(detail) - 1} more)" if len(detail) > 1 else ""
        return f"{where + ': ' if where else ''}{first.get('msg', '')}{more}"
    return detail if isinstance(detail, str) else short(detail, 90)


def ask(url: str, body: bytes | dict | list, label: str = "") -> tuple[int, Any]:
    """POST a body to the local server and print one line: the status and the answer (or why it was refused)."""
    status, answer = post_json(url, body)
    text = short(answer, 90) if status == 200 else _why(answer)
    say(f"{(label + ':').ljust(18) if label else ''}{status} {text}")
    return status, answer


# ============================================================================= the Challenge: postcode strips
def show_strip(path: str | Path) -> None:
    """Draw a postcode strip (28 rows x 140 columns) in text: '#' for ink."""
    pixels = read_image_grid(path)
    width = len(pixels) // 28
    say(f"{Path(path).name}: 28 x {width}")
    for r in range(28):
        row = pixels[r * width:(r + 1) * width]
        say("".join("#" if value >= 128 else "." for value in row))


def score_strips(read_strip_fn: Callable, model: str | Path = "models/centroids.json",
                 folder: str | Path = "strips") -> dict:
    """Run YOUR read_strip(path, centroids) on every strip and compare with strips/strips_truth.csv.

    The centroids come from the model file (read with the finished load_model). Prints one line per strip (read
    as, truth, right or not, a valid Madrid postcode or not) and the totals.
    """
    from .reference import lab4 as finished

    folder = Path(folder)
    if is_stub(read_strip_fn):
        say("This cell needs read_strip (the Challenge): write it in the cell above, run it, then this cell.")
        return {}
    if not have_model(model):
        return {}
    centroids = finished.centroids_of(finished.load_model(Path(model)))
    with open(folder / "strips_truth.csv", encoding="utf-8", newline="") as f:
        truth = {row["file"]: row["postcode"] for row in csv.DictReader(f)}
    valid_file = KIT_DIR / "data" / "valid_postcodes.txt"
    valid = {line.strip() for line in valid_file.read_text(encoding="utf-8").split("\n") if line.strip()}
    right = valid_count = 0
    say(f"{'strip':<14}{'read as':<10}{'truth':<8}{'right?':<8}valid postcode?")
    for name in sorted(truth):
        attempt = run_call(read_strip_fn, (folder / name, centroids), show=f"read_strip({name!r}, centroids)",
                           track=False)
        if attempt.error is not None:
            got = f"raised {type(attempt.error).__name__}"
        else:
            got = str(attempt.got)
        ok = got == truth[name]
        right += ok
        valid_count += got in valid
        say(f"{name:<14}{got[:9]:<10}{truth[name]:<8}{'yes' if ok else 'no':<8}{'yes' if got in valid else 'no'}")
    say(f"Read correctly: {right} of {len(truth)} strips · valid postcodes: {valid_count} of {len(truth)}")
    return {"right": right, "total": len(truth), "valid": valid_count}
