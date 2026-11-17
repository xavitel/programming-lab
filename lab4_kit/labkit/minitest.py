"""A tiny test runner (standard library only), for when pytest isn't installed. In the kit folder:

    python -m labkit.minitest tests                    # every tests/test_*.py file
    python -m labkit.minitest tests/test_digits.py     # one file

What it runs, in each file:
- every function whose name starts with test_ (the pytest style: plain assert). A parameter named tmp_path
  gets a new empty folder (a pathlib.Path), deleted afterwards, as in pytest;
- every unittest.TestCase class (the unittest style; python -m unittest runs those too).

It prints PASS or FAIL for each test, the reason for each failure, and a summary. Exit code 0 if every test
passed, 1 if one failed, 2 if there was nothing to run. Whatever a test prints is hidden, so the report
stays readable. The same tests also run with pytest: uv run --with pytest pytest -q (needs internet once).
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import shutil
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from typing import Any

__all__ = ["run_tests", "main"]


def _test_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(target.glob("test_*.py"))
    return []


def _load(path: Path, number: int):
    """Import a test file as a fresh module (never a cached one, so each run sees the current code)."""
    name = f"_minitest_{number}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _reason(error: BaseException) -> str:
    """'AssertionError: exit code was 0' plus the line of the test that failed."""
    text = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
    frames = [f for f in traceback.extract_tb(error.__traceback__) if Path(f.filename).name.startswith("test_")]
    if frames:
        text += f" (line {frames[-1].lineno}: {(frames[-1].line or '').strip()})"
    return text


def _run_function(fn: Any) -> str | None:
    """Run one pytest-style test; None if it passed, else the reason."""
    folder = None
    kwargs = {}
    try:
        if "tmp_path" in inspect.signature(fn).parameters:
            folder = Path(tempfile.mkdtemp(prefix="minitest_"))
            kwargs["tmp_path"] = folder
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn(**kwargs)
        return None
    except KeyboardInterrupt:
        raise
    except BaseException as error:               # noqa: BLE001 (SystemExit from a test counts as a failure)
        return _reason(error)
    finally:
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)


def run_tests(target: str | Path, *, quiet: bool = False) -> dict:
    """Run every test in a file or folder; return {"passed": [...], "failed": [(name, reason)], "files": n}."""
    target = Path(target)
    files = _test_files(target)
    result: dict = {"passed": [], "failed": [], "files": len(files)}
    say = (lambda *a: None) if quiet else print
    if not files:
        say(f"No test files found at {target.as_posix()} (names must start with test_ and end with .py).")
        return result
    folder = str((files[0].parent.parent if files[0].parent.name == "tests" else files[0].parent).resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)            # so a test can `import digits` from the kit folder
    for number, path in enumerate(files):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                module = _load(path, number)
        except Exception as error:                # noqa: BLE001 (the file itself has an error)
            result["failed"].append((path.name, f"the file can't be imported: {_reason(error)}"))
            say(f"FAIL  {path.name}: the file can't be imported: {_reason(error)}")
            continue
        for name, fn in vars(module).items():
            if name.startswith("test_") and inspect.isfunction(fn) and fn.__module__ == module.__name__:
                problem = _run_function(fn)
                label = f"{path.name}::{name}"
                if problem is None:
                    result["passed"].append(label)
                    say(f"PASS  {label}")
                else:
                    result["failed"].append((label, problem))
                    say(f"FAIL  {label}: {problem}")
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        if suite.countTestCases():
            outcome = unittest.TestResult()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                suite.run(outcome)
            bad = {str(test): text for test, text in outcome.failures + outcome.errors}
            for test in _flatten(suite):
                label = f"{path.name}::{test.id().split('.', 1)[-1]}"
                if str(test) in bad:
                    reason = bad[str(test)].strip().split("\n")[-1]
                    result["failed"].append((label, reason))
                    say(f"FAIL  {label}: {reason}")
                else:
                    result["passed"].append(label)
                    say(f"PASS  {label}")
    total = len(result["passed"]) + len(result["failed"])
    say(f"{len(result['passed'])} passed, {len(result['failed'])} failed ({total} tests in {len(files)} file"
        f"{'s' if len(files) != 1 else ''})" if total else "No tests found: name your test functions test_...")
    return result


def _flatten(suite: unittest.TestSuite) -> list:
    tests = []
    for item in suite:
        tests += _flatten(item) if isinstance(item, unittest.TestSuite) else [item]
    return tests


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m labkit.minitest FOLDER_OR_FILE   (e.g. python -m labkit.minitest tests)")
        return 2
    result = run_tests(argv[0])
    if not result["passed"] and not result["failed"]:
        return 2
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
