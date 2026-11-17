"""Are your tests any good? Run them against six buggy versions of digits.py (Lab 4, ENG1). In the kit folder:

    python -m labkit.mutate tests/test_digits.py

A test is only useful if it FAILS when the code is wrong. This tool first runs your tests on a correct
digits.py (labkit/reference/lab4.py): they must all pass. Then it runs them on six "mutants": copies of that
correct digits.py with ONE small bug each (labkit/mutants/lab4.py). A mutant is caught when at least one of
your tests fails on it. Your mutation score is the share of mutants caught.

Your tests must `import digits` and call digits.main([...]) or the digits functions: the tool puts each
version in the place of digits.py while your tests run (a test that starts `python digits.py` in a
subprocess would run your own file instead, so it can't catch anything here).
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path

from .minitest import run_tests
from .mutants.lab4 import MUTANTS

__all__ = ["mutation_score", "main"]

REFERENCE = Path(__file__).resolve().parent / "reference" / "lab4.py"


def _digits_from(source: str) -> types.ModuleType:
    """A module called digits, made from `source` (the reference text, possibly with one bug)."""
    module = types.ModuleType("digits")
    module.__file__ = str(REFERENCE)                  # so its KIT_FOLDER is the kit folder
    exec(compile(source, "digits.py (a version made by labkit.mutate)", "exec"), module.__dict__)
    return module


def _run_with(source: str, tests: Path) -> dict:
    saved = sys.modules.get("digits")
    sys.modules["digits"] = _digits_from(source)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_tests(tests, quiet=True)
    finally:
        if saved is None:
            sys.modules.pop("digits", None)
        else:
            sys.modules["digits"] = saved


def mutation_score(tests: str | Path, *, quiet: bool = False) -> dict:
    """Run the tests on the correct digits.py and on every mutant. Returns a dict with the results."""
    say = (lambda *a: None) if quiet else print
    tests = Path(tests)
    if not tests.exists():
        say(f"{tests.as_posix()} doesn't exist yet. Copy tests/test_example.py to tests/test_digits.py, write "
            "your tests in it, save it, then run this again.")
        return {"status": "no tests"}
    source = REFERENCE.read_text(encoding="utf-8")
    for description, old, _ in MUTANTS:
        if source.count(old) != 1:
            say(f"labkit problem: the mutant '{description}' doesn't fit the reference. Tell Javier.")
            return {"status": "broken"}
    base = _run_with(source, tests)
    n_tests = len(base["passed"]) + len(base["failed"])
    if n_tests == 0:
        say(f"No tests found in {tests.as_posix()}: name your test functions test_... (or use unittest).")
        return {"status": "no tests"}
    if base["failed"]:
        say(f"Your tests must pass on a correct digits.py first. {len(base['failed'])} of {n_tests} failed:")
        for name, reason in base["failed"]:
            say(f"   FAIL  {name}: {reason}")
        say("Fix those tests (what does the correct digits.py really do?), then run this again.")
        return {"status": "fails on correct", "failed": base["failed"]}
    say(f"Your {n_tests} test{'s' if n_tests != 1 else ''} pass on a correct digits.py. Now the {len(MUTANTS)} "
        "buggy versions:")
    caught, survived = [], []
    for description, old, new in MUTANTS:
        result = _run_with(source.replace(old, new, 1), tests)
        if result["failed"]:
            caught.append(description)
            say(f"   caught    {description}  (by {result['failed'][0][0].split('::')[-1]})")
        else:
            survived.append(description)
            say(f"   survived  {description}")
    score = len(caught) / len(MUTANTS)
    say(f"Mutants caught: {len(caught)} of {len(MUTANTS)} · mutation score {score:.0%}")
    if survived:
        say("For each survivor: which command would show the bug? Write a test that runs it and checks the exit "
            "code, stdout, stderr or the file it writes.")
    return {"status": "done", "tests": n_tests, "caught": caught, "survived": survived, "score": score}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m labkit.mutate tests/test_digits.py")
        return 2
    result = mutation_score(argv[0])
    return 0 if result.get("status") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
