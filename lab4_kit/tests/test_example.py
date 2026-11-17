"""Example tests for digits.py (Lab 4, OPTIONAL ENG1). Copy this file to tests/test_digits.py and add yours.

Run them in the kit folder:
    python -m labkit.minitest tests             (standard library only)
    uv run --with pytest pytest -q              (pytest; needs internet the first time)
    python -m labkit.mutate tests/test_digits.py    (are your tests good enough to catch bugs?)

A good test: (1) sets up what it needs, in a temporary folder (tmp_path) so it never touches your real
files; (2) runs ONE command through digits.main([...]), exactly as a terminal would; (3) checks what a user
would check: the exit code, what was printed on stdout and on stderr, the files it wrote.
"""
import contextlib
import io

import digits


def run(words):
    """Run digits.main(words) and return (exit code, stdout, stderr), as a terminal would show them."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = digits.main(words)
        except SystemExit as stop:          # argparse: --help, or a command line it can't parse
            code = stop.code
    return code, out.getvalue(), err.getvalue()


def test_inspect_draws_one_image():
    code, out, err = run(["inspect", "data/mnist_test_2k.csv.gz", "--index", "0"])
    assert code == 0, f"exit code {code}; stderr: {err}"
    assert "Image 0 (label" in out
    assert err == ""


def test_an_index_outside_the_file_is_one_error_line(tmp_path):
    # tmp_path is a new, empty folder for this test (a pathlib.Path); it isn't used here, but your train
    # tests will need it: e.g. str(tmp_path / "model.json") as the --out file.
    code, out, err = run(["inspect", "data/mnist_test_2k.csv.gz", "--index", "5000"])
    assert code == 1, f"exit code {code}, expected 1"
    assert out == "", "an error must not appear on stdout"
    assert err.startswith("error:") and err.count("\n") == 1, f"stderr was {err!r}"
