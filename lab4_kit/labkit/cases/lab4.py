"""Checks for Lab 4 · Ship It to the Depot (blueprint §5.4, plan §3).

The student's functions live in digits.py, so every CORE and OPTIONAL task has home="digits": check_all()
finds them there, and the notebook passes them as check("C1", digits.build_parser). Expected values that come
from the reference are stored in cases/lab4_data.json (generators/gen_cases.py); the hooks that need a correct
version at check time ask for it with ctx.reference(...). This file holds no solution code.

Tasks, in notebook order: W cmd_inspect (worked example) · C1 build_parser · C2 save_model (load_model is provided) ·
C3 cmd_train · C4 cmd_predict · C5 main · CP3 the terminal run (python digits.py in a subprocess) · OPTIONAL
ENG1 check_model, SCI1 cmd_evaluate, BLD1 cmd_batch · ADVANCED (sanity checks, no rescue) ENG2 setup_logging,
SCI2 predict_with, BLD2 predict (web_app.py, FastAPI; checked only where FastAPI is installed). Then the catch-up notebook between Lab 4 and Lab 5, bridge_4to5.ipynb,
which runs inside this kit (defined in the notebook, not in digits.py): BR1 CentroidModel, BR2 score.

Every test has a human title (TEST n: ...) and labelled inputs. Tasks checked by a hook run a fixed number of
tests: after the first failure the rest count as not passed, so "k of n" never changes its n.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePath

from ..check import PASS, is_stub, likely_cause, run_task, your_line
from ..cli import _is_reference, plain
from ..data import KIT_DIR
from ..spec import Case, Detector, Digits, Raises, Task, approx, custom
from ..text import hide_user_folder, short

DATA_FILES = ("valid_postcodes.txt",)       # the Challenge validates postcode strips against it
FINALE = "Putting It All Together"

TRAIN_FILE = "data/mnist_train_10k.csv.gz"
TEST_FILE = "data/mnist_test_2k.csv.gz"
TRAIN = Digits("mnist_train_10k.csv.gz", limit=500)     # the model the predict checks use (500 images)
TEST = Digits("mnist_test_2k.csv.gz", limit=20)

MISSING = "(missing)"
NEXT_HINT = "fix it in digits.py, save the file, then run this check again (it re-reads the file). Or open Hint 1."


# ============================================================================= small helpers
def _ns(**values) -> argparse.Namespace:
    return argparse.Namespace(**values)


def _partner(first: object, name: str):
    """A function that `first` works with (load_model for cmd_predict, ...).

    Normally from the module `first` lives in. When `first` is a rescued reference function, the partner comes
    from the student's live digits module (as the notebook and the terminal would use it), so a check never
    tests the reference partner while the student's own one is in digits.py.
    """
    if _is_reference(first):
        live = getattr(sys.modules.get("digits"), name, None)
        if callable(live):
            return live
    found = getattr(first, "__globals__", {}).get(name)
    return found if callable(found) else None


def _source_lines(error: BaseException | None) -> list[str]:
    """The source lines of every frame of an error's traceback (to tell which call went wrong)."""
    if error is None:
        return []
    import traceback
    return [(frame.line or "") for frame in traceback.extract_tb(error.__traceback__)]


def _first_cause(attempt, rules) -> str | None:
    for rule in rules:
        try:
            if rule.test(attempt):
                return rule.message
        except Exception:                     # a rule that doesn't apply to this attempt
            continue
    return None


def _preview(text: str, limit: int = 34) -> str:
    text = text.strip("\n")
    if not text:
        return "(nothing)"
    first = text.split("\n")[0]
    lines = text.count("\n") + 1
    shown = first if len(first) <= limit else first[: limit - 1] + "…"
    return repr(shown) + (f" (+{lines - 1} more lines)" if lines > 1 else "")


class _Said(str):
    """A description printed as it is (no quotes) in an Expected / Your result line."""

    def __repr__(self) -> str:
        return str(self)


class _Run:
    """A fixed number of tests for a hook: test(...) records one; after a failure the rest count as not passed."""

    def __init__(self, ctx, planned: int, obj: object = None) -> None:
        self.ctx, self.planned, self.obj, self.done, self.failed = ctx, planned, obj, 0, False

    def test(self, ok: bool, title: str, call: str, expected: str, got: str, cause: str | None = None,
             attempt=None, next_step: str | None = None) -> bool:
        if self.failed:
            return False
        self.done += 1
        if not ok and cause is None and attempt is not None:
            expected = getattr(attempt, "expected", None)
            if attempt.error is not None or expected is not None:     # never "the spec asks for None" by default
                cause = likely_cause(attempt, expected)               # the engine's generic causes (KeyError, ...)
        passed = self.ctx.require(bool(ok), case=call, title=title, expected=_Said(expected), got=_Said(got),
                                  cause=None if ok else cause, next_step=None if ok else (next_step or NEXT_HINT))
        if not passed:
            self.failed = True
            failure = self.ctx.failure
            if attempt is not None and attempt.error is not None and failure is not None:
                place = your_line(attempt.error, self.obj, always_name=True)
                if place:
                    failure.extra.append(place)
            if attempt is not None and attempt.printed.strip() and failure is not None:
                failure.extra.append(("Printed:", _preview(attempt.printed, 50)))
            self.ctx.total += max(self.planned - self.done, 0)
        return passed


def _outcome(attempt) -> str:
    if attempt.error is not None:
        return f"raised {attempt.error_text}"[:150]
    return f"returned {short(attempt.got, 60)}"


def _fixture_model(ctx) -> dict:
    """A real model (500 training images), made with the reference new_model and fit_centroids."""
    new_model, fit = ctx.reference("new_model"), ctx.reference("fit_centroids")
    return new_model(fit(TRAIN.images, TRAIN.labels), n_train=len(TRAIN.labels))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _grid(image: list[int]) -> str:
    return "\n".join(" ".join(map(str, image[r * 28:(r + 1) * 28])) for r in range(28)) + "\n"


def _valid_model() -> dict:
    """A complete version-1 model with easy numbers: centroid d is 784 times the value d."""
    return {"format": "garabato-centroids", "version": 1, "created": "2026-11-18T03:00:00", "n_train": 10,
            "image_size": 28, "centroids": {str(d): [float(d)] * 784 for d in range(10)}}


def _broken(change) -> dict:
    model = _valid_model()
    change(model)
    return model


# ============================================================================= W · cmd_inspect
W = Task(
    "W", "cmd_inspect", title="Anatomy of a command: inspect", home="digits",
    practise="the anatomy of a command: args in, a string out (Theory S7)",
    expected_label="Expected text", inputs_unchanged=False,
    cases=[
        Case(call=lambda fn: fn(_ns(data=TEST_FILE, index=3)), key="inspect-3", highlight=True,
             title="inspect data/mnist_test_2k.csv.gz --index 3: the file, the labels, and image 3",
             show_inputs={"Call": "cmd_inspect(Namespace(data=\"data/mnist_test_2k.csv.gz\", index=3))"}),
        Case(call=lambda fn: fn(_ns(data=TEST_FILE, index=5000)), key="inspect-5000",
             expected=Raises(ValueError, contains="--index must be from 0 to 1999"), highlight=True,
             title="An index outside the file: a ValueError that says what's allowed (main() prints it)",
             show_inputs={"Call": "cmd_inspect(Namespace(data=\"data/mnist_test_2k.csv.gz\", index=5000))"}),
    ],
    detectors=[
        Detector(lambda t: t.got is None and t.printed.strip(),
                 "cmd_inspect printed its text instead of returning it. Commands return; main() prints."),
    ],
)


# ============================================================================= C1 · build_parser
def _args(words: list[str], keys: tuple[str, ...]):
    """call: build the student's parser, parse `words`, and return the values of `keys` (or "(missing)")."""
    def call(fn):
        parser = fn()
        parsed = vars(parser.parse_args(list(words)))
        # type=Path is fine too: the cmd_* functions call Path() themselves, so a path compares as its text
        return {key: os.fspath(value) if isinstance(value, PurePath) else value
                for key, value in ((key, parsed.get(key, MISSING)) for key in keys)}
    return call


def _exit_code(words: list[str]):
    """call: the exit code argparse gives for `words` (2 for a usage error), or what it parsed if it didn't stop."""
    def call(fn):
        parser = fn()
        try:
            parsed = parser.parse_args(list(words))
        except SystemExit as stop:
            return stop.code
        return f"no error: it parsed {short(vars(parsed), 40)}"
    return call


def _c1(words: list[str], keys: tuple[str, ...], expected: dict, title: str, key: str, highlight: bool = False):
    return Case(call=_args(words, keys), expected=expected, key=key, title=title, highlight=highlight,
                show_inputs={"Command line": "digits " + " ".join(words)}, expected_label="Expected args")


def _c1_usage(words: list[str], title: str, key: str):
    return Case(call=_exit_code(words), expected=2, key=key, title=title, expected_label="Expected exit code",
                show_inputs={"Command line": "digits " + " ".join(words) if words else "digits   (no subcommand)"})


def _usage_said(t, text: str) -> bool:
    return t.raised(SystemExit) and text in t.stderr


def _no_error(t) -> bool:
    return isinstance(t.got, str) and t.got.startswith("no error")


C1 = Task(
    "C1", "build_parser", title="Words in, args out", home="digits",
    stub_fingerprint="38efe1c047d05610",     # the stub as shipped (= lab4_tools.STUB_FINGERPRINTS; the build checks)
    practise="argparse: subcommands, positional arguments, options with a type, flags (Theory S7)",
    minimum_path=True, inputs_unchanged=False,
    cases=[
        _c1(["train", "d.csv", "--out", "m.json"], ("command", "data", "out"),
            {"command": "train", "data": "d.csv", "out": "m.json"}, key="train-required", highlight=True,
            title="train with its required parts: the data file (positional) and --out"),
        _c1(["train", "d.csv", "--out", "m.json"], ("limit", "force"), {"limit": None, "force": False},
            key="train-defaults", title="train without --limit and --force: their defaults, None and False"),
        _c1(["train", "d.csv", "--out", "m.json", "--limit", "500", "--force"], ("limit", "force"),
            {"limit": 500, "force": True}, key="train-options", highlight=True,
            title="train --limit 500 --force: the number 500 (an int, not text) and True"),
        _c1(["predict", "m.json", "p.txt", "--show"], ("command", "model", "image", "show"),
            {"command": "predict", "model": "m.json", "image": "p.txt", "show": True}, key="predict-show",
            highlight=True, title="predict with two positional arguments, model and image, and the flag --show"),
        _c1(["predict", "m.json", "p.txt"], ("show",), {"show": False}, key="predict-no-show",
            title="predict without --show: show is False"),
        _c1(["inspect", "d.csv", "--index", "3"], ("command", "data", "index"),
            {"command": "inspect", "data": "d.csv", "index": 3}, key="inspect",
            title="inspect (already written in the stub) still works"),
        _c1_usage(["train", "d.csv"], "train without --out: a usage error, exit code 2", key="usage-no-out"),
        _c1_usage(["train", "d.csv", "--out", "m.json", "--limit", "ten"],
                  "--limit ten: a usage error (a limit must be a whole number)", key="usage-limit"),
        _c1_usage([], "No subcommand at all: a usage error", key="usage-empty"),
        Case(call=lambda fn: fn().prog, expected="digits", key="prog", title="The program's name in --help: digits",
             show_inputs={"Call": "build_parser().prog"}, expected_label="Expected"),
    ],
    detectors=[
        Detector(lambda t: t.raised(AttributeError) and "NoneType" in str(t.error),
                 "build_parser returned None: its last line must be `return parser` (one indent in, not inside "
                 "an if)."),
        Detector(lambda t: _usage_said(t, "invalid choice: 'train'"),
                 "your parser has no train subcommand yet: train = subparsers.add_parser(\"train\", help=...) "
                 "(copy the pattern of inspect).", "open Hint 2 below (the plan)."),
        Detector(lambda t: _usage_said(t, "invalid choice: 'predict'"),
                 "your parser has no predict subcommand yet: predict = subparsers.add_parser(\"predict\", ...).",
                 "open Hint 2 below (the plan)."),
        Detector(lambda t: _usage_said(t, "unrecognized arguments: --limit") or _usage_said(t, "unrecognized "
                 "arguments: 500"), "train has no --limit option yet: train.add_argument(\"--limit\", type=int, "
                 "default=None)."),
        Detector(lambda t: _usage_said(t, "unrecognized arguments: --force") or _usage_said(t, "unrecognized "
                 "arguments: --show"), "a flag is missing: add_argument(\"--force\", action=\"store_true\") on "
                 "train, and add_argument(\"--show\", action=\"store_true\") on predict."),
        Detector(lambda t: _usage_said(t, "the following arguments are required: --model") or _usage_said(
                 t, "unrecognized arguments: m.json"),
                 "model and image are positional arguments: add_argument(\"model\") and add_argument(\"image\"), "
                 "without dashes. Dashes make an option (--show), which is typed with its name."),
        Detector(lambda t: _usage_said(t, "expected one argument") and "--force" in t.stderr,
                 "--force is a flag (it takes no value): add_argument(\"--force\", action=\"store_true\")."),
        Detector(lambda t: _usage_said(t, "unrecognized arguments") and t.title.startswith("predict"),
                 "predict takes two positional arguments, model and image: add_argument(\"model\") and "
                 "add_argument(\"image\"), without dashes (an option with dashes must be typed with its name)."),
        Detector(lambda t: t.raised(SystemExit),
                 "argparse rejected this command line (a usage error, exit code 2): your parser doesn't accept these "
                 "words yet. Compare your add_argument lines with the table on the card."),
        Detector(lambda t: isinstance(t.got, dict) and t.got.get("command") == MISSING,
                 "args has no command: give add_subparsers(...) dest=\"command\". Without it, main() can't tell "
                 "which subcommand was typed."),
        Detector(lambda t: isinstance(t.got, dict) and t.got.get("limit") == "500",
                 "--limit arrived as the text '500'. Add type=int to add_argument(\"--limit\", ...): argparse "
                 "then converts it, and rejects --limit ten with a usage error."),
        Detector(lambda t: isinstance(t.got, dict) and (t.got.get("model") == MISSING or t.got.get("image") ==
                 MISSING),
                 "args.model or args.image is missing: name the positional arguments \"model\" and \"image\" in "
                 "lower case (metavar=\"MODEL\" only changes how --help shows them)."),
        Detector(lambda t: isinstance(t.got, dict) and t.got.get("out") == MISSING,
                 "args.out is missing: the option must be called --out (add_argument(\"--out\", required=True))."),
        Detector(lambda t: isinstance(t.got, dict) and t.got.get("data") == MISSING,
                 "args.data is missing: train's positional argument must be called \"data\"."),
        Detector(lambda t: isinstance(t.got, dict) and t.title.startswith("train without --limit")
                 and t.got.get("limit") not in (None, MISSING),
                 "without --limit, args.limit must be None (\"use every image\"): add_argument(\"--limit\", "
                 "type=int, default=None)."),
        Detector(lambda t: isinstance(t.got, dict) and (("force" in t.got and t.got["force"] is None)
                                                          or ("show" in t.got and t.got["show"] is None)),
                 "a flag is None when it isn't typed: use action=\"store_true\" (False unless typed, True when "
                 "typed)."),
        Detector(lambda t: _no_error(t) and "without --out" in t.title,
                 "train accepted a command line without --out: add required=True to add_argument(\"--out\", ...)."),
        Detector(lambda t: _no_error(t) and "--limit ten" in t.title,
                 "--limit ten was accepted: add type=int, so argparse rejects text that isn't a whole number."),
        Detector(lambda t: _no_error(t),
                 "no subcommand was accepted: add_subparsers(dest=\"command\", required=True) (already in the "
                 "stub: is it still there?)."),
    ],
)


# ============================================================================= C2 · save_model (load_model is provided)
C2_RULES = [
    Detector(lambda t: t.raised(FileExistsError) and getattr(t.error, "errno", None) is not None,
             "the folder already exists, and mkdir refuses an existing folder unless you say it's fine: "
             "path.parent.mkdir(parents=True, exist_ok=True)."),
    Detector(lambda t: t.raised(FileNotFoundError) and "models" in str(t.error),
             "the folder models/ doesn't exist yet, so open() can't create the file in it. Create it first: "
             "path.parent.mkdir(parents=True, exist_ok=True)."),
    Detector(lambda t: t.raised(TypeError) and "not JSON serializable" in str(t.error),
             "json.dump can only save dicts, lists, str, numbers, True/False and None. Save the model dict itself."),
    Detector(lambda t: t.raised(AttributeError) and "'str' object has no attribute" in str(t.error),
             "path arrived as text (a str): turn it into a Path first, path = Path(path), then use path.exists() "
             "and path.parent."),
]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _c2(fn, ctx) -> None:
    save = fn
    if getattr(fn, "__name__", "") == "load_model":           # load_model is provided: check save_model
        save = _partner(fn, "save_model") or fn
    run = _Run(ctx, 5, save)
    model = _valid_model()
    other = dict(_valid_model(), n_train=99)
    with ctx.temp_dir(chdir=True):
        path = Path("models") / "m.json"
        call = 'save_model(model, Path("models/m.json"))   (models/ doesn\'t exist yet)'
        a = ctx.call(save, model, path, show=call, track=False)
        if not run.test(a.error is None and path.is_file(), "save_model writes the file, creating its folder",
                        call, "no error, and models/m.json exists",
                        _outcome(a) if a.error else ("no file models/m.json" if not path.is_file() else "ok"),
                        _first_cause(a, C2_RULES) or (None if a.error is not None else
                                                      "save_model returned without writing the file: open(path, "
                                                      "\"w\", encoding=\"utf-8\") and json.dump(model, f)."),
                        attempt=a):
            return
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            same, what = on_disk == model, "the same dict" if on_disk == model else f"a different value: {short(on_disk, 30)}"
        except (ValueError, UnicodeDecodeError) as error:
            same, what = False, f"not JSON ({type(error).__name__})"
        if not run.test(same, "The file is JSON: json.load(file) gives back the same dict (so load_model can "
                        "read it)", "json.load(open(\"models/m.json\"))", "the same dict", what,
                        "write the model with json.dump(model, f). str(model) or f.write(repr(model)) write Python's "
                        "own text (single quotes, True), which isn't JSON."):
            return
        second = Path("models") / "second.json"
        call = 'save_model(model, Path("models/second.json"))   (models/ exists now)'
        b = ctx.call(save, model, second, show=call, track=False)
        if not run.test(b.error is None and second.is_file(), "A second model in the same folder (models/ is "
                        "already there)", call, "no error, and models/second.json exists",
                        _outcome(b) if b.error else ("no file models/second.json" if not second.is_file() else "ok"),
                        _first_cause(b, C2_RULES), attempt=b):
            return
        before = path.read_text(encoding="utf-8")
        c = ctx.call(save, other, path, show="save_model(other_model, Path(\"models/m.json\"))", track=False)
        refused = c.raised(FileExistsError) and "--force" in str(c.error)
        unchanged = path.read_text(encoding="utf-8") == before
        if not run.test(refused and unchanged, "The file exists: no overwrite without force (FileExistsError)",
                        "save_model(other_model, Path(\"models/m.json\"))   (the file is already there)",
                        "raises FileExistsError('models/m.json exists; use --force to overwrite'), file unchanged",
                        _outcome(c) + ("" if unchanged else " · the file was overwritten"),
                        (_first_cause(c, C2_RULES[:1]) or
                         "your FileExistsError needs the message from the docstring: f\"{path} exists; use --force "
                         "to overwrite\" (it tells the night shift what to do)." if c.raised(FileExistsError) else
                         "check first: if path.exists() and not force, raise FileExistsError(f\"{path} exists; use "
                         "--force to overwrite\"), before opening the file (opening with \"w\" empties it)."),
                        attempt=c if c.error is not None and not c.raised(FileExistsError) else None):
            return
        d = ctx.call(save, other, path, force=True, show="save_model(other_model, Path(\"models/m.json\"), force=True)",
                     track=False)
        overwritten = _read_json(path) == other
        run.test(d.error is None and overwritten, "force=True overwrites the file",
                 "save_model(other_model, Path(\"models/m.json\"), force=True)", "no error, the new model in "
                 "the file", _outcome(d) if d.error else ("the new model" if overwritten else "the old model"),
                 _first_cause(d, C2_RULES) or
                 "with force=True, save_model must overwrite: raise only when path.exists() and not force.",
                 attempt=d)


C2 = Task(
    "C2", "save_model", title="A model is a file", home="digits",
    practise="json, pathlib and raise: saving a file safely, and refusing to overwrite it (Theory S5, S7)",
    minimum_path=True, inputs_unchanged=False, custom=_c2, detectors=C2_RULES,
)


# ============================================================================= C3 · cmd_train
C3_RULES = [
    Detector(lambda t: t.got is None and t.error is None and t.printed.strip(),
             "you printed the message; return it instead (return f\"Trained on ...\"). main() prints what a "
             "command returns, and a test can only read what is returned."),
    Detector(lambda t: t.got is None and t.error is None,
             "cmd_train returned None: end it with return f\"Trained on {len(images):,} images in ...\"."),
    Detector(lambda t: t.raised(TypeError) and "limit must be a whole number" in str(t.error),
             "args.limit is text: that's C1 (add type=int to --limit). In this check, pass limit=args.limit as it is."),
    Detector(lambda t: t.raised(AttributeError) and "Namespace" in str(t.error),
             "args has no such attribute: the names are args.data, args.out, args.limit and args.force."),
    Detector(lambda t: t.raised(FileExistsError) and getattr(t.args[0], "force", False),
             "args.force is True, but save_model still refused: pass force=args.force to save_model."),
    Detector(lambda t: t.raised(AttributeError) and "'str' object has no attribute 'stat'" in str(t.error),
             "args.out is text (a str), and text has no .stat(): Path(args.out).stat().st_size."),
    Detector(lambda t: t.raised(TypeError) and "n_train" in str(t.error) and "missing" in str(t.error),
             "new_model needs two things: the centroids and how many images they came from: "
             "new_model(fit_centroids(images, labels), n_train=len(images))."),
]

TRAIN_MESSAGE = re.compile(r"^Trained on 200 images in \d+(?:\.\d+)? s: saved models[/\\]c\.json "
                           r"\((\d+(?:\.\d+)?) KB\)$")


def _size_cause(message: str, kb: float) -> str | None:
    """None if the KB in the message is the file's size; otherwise what is wrong with it."""
    match = TRAIN_MESSAGE.match(message)
    if match is None:
        return "shape"
    said = float(match.group(1))
    if abs(said - kb) <= 1:
        return None
    if kb > 0 and abs(said / kb - 1024) < 60:
        return (f"the message says {match.group(1)} KB, but that is the size in bytes: divide st_size by 1024 "
                f"(this file is about {kb:.0f} KB).")
    return f"the message says {match.group(1)} KB, but the file is about {kb:.0f} KB: size_kb = " \
           "Path(args.out).stat().st_size / 1024."


def _c3(fn, ctx) -> None:
    run = _Run(ctx, 6, fn)
    shown = "cmd_train(args) for: train data/mnist_train_10k.csv.gz --out models/c.json --limit 200"
    with ctx.temp_dir(chdir=True):
        out = Path("models") / "c.json"
        args = _ns(data=TRAIN_FILE, out="models/c.json", limit=200, force=False)
        a = ctx.call(fn, args, show=shown, track=False)
        a.args = (args,)
        if not run.test(a.error is None and isinstance(a.got, str) and not a.printed.strip(),
                        "It returns its message as a str, and prints nothing itself", shown,
                        "a str (main() prints it)", _outcome(a) + (" · it printed" if a.printed.strip() else ""),
                        _first_cause(a, C3_RULES) or (None if a.error is not None else
                                                      "print() inside a command shows text that a test or a "
                                                      "script can't use: return the message instead."),
                        attempt=a):
            return
        message = a.got.strip()
        kb = out.stat().st_size / 1024 if out.is_file() else 0.0
        size = _size_cause(message, kb)
        if not run.test(size is None, "The message: Trained on 200 images in … s: saved models/c.json (… KB), with "
                        "the file's real size", shown, f"Trained on 200 images in S.S s: saved models/c.json "
                        f"({kb:.0f} KB)", repr(message[:90]),
                        "--limit 200 was ignored, so it learnt from every image: pass limit=args.limit to load_digits."
                        if message.startswith("Trained on 10,000") or message.startswith("Trained on 10000") else
                        size if size != "shape" else
                        "build it with one f-string: f\"Trained on {len(images):,} images in {seconds:.1f} s: saved "
                        "{args.out} ({size_kb:.0f} KB)\", with size_kb = Path(args.out).stat().st_size / 1024."):
            return
        try:
            model = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            model = None
        keys = ("format", "version", "created", "n_train", "image_size", "centroids")
        complete = isinstance(model, dict) and all(k in model for k in keys) and len(model.get("centroids", {})) == 10
        if not run.test(complete, "models/c.json is a complete model (format, version, …, 10 centroids)",
                        "json.load(open(\"models/c.json\"))", "a dict with " + ", ".join(keys),
                        "no readable file" if model is None else f"keys {sorted(model)[:6]}",
                        "build the model with the provided new_model(fit_centroids(images, labels), "
                        "n_train=len(images)): it has every key load_model needs."):
            return
        if not run.test(model.get("n_train") == 200, "--limit 200: the model learnt from 200 images",
                        "json.load(open(\"models/c.json\"))[\"n_train\"]", "200", repr(model.get("n_train")),
                        "pass limit=args.limit to load_digits, and n_train=len(images) to new_model."):
            return
        b = ctx.call(fn, args, show=shown + "   (again: models/c.json exists)", track=False)
        if not run.test(b.raised(FileExistsError), "Again, without --force: FileExistsError (main() turns it into "
                        "an error line)", shown + "   (again: models/c.json exists)", "raises FileExistsError",
                        _outcome(b),
                        "pass force=args.force to save_model (args.force is False here), and let its "
                        "FileExistsError through: main() catches it.", attempt=b if b.error is not None else None):
            return
        forced = _ns(data=TRAIN_FILE, out="models/c.json", limit=100, force=True)
        c = ctx.call(fn, forced, show="cmd_train(args) for: train ... --out models/c.json --limit 100 --force",
                     track=False)
        c.args = (forced,)
        try:
            n_after = json.loads(out.read_text(encoding="utf-8")).get("n_train")
        except (OSError, ValueError, AttributeError):
            n_after = None
        run.test(c.error is None and n_after == 100, "--force: it overwrites the model (now n_train is 100)",
                 "cmd_train(args) for: train ... --out models/c.json --limit 100 --force",
                 "no error, n_train 100 in the file", _outcome(c) if c.error else f"n_train {n_after!r}",
                 _first_cause(c, C3_RULES) or ("pass force=args.force to save_model." if c.error is None
                                               or c.raised(FileExistsError) else None), attempt=c)


C3 = Task(
    "C3", "cmd_train", title="Train from the command line", home="digits",
    practise="composing provided functions, and returning a message instead of printing it (Theory S7)",
    minimum_path=True, depends_on=("C2",), inputs_unchanged=False, custom=_c3, detectors=C3_RULES,
)


# ============================================================================= C4 · cmd_predict
def _whole_model(t) -> bool:
    """predict_centroid(image, model): the whole model dict where the centroids belong."""
    return t.raised(TypeError, KeyError, AttributeError, ValueError) and any(
        "predict_centroid(" in line and "centroids" not in line.split("predict_centroid(", 1)[1]
        for line in _source_lines(t.error))


C4_RULES = [
    Detector(_whole_model,
             "predict_centroid needs the centroids, not the whole model dict (which also holds \"format\", "
             "\"version\", ...): predict_centroid(image, centroids_of(model))."),
    Detector(lambda t: t.got is None and t.error is None and t.printed.strip(),
             "you printed the prediction; return the text instead: return f\"Prediction: {label}\"."),
    Detector(lambda t: isinstance(t.got, int),
             "you returned the label itself; return the text for the user: f\"Prediction: {label}\"."),
    Detector(lambda t: t.raised(ValueError) and "expected one line of pixels" in str(t.error),
             "a .txt image was read with read_image_csv. Choose the reader by Path(args.image).suffix: \".csv\" "
             "-> read_image_csv, \".txt\" -> read_image_grid."),
    Detector(lambda t: t.raised(ValueError) and "found 28 lines" not in str(t.error) and "unsupported image "
             "type" in str(t.error),
             "a .csv or .txt image was refused: Path(...).suffix includes the dot, so compare with \".csv\" and "
             "\".txt\" (not \"csv\")."),
    Detector(lambda t: t.raised(TypeError) and "NoneType" in str(t.error),
             "a value is None: does your reader line assign image? Do you use the model that load_model returns?"),
]


def _c4(fn, ctx) -> None:
    run = _Run(ctx, 5, fn)
    load = _partner(fn, "load_model")
    reference = ctx.reference("cmd_predict")
    if load is None or is_stub(load):
        run.test(False, "cmd_predict can load a model (with the provided load_model)", "load_model(...)",
                 "load_model, as provided at the top of digits.py", "load_model is missing or changed to `pass`",
                 "digits.py has lost the provided load_model: copy it back from labkit/reference/lab4.py, save, "
                 "then run this check again.")
        return
    with ctx.temp_dir(chdir=True):
        _write_json(Path("models") / "m.json", _fixture_model(ctx))
        Path("first.txt").write_text(_grid(TEST.images[0]), encoding="utf-8")
        Path("second.csv").write_text(",".join(map(str, TEST.images[1])) + "\n", encoding="utf-8")
        Path("photo.png").write_bytes(b"\x89PNG\r\n\x1a\n not really a picture")
        cases = [
            ("A .txt image (28 lines of 28 numbers)", _ns(model="models/m.json", image="first.txt", show=False),
             "predict models/m.json first.txt"),
            ("A .csv image (one line of 784 numbers)", _ns(model="models/m.json", image="second.csv", show=False),
             "predict models/m.json second.csv"),
            ("--show: the picture first, then the prediction line",
             _ns(model="models/m.json", image="first.txt", show=True), "predict models/m.json first.txt --show"),
        ]
        for title, args, words in cases:
            expected = reference(args)
            shown = f"cmd_predict(args) for: {words}"
            a = ctx.call(fn, args, show=shown, track=False)
            a.args, a.expected = (args,), expected
            ok = a.error is None and a.got == expected and not a.printed.strip()
            got = _outcome(a) if a.error else (repr(a.got) if not isinstance(a.got, str) or "\n" not in a.got
                                               else _preview(a.got, 30))
            cause = _first_cause(a, C4_RULES)
            if cause is None and args.show and isinstance(a.got, str):
                if a.got.replace("\n", "") == expected.replace("\n", ""):
                    cause = ("put a newline between the picture and the prediction line: picture(image) + \"\\n\" + "
                             "f\"Prediction: {label}\".")
                elif a.got.strip() == expected.split("\n")[-1]:
                    cause = "args.show is True but there is no picture: return picture(image) + \"\\n\" + the prediction line."
                elif a.got.startswith("Prediction"):
                    cause = "the picture comes first, then the prediction line: picture(image) + \"\\n\" + f\"Prediction: {label}\"."
            if cause is None and isinstance(a.got, str) and a.got.strip() != expected and "\n" not in expected:
                cause = f"the text must be exactly {expected!r}: \"Prediction: \" and the label."
            if not run.test(ok, title, shown, repr(expected) if "\n" not in expected else _preview(expected, 30),
                            got, cause, attempt=a):
                return
        args = _ns(model="models/m.json", image="photo.png", show=False)
        b = ctx.call(fn, args, show="cmd_predict(args) for: predict models/m.json photo.png", track=False)
        wanted = "unsupported image type: .png (use .csv or .txt)"
        if not run.test(b.raised(ValueError) and str(b.error) == wanted,
                        "A .png image: ValueError('unsupported image type: .png (use .csv or .txt)')",
                        "cmd_predict(args) for: predict models/m.json photo.png", f"raises ValueError({wanted!r})",
                        _outcome(b),
                        ("the message must be exactly f\"unsupported image type: {suffix} (use .csv or .txt)\": it "
                         "tells the user what to do." if b.raised(ValueError) else
                         "for any other suffix, raise ValueError(f\"unsupported image type: {suffix} (use .csv or "
                         ".txt)\") (an else after your if/elif)."),
                        attempt=b if b.error is not None and not b.raised(ValueError) else None):
            return
        args = _ns(model="models/nope.json", image="first.txt", show=False)
        c = ctx.call(fn, args, show="cmd_predict(args) for: predict models/nope.json first.txt", track=False)
        run.test(c.raised(FileNotFoundError), "A missing model file: FileNotFoundError (main() reports it)",
                 "cmd_predict(args) for: predict models/nope.json first.txt", "raises FileNotFoundError",
                 _outcome(c), "let the FileNotFoundError from load_model through: main() catches it.",
                 attempt=c if c.error is not None else None)


C4 = Task(
    "C4", "cmd_predict", title="Read one parcel", home="digits",
    practise="choosing what to do from a file's suffix, with pathlib (Theory S7)",
    minimum_path=True, inputs_unchanged=False, custom=_c4, detectors=C4_RULES,
)


# ============================================================================= C5 · main
def _code_of(attempt):
    if attempt.raised(SystemExit):
        code = attempt.error.code
        return 0 if code is None else code
    return attempt.got


def _summary(attempt) -> str:
    if attempt.error is not None and not attempt.raised(SystemExit):
        return f"raised {attempt.error_text}"[:120]
    code = _code_of(attempt)
    head = f"exit code {code!r}" + (" (from SystemExit)" if attempt.raised(SystemExit) else "")
    return f"{head} · stdout {_preview(attempt.printed)} · stderr {_preview(attempt.stderr)}"


C5_RULES = [
    Detector(lambda t: t.raised(SystemExit) and t.error.code == 2 and t.expected != 2,
             "argparse stopped main() with a usage error (exit code 2): your parser doesn't accept this command "
             "line yet. That's C1: run check(\"C1\", digits.build_parser)."),
    Detector(lambda t: t.raised(SystemExit) and t.expected != 2,
             "main() stopped with SystemExit: is there a sys.exit(...) or exit() in main or in a command? main must "
             "RETURN the code (return 1); only the last line of digits.py, raise SystemExit(main()), turns it into "
             "the exit code. Then a test can call main([...]) and read the code."),
    Detector(lambda t: t.raised(KeyError) and str(t.error).strip("'\"") in ("inspect", "train", "predict"),
             "your dispatch dict has no key for this subcommand: it needs all three, {\"inspect\": cmd_inspect, "
             "\"train\": cmd_train, \"predict\": cmd_predict}."),
    Detector(lambda t: t.raised(NameError) and not t.raised(UnboundLocalError),
             "Python doesn't know the name on your line (shown above): check its spelling in digits.py, or give "
             "it a value before you use it. Save, then run this check again."),
    Detector(lambda t: t.raised(FileNotFoundError, FileExistsError, ValueError),
             "the error got out of main(), so a terminal would show a traceback. Put the command call in try/except "
             "(FileNotFoundError, FileExistsError, ValueError) as err: print(f\"error: {err}\", file=sys.stderr), "
             "then return 1."),
    Detector(lambda t: t.got is None and t.error is None and not t.printed.strip(),
             "main() returned None and printed nothing: call the command (commands[args.command](args)), print the "
             "string it returns, then return 0."),
    Detector(lambda t: t.got is None and t.error is None,
             "main() returned None: after printing, return 0 (and return 1 after an error). The exit code is how a "
             "script running digits.py at 3 a.m. knows whether it worked."),
    Detector(lambda t: "Traceback" in t.stderr,
             "users never see a traceback: print one line, print(f\"error: {err}\", file=sys.stderr)."),
    Detector(lambda t: t.expected == 1 and "error" in t.printed.lower() and not t.stderr.strip(),
             "the error message went to stdout. Errors go to stderr: print(f\"error: {err}\", file=sys.stderr). "
             "stdout is for results only: a script that saves stdout to a file must never find an error in it."),
    Detector(lambda t: t.expected == 1 and t.got == 0,
             "main() returned 0 after an error: return 1 in the except block."),
    Detector(lambda t: t.expected == 0 and t.got == 0 and not t.printed.strip(),
             "nothing on stdout: print the string the command returns, then return 0."),
]


def _needs_note(fn, ctx, name: str, task_id: str) -> str | None:
    """'main uses cmd_train, which isn't passing yet (C3) ...' when that command is a stub or not passing."""
    command = getattr(fn, "__globals__", {}).get(name)
    task = ctx.lab_cases.find(task_id) if ctx.lab_cases is not None else None
    if command is None or task is None:
        return None
    if not is_stub(command):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if run_task(task, command, ctx.lab_cases)[0].status == PASS:
                    return None
        except Exception:                                    # noqa: BLE001
            return None
    return (f"main uses {name}, which isn't passing yet ({task_id}). Fix {task_id} first (check(\"{task_id}\", "
            f"digits.{name})), or borrow it: digits.{name} = rescue(\"{task_id}\").")


def _c5(fn, ctx) -> None:
    run = _Run(ctx, 8, fn)
    needs = {"train": ("cmd_train", "C3"), "predict": ("cmd_predict", "C4"), "inspect": None}
    with ctx.temp_dir(chdir=True):
        _write_json(Path("models") / "m.json", _fixture_model(ctx))
        Path("p.txt").write_text(_grid(TEST.images[0]), encoding="utf-8")
        Path("notes.png").write_bytes(b"not a picture")
        Path("broken.json").write_text("this is not JSON", encoding="utf-8")
        tests = [
            ("inspect: exit code 0, the result on stdout, nothing on stderr",
             ["inspect", TEST_FILE, "--index", "0"], 0, lambda out, err: "Image 0 (label" in out and not err.strip(),
             "exit code 0 · stdout: the picture of image 0 · stderr: nothing"),
            ("train: exit code 0, its message on stdout",
             ["train", TRAIN_FILE, "--out", "models/new.json", "--limit", "100"], 0,
             lambda out, err: out.startswith("Trained on 100 images") and not err.strip(),
             "exit code 0 · stdout: Trained on 100 images … · stderr: nothing"),
            ("predict: exit code 0, the prediction on stdout", ["predict", "models/m.json", "p.txt"], 0,
             lambda out, err: out.startswith("Prediction: ") and not err.strip(),
             "exit code 0 · stdout: Prediction: … · stderr: nothing"),
            ("A missing model file: exit code 1, ONE line on stderr that names it, nothing on stdout",
             ["predict", "models/nope.json", "p.txt"], 1,
             lambda out, err: not out.strip() and err.startswith("error:") and "nope.json" in err
             and err.strip().count("\n") == 0, "exit code 1 · stdout: nothing · stderr: error: … nope.json …"),
            ("train onto an existing model without --force: exit code 1 and the --force hint",
             ["train", TRAIN_FILE, "--out", "models/m.json", "--limit", "100"], 1,
             lambda out, err: not out.strip() and err.startswith("error:") and "--force" in err,
             "exit code 1 · stdout: nothing · stderr: error: … use --force to overwrite"),
            ("An image type digits.py can't read: exit code 1, one error line",
             ["predict", "models/m.json", "notes.png"], 1,
             lambda out, err: not out.strip() and err.startswith("error:") and err.strip().count("\n") == 0,
             "exit code 1 · stdout: nothing · stderr: error: unsupported image type: .png …"),
            ("A file that isn't a model: exit code 1, one error line", ["predict", "broken.json", "p.txt"], 1,
             lambda out, err: not out.strip() and err.startswith("error:") and err.strip().count("\n") == 0,
             "exit code 1 · stdout: nothing · stderr: error: not a valid model file: …"),
            ("A wrong command line is argparse's job: exit code 2", ["train"], 2,
             lambda out, err: True, "exit code 2 (argparse prints the usage on stderr)"),
        ]
        for title, words, code, output_ok, expected in tests:
            shown = "main(" + repr(words).replace("'", '"') + ")"
            a = ctx.call(fn, words, show=shown, track=False)
            a.expected = code
            got_code = _code_of(a)
            ok = (a.error is None or a.raised(SystemExit)) and got_code == code and output_ok(a.printed, a.stderr)
            if code != 2 and a.raised(SystemExit):
                ok = False
            cause = None
            if not ok and needs.get(words[0]) and not (a.raised(SystemExit) and a.error.code == 2):
                cause = _needs_note(fn, ctx, *needs[words[0]])
            if not run.test(ok, title, shown, expected, _summary(a),
                            cause or _first_cause(a, C5_RULES) or (
                                None if a.error is not None and not a.raised(SystemExit) else
                                "stdout is for the result, stderr for one error line: compare yours with the "
                                "Expected line."),
                            attempt=a if a.error is not None and not a.raised(SystemExit) else None):
                return


C5 = Task(
    "C5", "main", title="One entry point, honest exit codes", home="digits",
    stub_fingerprint="f3930531159b10e2",     # the stub as shipped (= lab4_tools.STUB_FINGERPRINTS; the build checks)
    practise="the main pattern: a dispatch dict, stdout vs stderr, exit codes 0/1/2 (Theory S7)",
    minimum_path=True, depends_on=("C1",), inputs_unchanged=False, custom=_c5, detectors=C5_RULES,
)


# ============================================================================= CP3 · run it in a real terminal
def _terminal(words: list[str]) -> subprocess.CompletedProcess | str:
    script = KIT_DIR / "digits.py"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHON_COLORS="0")   # plain --help on 3.14
    try:
        proc = subprocess.run([sys.executable, str(script)] + words, cwd=str(KIT_DIR), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", env=env, timeout=60)
        proc.stdout, proc.stderr = plain(proc.stdout), plain(proc.stderr)
        return proc
    except subprocess.TimeoutExpired:
        return "it ran for more than 60 s"


_CORE_TASKS = {"build_parser": "C1", "save_model": "C2", "cmd_train": "C3", "cmd_predict": "C4", "main": "C5"}


def _rescued_now() -> dict[str, str]:
    """{function name: task id} for the CORE functions of digits.py that are rescued (reference) right now."""
    module = sys.modules.get("digits")
    found = {}
    for name, task_id in _CORE_TASKS.items():
        value = getattr(module, name, None)
        if _is_reference(value):
            found[name] = task_id
    return found


def _rescue_cause(rescued: dict[str, str], names: tuple[str, ...]) -> str | None:
    """The cause for a terminal test that failed while some of the functions it needs are rescued."""
    involved = [(name, rescued[name]) for name in names if name in rescued]
    if not involved:
        return None
    which = ", ".join(f"{name} ({task_id})" for name, task_id in involved)
    return (f"rescue() works only inside the notebook, but the terminal runs digits.py as saved, where {which} "
            "is still yours. Finish it in digits.py (or copy it from labkit/reference/lab4.py), save, then run "
            "this check again.")


def _ran(proc) -> str:
    if isinstance(proc, str):
        return proc
    return f"exit code {proc.returncode} · stdout {_preview(proc.stdout)} · stderr {_preview(proc.stderr, 50)}"


def _cp3(_obj, ctx) -> None:
    run = _Run(ctx, 6)
    script = KIT_DIR / "digits.py"
    rescued = _rescued_now()
    ctx.note(f"Ran with {hide_user_folder(sys.executable)} in {hide_user_folder(str(KIT_DIR))}")
    if rescued:
        ctx.note("Rescued in this notebook: " + ", ".join(f"{name} ({task_id})" for name, task_id in rescued.items())
                 + ". A terminal doesn't see rescues: it runs digits.py as saved.")
    if not run.test(script.is_file(), "digits.py is in the kit folder", "the kit folder", "digits.py",
                    "digits.py" if script.is_file() else "no digits.py",
                    "re-extract the kit (or copy digits.py back from the zip)."):
        return
    first = _terminal(["--help"])
    if isinstance(first, str) or first.returncode != 0 or "usage: digits" not in first.stdout:
        quiet = not isinstance(first, str) and first.returncode == 0 and not first.stdout.strip()
        run.test(False, "python digits.py --help: exit code 0 and the usage", "python digits.py --help",
                 "exit code 0 · stdout: usage: digits ...", _ran(first),
                 "digits.py printed nothing: its last two lines must call main(): if __name__ == \"__main__\": "
                 "raise SystemExit(main())." if quiet else
                 "read the stderr line: Python stops at the first error in digits.py. Fix that line and save.")
        return
    run.test(True, "python digits.py --help: exit code 0 and the usage", "python digits.py --help",
             "exit code 0 · stdout: usage: digits ...", "exit code 0 · stdout: usage: digits ...")
    braces = re.search(r"\{([^}]*)\}", first.stdout)            # usage: digits [-h] {inspect,train,...} ...
    choices = braces.group(1).split(",") if braces else []
    listed = [name for name in ("inspect", "train", "predict") if name in choices]
    if not run.test(len(listed) == 3, "--help lists inspect, train and predict", "python digits.py --help",
                    "inspect, train, predict", ", ".join(listed) or "none",
                    _rescue_cause(rescued, ("build_parser",)) or
                    "--help lists the subcommands of build_parser: add the missing ones (C1)."):
        return
    second = _terminal(["inspect", "--index", "0"])
    ok = not isinstance(second, str) and second.returncode == 0 and "Image 0 (label" in second.stdout
    if not run.test(ok, "python digits.py inspect --index 0: exit code 0 and a picture",
                    "python digits.py inspect --index 0", "exit code 0 · stdout: the picture of image 0", _ran(second),
                    _rescue_cause(rescued, ("main",)) or
                    "main() must print the command's result and return 0 (C5): run check(\"C5\", digits.main)."):
        return
    model = "output/cp3_check_model.json"
    words = ["train", TRAIN_FILE, "--out", model, "--limit", "50", "--force"]
    third = _terminal(words)
    ok = not isinstance(third, str) and third.returncode == 0 and third.stdout.startswith("Trained on 50 images")
    if not run.test(ok, "python digits.py train ... --limit 50 --force: exit code 0 and its message",
                    "python digits.py " + " ".join(words), "exit code 0 · stdout: Trained on 50 images …",
                    _ran(third),
                    _rescue_cause(rescued, ("build_parser", "main", "cmd_train", "save_model")) or
                    "the terminal can't train yet: run check(\"C3\", digits.cmd_train) and check(\"C5\", "
                    "digits.main), and read the stderr line above."):
        return
    words = ["predict", model, "samples/parcel_021.txt"]
    fourth = _terminal(words)
    ok = not isinstance(fourth, str) and fourth.returncode == 0 and fourth.stdout.startswith("Prediction: ")
    run.test(ok, "python digits.py predict ...: exit code 0 and the prediction", "python digits.py " + " ".join(words),
             "exit code 0 · stdout: Prediction: …", _ran(fourth),
             _rescue_cause(rescued, ("build_parser", "main", "cmd_predict")) or
             "the terminal can't predict yet: run check(\"C4\", digits.cmd_predict) and check(\"C5\", digits.main), "
             "and read the stderr line above.")


CP3 = Task(
    "CP3", "terminal", kind="custom", title="Run it in a real terminal", rescue=False, quality=False,
    practise="running a script from a terminal: the same code, a new face (Theory S7)",
    custom=_cp3,
)


# ============================================================================= ENG1 · check_model (OPTIONAL)
ENG1_RULES = [
    Detector(lambda t: t.raised(KeyError),
             "you read a key that isn't there: check `key in model` first and raise ValueError. main() only "
             "catches ValueError, so a KeyError would show the night shift a traceback."),
    Detector(lambda t: t.raised(TypeError, AttributeError),
             "check the type before you use a value: isinstance(model, dict), isinstance(values, list), and that "
             "every value is an int or a float (but not a bool)."),
    Detector(lambda t: t.error is None and t.expected is not None,
             "this model is broken (see the test's title), but check_model accepted it: add a check that raises "
             "ValueError for it."),
    Detector(lambda t: t.error is not None and t.expected is None,
             "check_model rejected a complete, valid model: compare your conditions with the docstring."),
]


def _model_case(model: dict, what: str, title: str, key: str, expected=None, highlight: bool = False) -> Case:
    return Case((model,), Raises(ValueError) if expected is None and key != "valid" else expected, key=key,
                title=title, show_inputs={"Model": what}, highlight=highlight,
                expected_label="Expected")


def _eng1(fn, ctx) -> None:
    load = _partner(fn, "load_model")
    run = _Run(ctx, 1, fn)
    if load is None or is_stub(load):
        run.test(False, "load_model calls check_model", "load_model(...)", "the provided load_model",
                 "load_model is missing or changed to `pass`",
                 "digits.py has lost the provided load_model: copy it back from labkit/reference/lab4.py and save.")
        return
    with ctx.temp_dir(chdir=True):
        model = _valid_model()
        model["version"] = 2
        _write_json(Path("v2.json"), model)
        a = ctx.call(load, Path("v2.json"), show="load_model(Path(\"v2.json\"))   (a version-2 model)", track=False)
        run.test(a.raised(ValueError), "load_model calls your check_model: a version-2 model file is refused",
                 "load_model(Path(\"v2.json\"))   (a version-2 model)", "raises ValueError", _outcome(a),
                 "the provided load_model calls check_model(model) after the format check: is that line still "
                 "there? (Copy load_model back from labkit/reference/lab4.py if it changed.)")


ENG1 = Task(
    "ENG1", "check_model", title="Bullet-proof and tested", home="digits",
    practise="validating data before you trust it, and raising ValueError with a clear message",
    input_labels=("Model",), inputs_unchanged=True,
    cases=[
        _model_case(_valid_model(), "a complete version-1 model (10 centroids of 784 numbers)",
                    "A complete, valid model: no error (returns None)", "valid", expected=None, highlight=True),
        _model_case(_broken(lambda m: m.update(version=2)), 'the same model with "version": 2',
                    "A model of another version: ValueError", "version", highlight=True),
        _model_case(_broken(lambda m: m.pop("centroids")), 'the same model without "centroids"',
                    "A missing key: ValueError (not KeyError)", "no-centroids", highlight=True),
        _model_case(_broken(lambda m: m["centroids"].pop("9")), 'the same model without the centroid "9"',
                    "Nine digits instead of ten: ValueError", "nine"),
        _model_case(_broken(lambda m: m["centroids"].update({"3": [0.0] * 783})),
                    'centroid "3" has 783 numbers', "A centroid of the wrong length: ValueError", "length"),
        _model_case(_broken(lambda m: m["centroids"]["5"].__setitem__(10, "12.5")),
                    'centroid "5" holds the text "12.5"', "A value that isn't a number: ValueError", "text-value"),
        _model_case(["format", "garabato-centroids"], 'a JSON list: ["format", "garabato-centroids"]',
                    "Not a dict at all: ValueError", "list"),
    ],
    detectors=ENG1_RULES, custom=_eng1,
)


# ============================================================================= SCI1 · cmd_evaluate (OPTIONAL)
def _sci1(fn, ctx) -> None:
    run = _Run(ctx, 4, fn)
    reference = ctx.reference("cmd_evaluate")
    with ctx.temp_dir(chdir=True):
        _write_json(Path("models") / "m.json", _fixture_model(ctx))
        args = _ns(data=TEST_FILE, model="models/m.json", limit=200, report=None)
        shown = "cmd_evaluate(args) for: evaluate data/mnist_test_2k.csv.gz --model models/m.json --limit 200"
        expected = reference(args).split("\n")
        a = ctx.call(fn, args, show=shown, track=False)
        lines = a.got.split("\n") if isinstance(a.got, str) else []
        if not run.test(a.error is None and isinstance(a.got, str) and len(lines) == 3, "It returns three lines (no "
                        "--report)", shown, "3 lines: Evaluated … / Accuracy: … / Per digit: …",
                        _outcome(a) if a.error else (f"{len(lines)} lines" if lines else repr(a.got)),
                        "return \"\\n\".join(lines) with the three lines of the docstring (print nothing).",
                        attempt=a):
            return
        if not run.test(lines[0].startswith("Evaluated models/m.json on 200 images in "), "Line 1: what, on how "
                        "many images, how long", shown, "Evaluated models/m.json on 200 images in … s",
                        repr(lines[0][:70]), "pass limit=args.limit to load_digits; f\"Evaluated {args.model} on "
                        "{len(labels):,} images in {seconds:.1f} s\"."):
            return
        if not run.test(lines[1:] == expected[1:], "Lines 2 and 3: the accuracy, then one share per digit", shown,
                        " / ".join(expected[1:])[:120], " / ".join(lines[1:])[:120],
                        "count per digit with two dicts (images seen, images right); a share is right / seen; "
                        "format with :.1% and :.0% (a fraction, not a percentage number)."):
            return
        with_report = _ns(data=TEST_FILE, model="models/m.json", limit=200, report="out/report.json")
        reference(_ns(data=TEST_FILE, model="models/m.json", limit=200, report="ref/report.json"))
        b = ctx.call(fn, with_report, show=shown + " --report out/report.json", track=False)
        try:
            mine = json.loads(Path("out/report.json").read_text(encoding="utf-8"))
            theirs = json.loads(Path("ref/report.json").read_text(encoding="utf-8"))
            same = {k: v for k, v in mine.items() if k != "seconds"} == {k: v for k, v in theirs.items() if k != "seconds"}
        except (OSError, ValueError, AttributeError):
            mine, same = None, False
        run.test(b.error is None and same, "--report out/report.json: the same numbers as a JSON file",
                 shown + " --report out/report.json", "out/report.json with model, data, n, correct, accuracy, "
                 "per_digit, seconds", _outcome(b) if b.error else ("no readable out/report.json" if mine is None
                                                                    else f"keys {sorted(mine)}"),
                 "create the folder (path.parent.mkdir(parents=True, exist_ok=True)) and json.dump the dict of the "
                 "docstring, with indent=2.", attempt=b)


SCI1 = Task(
    "SCI1", "cmd_evaluate", title="Measure it from the command line", home="digits",
    practise="a new subcommand end to end: counting with dicts, f-string formats, an optional JSON report",
    inputs_unchanged=False, custom=_sci1,
)


# ============================================================================= BLD1 · cmd_batch (OPTIONAL)
def _bld1(fn, ctx) -> None:
    run = _Run(ctx, 5, fn)
    with ctx.temp_dir(chdir=True):
        _write_json(Path("models") / "m.json", _fixture_model(ctx))
        inbox = Path("inbox")
        inbox.mkdir()
        (inbox / "a_0001.txt").write_text(_grid(TEST.images[2]), encoding="utf-8")
        (inbox / "b_0002.csv").write_text(",".join(map(str, TEST.images[3])) + "\n", encoding="utf-8")
        (inbox / "c_0003.txt").write_text(_grid(TEST.images[4]).replace(" 0\n", "\n", 1), encoding="utf-8")
        (inbox / "notes.md").write_text("not an image", encoding="utf-8")
        args = _ns(inbox="inbox", model="models/m.json", out="predictions.csv")
        shown = "cmd_batch(args) for: batch inbox --model models/m.json --out predictions.csv"
        a = ctx.call(fn, args, show=shown, track=False)
        ok = a.error is None and isinstance(a.got, str) and "\n" not in a.got.strip() and not a.printed.strip()
        if not run.test(ok, "It reads the inbox and returns ONE summary line", shown + "   (2 images, 1 broken file, "
                        "notes.md)", "one line, e.g. Read 3 files: 2 predicted, 1 unreadable · ...",
                        _outcome(a) if a.error else repr(str(a.got)[:80]),
                        ("one unreadable file must not stop the overnight run: try/except ValueError around each "
                         "read, and write \"error: ...\" in its row." if a.raised(ValueError) else
                         "return one line (no \\n), and print nothing."), attempt=a):
            return
        try:
            with open("predictions.csv", encoding="utf-8", newline="") as f:
                rows = [row for row in csv.reader(f) if row]
        except OSError:
            rows = []
        first = rows[:1] == [["file", "prediction"]] and [r[0] for r in rows[1:]] == ["a_0001.txt", "b_0002.csv",
                                                                                       "c_0003.txt"]
        if not run.test(first and rows[3][-1].startswith("error"), "predictions.csv: a header, one row "
                        "per image (sorted), error: … for the broken one", "open(\"predictions.csv\")",
                        "file,prediction / a_0001.txt,… / b_0002.csv,… / c_0003.txt,error: …",
                        " / ".join(",".join(r)[:30] for r in rows[:4])[:120] or "no predictions.csv",
                        "csv.DictWriter with fieldnames [\"file\", \"prediction\"]; only .csv and .txt files, sorted by "
                        "name (notes.md is not an image)."):
            return
        left = sorted(p.name for p in inbox.glob("*") if p.is_file())
        archives = [p for p in (inbox / "archive").glob("*") if p.is_dir()] if (inbox / "archive").is_dir() else []
        moved = len(archives) == 1 and re.fullmatch(r"\d{8}_\d{6}", archives[0].name) is not None and sorted(
            p.name for p in archives[0].iterdir()) == ["a_0001.txt", "b_0002.csv", "c_0003.txt"]
        if not run.test(left == ["notes.md"] and moved, "The images moved to inbox/archive/YYYYmmdd_HHMMSS/",
                        "the inbox after the run", "inbox/: notes.md only · archive/<timestamp>/: the 3 images",
                        f"inbox/: {', '.join(left) or 'empty'} · archive folders: "
                        f"{', '.join(p.name for p in archives) or 'none'}",
                        "stamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\"); make inbox/archive/stamp/ and "
                        "shutil.move every file you read into it."):
            return
        (inbox / "d_0004.csv").write_text(",".join(map(str, TEST.images[5])) + "\n", encoding="utf-8")
        b = ctx.call(fn, args, show=shown + "   (the next night: 1 new image)", track=False)
        backups = sorted(p.name for p in Path(".").glob("predictions_*.csv"))
        if not run.test(b.error is None and len(backups) == 1 and re.fullmatch(r"predictions_\d{8}_\d{6}\.csv",
                        backups[0]) is not None, "The next night: the old predictions.csv is kept as a timestamped "
                        "backup", shown + "   (the next night)", "predictions_YYYYmmdd_HHMMSS.csv next to it",
                        _outcome(b) if b.error else (", ".join(backups) or "no backup"),
                        "before writing, if the CSV exists: shutil.copy2(out, out.with_name(f\"{out.stem}_{stamp}"
                        "{out.suffix}\")).", attempt=b):
            return
        try:
            with open("predictions.csv", encoding="utf-8", newline="") as f:
                now = [row[0] for row in csv.reader(f) if row]
        except OSError:
            now = []
        run.test(now == ["file", "d_0004.csv"], "…and predictions.csv now holds only the new night",
                 "open(\"predictions.csv\")", "file,prediction / d_0004.csv,…", " / ".join(now) or "nothing",
                 "write the CSV again from scratch (mode \"w\"), with only the files of this run.")


BLD1 = Task(
    "BLD1", "cmd_batch", title="The overnight batch", home="digits",
    practise="files and folders with pathlib, csv, shutil and timestamps: the backup pattern (Theory S7)",
    inputs_unchanged=False, custom=_bld1,
)


# ============================================================================= ADVANCED (sanity checks, no rescue)
def _log_probe(verbose: bool):
    def call(fn):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            logger = fn(verbose)
            logger.info("probe info")
            logger.warning("probe warning")
        text = err.getvalue()
        return {"INFO": "probe info" in text, "WARNING": "probe warning" in text,
                "stdout lines": len(out.getvalue().splitlines())}
    return call


ENG2 = Task(
    "ENG2", "setup_logging", title="Hygiene: logging", rescue=False,
    practise="logging to stderr, with a level you choose from the command line",
    cases=[
        Case(call=_log_probe(True), expected={"INFO": True, "WARNING": True, "stdout lines": 0},
             key="verbose", title="--verbose: the INFO and the WARNING line on stderr, nothing on stdout", highlight=True,
             show_inputs={"Call": "logger = setup_logging(True); logger.info(...); logger.warning(...)"}),
        Case(call=_log_probe(False), expected={"INFO": False, "WARNING": True, "stdout lines": 0},
             key="quiet", title="Without --verbose: only the WARNING line (and worse), still on stderr", highlight=True,
             show_inputs={"Call": "logger = setup_logging(False); logger.info(...); logger.warning(...)"}),
    ],
    detectors=[
        Detector(lambda t: t.raised(AttributeError) and "NoneType" in str(t.error),
                 "setup_logging must return the logger (logging.getLogger(\"digits\"))."),
        Detector(lambda t: isinstance(t.got, dict) and t.got.get("stdout lines"),
                 "the log lines went to stdout: logging.StreamHandler(sys.stderr). stdout is for results only."),
    ],
)

CENTROID_TOY = {"format": "garabato-centroids", "version": 1, "n_train": 2, "image_size": 2,
                "centroids": {"0": [0.0, 0.0, 0.0, 0.0], "1": [255.0, 255.0, 255.0, 255.0]}}
PERCEPTRON_TOY = {"format": "garabato-perceptron", "version": 1, "image_size": 2,
                  "weights": {"0": [1, 0, 0, 0, 0], "1": [0, 0, 0, 1, 0], "2": [0, 0, 0, 0, 100]}}


def _perceptron_learns(fn):
    """call: train_perceptron (next to predict_with) on a tiny separable set; how many it reads right after."""
    train = _partner(fn, "train_perceptron")
    if train is None or is_stub(train):
        return "train_perceptron is missing or still `pass`"
    images = [[250, 10, 0, 0], [240, 0, 20, 0], [0, 0, 10, 250], [10, 0, 0, 230], [0, 240, 250, 0], [10, 250, 230, 0]]
    labels = [0, 0, 1, 1, 2, 2]
    model = train(images, labels, epochs=5)
    return sum(1 for image, label in zip(images, labels) if fn(model, image) == label)


SCI2 = Task(
    "SCI2", "predict_with", title="A model registry", rescue=False,
    practise="dispatching on a model's format with a dict of functions, and a perceptron in pure Python",
    input_labels=("Model", "Image"), expected_label="Expected label",
    cases=[
        Case((CENTROID_TOY, [200, 220, 255, 255]), 1, key="centroid", highlight=True,
             title="A centroid model (format garabato-centroids): the closest average wins",
             show_inputs={"Model": "2 centroids of 4 pixels: \"0\" all 0, \"1\" all 255", "Image": "[200, 220, 255, 255]"}),
        Case((PERCEPTRON_TOY, [10, 0, 0, 50]), 2, key="perceptron-bias", highlight=True,
             title="A perceptron model: scores 10, 50 and 0 + bias 100, so 2 wins",
             show_inputs={"Model": "weights \"0\" [1,0,0,0 | 0], \"1\" [0,0,0,1 | 0], \"2\" [0,0,0,0 | 100]",
                          "Image": "[10, 0, 0, 50]"}),
        Case((PERCEPTRON_TOY, [255, 0, 0, 50]), 0, key="perceptron",
             title="The same perceptron: scores 255, 50 and 100, so 0 wins",
             show_inputs={"Model": "the same perceptron", "Image": "[255, 0, 0, 50]"}),
        Case(({"format": "garabato-knn", "version": 1}, [0, 0, 0, 0]), Raises(ValueError), key="unknown",
             title="A format the registry doesn't know: ValueError",
             show_inputs={"Model": "{\"format\": \"garabato-knn\", ...}", "Image": "[0, 0, 0, 0]"}),
        Case(call=_perceptron_learns, expected=6, key="learns", expected_label="Expected",
             title="train_perceptron learns a tiny separable set (6 images, 5 epochs): all 6 read right",
             show_inputs={"Call": "predict_with(train_perceptron(images, labels, epochs=5), image) for the 6 images"}),
    ],
)


def _fake_top3(pixels):
    return [[7, 1.0], [1, 0.8], [9, 0.5]]


def _web(fn):
    """The module the endpoint lives in (web_app.py): its Drawing model and its app."""
    return sys.modules.get(getattr(fn, "__module__", "") or "")


def _fields(fn):
    return sorted(_web(fn).Drawing.model_fields)


def _routes(fn):
    """'METHOD /path' of every route of app whose endpoint is predict."""
    return sorted(f"{method} {route.path}" for route in _web(fn).app.routes
                  if getattr(route, "endpoint", None) is fn for method in sorted(getattr(route, "methods", ())))


def _answer(body: bytes):
    """call: what FastAPI answers to this body: 422 if the Drawing model refuses it (before predict runs), else
    (200, predict(drawing)), with a stand-in top3 (no model file needed)."""
    def call(fn):
        from pydantic import ValidationError

        module = _web(fn)
        try:
            drawing = module.Drawing.model_validate_json(body)
        except ValidationError:
            return 422
        saved = getattr(module, "top3", None)
        module.top3 = _fake_top3
        try:
            return 200, fn(drawing)
        finally:
            module.top3 = saved
    return call


def _body(pixels) -> bytes:
    return json.dumps({"pixels": pixels}).encode()


_ACCEPTED = (lambda t: t.expected == 422 and isinstance(t.got, tuple))

BLD2 = Task(
    "BLD2", "predict", title="A web API for the canvas", rescue=False, home="web_app", quality=False,
    practise="a web API with FastAPI: a Pydantic model as the contract, a route, and the status codes 200 and 422 "
             "(Theory S16)",
    expected_label="Expected status",
    cases=[
        Case(call=_fields, expected=["pixels"], key="fields", highlight=True, expected_label="Expected fields",
             title="Drawing describes the body: one field, pixels", show_inputs={"Call": "Drawing.model_fields"}),
        Case(call=_routes, expected=["POST /predict"], key="route", highlight=True, expected_label="Expected route",
             title="predict is the route POST /predict of app", show_inputs={"Call": "app.routes"}),
        Case(call=_answer(_body([0] * 784)), expected=(200, {"top3": [[7, 1.0], [1, 0.8], [9, 0.5]]}), key="ok",
             highlight=True, expected_label="Expected", title="784 pixels: status 200 and {\"top3\": what top3 says}",
             show_inputs={"Body": "{\"pixels\": [0, 0, ...]} (784 numbers)"}),
        Case(call=_answer(_body([0] * 783)), expected=422, key="783", highlight=True,
             title="783 pixels: 422, refused before predict runs", show_inputs={"Body": "{\"pixels\": [...]} (783 numbers)"}),
        Case(call=_answer(_body([0] * 785)), expected=422, key="785", title="785 pixels: 422",
             show_inputs={"Body": "{\"pixels\": [...]} (785 numbers)"}),
        Case(call=_answer(_body([300] + [0] * 783)), expected=422, key="300", highlight=True,
             title="A pixel of 300: 422 (pixels go from 0 to 255)", show_inputs={"Body": "{\"pixels\": [300, 0, ...]}"}),
        Case(call=_answer(_body([-1] + [0] * 783)), expected=422, key="minus", title="A pixel of -1: 422",
             show_inputs={"Body": "{\"pixels\": [-1, 0, ...]}"}),
        Case(call=_answer(json.dumps({"image": [0] * 784}).encode()), expected=422, key="no-pixels",
             title="No \"pixels\" key: 422", show_inputs={"Body": "{\"image\": [...]}"}),
        Case(call=_answer(b"pixels=1,2,3"), expected=422, key="not-json", title="A body that isn't JSON: 422",
             show_inputs={"Body": "pixels=1,2,3"}),
    ],
    detectors=[
        Detector(lambda t: t.raised(AttributeError) and "Drawing" in str(t.error),
                 "web_app.py has no Drawing model: keep `class Drawing(BaseModel):` with its field, then save."),
        Detector(lambda t: t.raised(AttributeError) and "pixels" in str(t.error),
                 "predict receives a Drawing: its pixels are drawing.pixels, and Drawing needs the field "
                 "pixels: list[Pixel] = Field(min_length=784, max_length=784)."),
        Detector(lambda t: t.expected == ["pixels"],
                 "Drawing needs exactly one field, written like a type hint inside the class: pixels: list[Pixel] "
                 "= Field(min_length=784, max_length=784). Replace `pass` with it and save web_app.py."),
        Detector(lambda t: t.expected == ["POST /predict"],
                 "put the decorator @app.post(\"/predict\") on the line just above def predict: without it FastAPI "
                 "doesn't know the route, and the page gets 404 Not Found."),
        Detector(lambda t: _ACCEPTED(t) and ("783" in t.title or "785" in t.title),
                 "Drawing accepted the wrong number of pixels: Field(min_length=784, max_length=784) on the pixels "
                 "field fixes the length."),
        Detector(lambda t: _ACCEPTED(t) and ("300" in t.title or "-1" in t.title),
                 "Drawing accepted a pixel outside 0 to 255: the field must be list[Pixel] (Pixel is provided: an "
                 "int with Field(ge=0, le=255)), not list[int]."),
        Detector(lambda t: t.expected != 422 and isinstance(t.got, tuple) and not isinstance(t.got[1], dict),
                 "return a dict, {\"top3\": top3(drawing.pixels)}: FastAPI turns it into the JSON the page reads."),
    ],
)


# ============================================================================= bridge 4 -> 5 (BR1-BR2)
# bridge_4to5.ipynb, the 15-minute catch-up notebook between Lab 4 and Lab 5 (a gentle preview of Theory S8 and
# of Lab 5): Lab 1's fit_centroids and predict_centroid become one class (BR1), and score() reads ANY model
# through its predict_one method (BR2). Both are defined in the notebook. Expected values are literal, except the
# real-data case (cases/lab4_data.json); generators/gen_cases.py checks them against reference/lab4.py.
BR_TOY_X = [[0, 2], [4, 6], [10, 10], [12, 14]]                # four 2-pixel images ...
BR_TOY_Y = [1, 1, 7, 7]                                        # ... two 1s and two 7s
BR_TOY_TEXT = "[[0, 2], [4, 6], [10, 10], [12, 14]], [1, 1, 7, 7]"
BR_QUERIES = [[1, 3], [9, 9], [3, 5]]
TRAIN14 = Digits("mnist_train_10k_14x14.csv.gz", limit=1000)  # = train_images, train_labels in the bridge
TEST14 = Digits("mnist_test_2k_14x14.csv.gz", limit=10)


def _class_case(call, expected=None, *, title: str, show: str, **kwargs) -> Case:
    """A class case: `call(cls)` gives the result; `show` is the code the test runs (printed as Call:)."""
    from ..spec import REFERENCE
    return Case(call=call, expected=REFERENCE if expected is None else expected, title=title, show=show, **kwargs)


def _chained(model) -> bool:
    return model.fit(BR_TOY_X, BR_TOY_Y) is model


def _predict_all(model, images) -> list:
    return [model.predict_one(image) for image in images]


_TAKES = re.compile(r"(?:(\w+)\.)?(\w+)\(\) takes \d+ positional arguments? but \d+ (?:was|were) given")


def _method_without_self(t) -> bool:
    """A TypeError that names a method of the student's class whose def line has no `self` first."""
    if not t.raised(TypeError):
        return False
    found = _TAKES.search(str(t.error))
    cls = t.args[0] if t.args and isinstance(t.args[0], type) else None
    if found is None or cls is None:
        return False
    owner, method = found.group(1), found.group(2)
    for klass in cls.__mro__:
        if klass is object or (owner and klass.__name__ != owner):
            continue
        raw = vars(klass).get(method)
        if raw is None or isinstance(raw, (staticmethod, classmethod)) or not callable(raw):
            continue
        try:
            params = list(inspect.signature(raw).parameters)
        except (TypeError, ValueError):
            return False
        return not params or params[0] != "self"
    return False


def _swapped_predict_centroid(t) -> bool:
    """The error happened on a student line that calls predict_centroid(self.centroids, ...)."""
    if not t.raised(KeyError, TypeError, IndexError):
        return False
    place = your_line(t.error)
    return place is not None and "predict_centroid(self.centroids" in place[1].replace(" ", "")


BR1 = Task(
    "BR1", "CentroidModel", kind="class", title="Two functions become one class",
    practise="a class with __init__, state on self, and methods (a preview of Theory S8)",
    cases=[
        _class_case(lambda cls: cls().centroids, {}, title="Step 1 · A new model has no centroids yet: an empty dict",
                    show="CentroidModel().centroids"),
        _class_case(lambda cls: cls().fit(BR_TOY_X, BR_TOY_Y).centroids, {1: [2.0, 4.0], 7: [11.0, 12.0]},
                    compare=approx(1e-9), highlight=True,
                    title="Step 2 · fit() learns one average image per label and keeps it in self.centroids",
                    show=f"CentroidModel().fit({BR_TOY_TEXT}).centroids"),
        _class_case(lambda cls: _chained(cls()), True,
                    title="Step 2 · fit() returns the model itself, so calls can be chained",
                    show=f"model = CentroidModel(); model.fit({BR_TOY_TEXT}) is model"),
        _class_case(lambda cls: _predict_all(cls().fit(BR_TOY_X, BR_TOY_Y), BR_QUERIES), [1, 7, 1], highlight=True,
                    title="Step 3 · predict_one() answers the label of the closest average",
                    show=f"model = CentroidModel().fit({BR_TOY_TEXT}); "
                         "[model.predict_one(image) for image in [[1, 3], [9, 9], [3, 5]]]"),
        _class_case(lambda cls: (cls().fit(BR_TOY_X, BR_TOY_Y), cls().centroids)[1], {},
                    title="Two models never share their centroids: a second, new model still has none",
                    show=f"CentroidModel().fit({BR_TOY_TEXT}); CentroidModel().centroids"),
        _class_case(lambda cls: _predict_all(cls().fit(TRAIN14.images, TRAIN14.labels), TEST14.images),
                    key="real-10", title="Real data: learn from the 1,000 training images, then read 10 test images",
                    show="model = CentroidModel().fit(train_images, train_labels); "
                         "[model.predict_one(image) for image in test_images[:10]]"),
    ],
    detectors=[
        Detector(lambda t: t.raised(AttributeError) and "has no attribute 'centroids'" in str(t.error)
                 and "'NoneType'" not in str(t.error),
                 "the object has no centroids attribute. Create it in __init__: self.centroids = {} (the word "
                 "self. makes it belong to the object).", "open Hint 2 below."),
        Detector(lambda t: ("share" in (t.title or "") or "no centroids yet" in (t.title or ""))
                 and isinstance(t.got, dict) and t.got,
                 "a brand-new model already has centroids: they belong to the CLASS, so every model shares one "
                 "dict. Create the dict inside __init__ (self.centroids = {}), so that each object gets its own.",
                 "open Hint 2 below."),
        Detector(lambda t: "no centroids yet" in (t.title or "") and t.got is None,
                 "a new model's centroids are None: start with an empty dict, self.centroids = {}, so that the type "
                 "never changes (a dict before fit, a dict after)."),
        Detector(lambda t: t.raised(AttributeError) and any(f"has no attribute '{name}'" in str(t.error)
                                                            for name in ("fit_centroids", "predict_centroid")),
                 "fit_centroids and predict_centroid are Lab 1's plain functions, not methods of your class: call "
                 "them without self., e.g. self.centroids = fit_centroids(images, labels)."),
        Detector(lambda t: t.raised(TypeError) and re.search(r"(fit_centroids|predict_centroid)\(\) takes 2 "
                                                             r"positional arguments but 3 were given", str(t.error)),
                 "fit_centroids and predict_centroid are plain functions: don't pass self to them. Write "
                 "fit_centroids(images, labels) and predict_centroid(image, self.centroids)."),
        Detector(lambda t: t.raised(NameError) and any(word in str(t.error) for word in
                                                       ("'fit_centroids'", "'predict_centroid'")),
                 "Lab 1's functions aren't imported yet: run the Step 0 cell (it imports fit_centroids and "
                 "predict_centroid), then the class cell, then this check."),
        Detector(lambda t: t.raised(NameError) and any(word in str(t.error) for word in ("'centroids'", "'self'")),
                 "inside a method, the object's data is reached through self: self.centroids. A bare name only "
                 "exists inside the method that created it, and self must be the first parameter of every method."),
        Detector(_method_without_self,
                 "a method is missing `self`. Python passes the object itself as the first argument of every "
                 "method, so the def line needs it: def fit(self, images, labels):, def predict_one(self, image):.",
                 "add self as the first parameter of each method, run the class cell again, then this check."),
        Detector(lambda t: (t.raised(AttributeError) and "'NoneType' object has no attribute" in str(t.error)
                            and ".fit(" in (t.show or ""))
                 or (t.got is False and "itself" in (t.title or "")),
                 "fit() returned None, so the next call in the chain (CentroidModel().fit(...).predict_one(...)) has "
                 "no model to work on. End fit() with `return self`.",
                 "add `return self` as the last line of fit, run the class cell again, then this check."),
        Detector(lambda t: "average image per label" in (t.title or "") and t.got == {},
                 "self.centroids is still empty after fit(): fit() must keep what it learns on the object, "
                 "self.centroids = fit_centroids(images, labels). A plain name (centroids = ...) disappears when "
                 "fit ends.", "open Hint 2 below."),
        Detector(_swapped_predict_centroid,
                 "predict_centroid takes (image, centroids): the image first, then the dict of averages. Write "
                 "predict_centroid(image, self.centroids)."),
        Detector(lambda t: "closest average" in (t.title or "") and t.got in ([7, 7, 7], [1, 1, 1], [-1, -1, -1]),
                 "every image gets the same answer: predict_one must compare THIS image with the centroids it learnt "
                 "(predict_centroid(image, self.centroids) does it)."),
    ],
)


class ToyModel:
    """A toy model for the BR2 tests: it answers the first pixel of each image (so [7] is read as 7)."""

    def fit(self, images, labels):
        return self

    def predict_one(self, image):
        return image[0]

    def __repr__(self) -> str:
        return "ToyModel()"


class ToyModelNoFit(ToyModel):
    """The same toy model, but fit() complains: score() must only read, never train."""

    def fit(self, images, labels):
        raise RuntimeError("score() called fit(): it must only read the model, never train it")

    def __repr__(self) -> str:
        return "ToyModelNoFit()"


def _score_case(model, images, labels, expected, *, title: str, key: str, highlight: bool = False) -> Case:
    what = ("ToyModel(): it answers the first pixel of each image, so [7] is read as 7" if type(model) is ToyModel
            else "ToyModelNoFit(): the same toy model, but its fit() raises an error")
    return Case(call=lambda fn: fn(model, [list(i) for i in images], list(labels)), expected=expected,
                compare=approx(1e-9), key=key, title=title, highlight=highlight,
                show=f"score({model!r}, {images!r}, {labels!r})",
                show_inputs={"Model": what, "Images": repr(images), "Labels": repr(labels)})


# how many images each BR2 test reads right (a count returned instead of the fraction)
_RIGHT_COUNTS = {"Three images, two read right": 2, "Every image read right": 2,
                 "score() only reads": 1}


BR2 = Task(
    "BR2", "score", title="One function scores any model",
    practise="calling a method of an object you were given: model.predict_one(image)",
    expected_label="Expected score",
    cases=[
        _score_case(ToyModel(), [[7], [2], [3]], [7, 2, 1], 2 / 3, key="two-of-three", highlight=True,
                    title="Three images, two read right: 2/3"),
        _score_case(ToyModel(), [[4], [0]], [4, 0], 1.0, key="all", title="Every image read right: 1.0"),
        _score_case(ToyModel(), [], [], 0.0, key="empty", highlight=True,
                    title="No images at all: 0.0 (there is nothing to divide by)"),
        _score_case(ToyModelNoFit(), [[5], [6]], [5, 1], 0.5, key="no-fit",
                    title="score() only reads: it never calls fit()"),
    ],
    detectors=[
        Detector(lambda t: t.raised(ZeroDivisionError),
                 "with no images there is nothing to divide by: return 0.0 first when labels is empty "
                 "(if not labels: return 0.0)."),
        Detector(lambda t: t.raised(TypeError) and "object is not callable" in str(t.error),
                 "a model is an object, not a function: ask it with its method, model.predict_one(image)."),
        Detector(lambda t: t.raised(NameError) and "'predict_one'" in str(t.error),
                 "predict_one is a method of the model: call it on the object, model.predict_one(image)."),
        Detector(lambda t: t.raised(NameError) and any(w in str(t.error) for w in ("'predict_centroid'", "'centroids'")),
                 "score() works with ANY model, not only centroids: ask the model itself, model.predict_one(image)."),
        Detector(lambda t: t.raised(AttributeError) and "has no attribute 'centroids'" in str(t.error),
                 "score() works with ANY model, so it can't reach into model.centroids (this toy model has none): "
                 "ask the model itself, model.predict_one(image). That is the common interface Lab 5 is built on."),
        Detector(lambda t: t.raised(AttributeError) and "has no attribute 'predict'" in str(t.error),
                 "the method is called predict_one, and it reads ONE image: model.predict_one(image), in a loop."),
        Detector(lambda t: t.raised(RuntimeError) and "called fit()" in str(t.error),
                 "score() only reads: the model arrives already trained, so don't call model.fit(...) here."),
        Detector(lambda t: isinstance(t.got, int) and not isinstance(t.got, bool)
                 and (t.title or "").split(":")[0] in _RIGHT_COUNTS and t.got == _RIGHT_COUNTS[t.title.split(":")[0]],
                 "you returned how many were right. A score is a fraction: divide by the number of images, "
                 "len(labels)."),
        Detector(lambda t: isinstance(t.got, (int, float)) and not isinstance(t.got, bool) and t.expected
                 and abs(t.got - 100 * t.expected) < 1e-6,
                 "a score is a fraction between 0.0 and 1.0, not a percentage: leave out the * 100."),
        Detector(lambda t: isinstance(t.got, list),
                 "return one number, the fraction right, not the list of answers."),
    ],
)


# notebook order; then the bridge 4 -> 5 (BR1-BR2), which runs inside this kit
TASKS = [W, C1, C2, C3, C4, C5, CP3, ENG1, SCI1, BLD1, ENG2, SCI2, BLD2, BR1, BR2]
