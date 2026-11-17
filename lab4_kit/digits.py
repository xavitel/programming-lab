"""digits.py: Garabato's digit reader as a command-line tool (Lab 4 · Ship It to the Depot).

Ops doesn't do Jupyter: the night shift needs one command. This file is that command.

In a terminal, in the kit folder (Windows: py instead of python; macOS: python3; uv: uv run python):
    python digits.py --help
    python digits.py inspect data/mnist_test_2k.csv.gz --index 3
    python digits.py train data/mnist_train_10k.csv.gz --out models/centroids.json
    python digits.py predict models/centroids.json samples/parcel_021.txt --show

From the notebook (the notebook route; Colab: !python digits.py --help):
    run_cli(digits.main, "predict models/centroids.json samples/parcel_021.txt --show")

How to work on it: write a function below, SAVE this file (Ctrl+S, or Cmd+S on a Mac), then run its check in
the notebook. Everything marked "Provided" is already implemented: use it, don't modify it.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from labkit.data import load_digits, read_image_csv, read_image_grid
from labkit.reference.lab1 import fit_centroids, predict_centroid, render_digit
from labkit.show import describe

# ============================================================================= Provided: don't modify
KIT_FOLDER = Path(__file__).resolve().parent             # the folder this file is in: the kit folder
DEFAULT_DATA = KIT_FOLDER / "data" / "mnist_test_2k.csv.gz"
MODEL_FORMAT = "garabato-centroids"                      # the first thing load_model checks
MODEL_VERSION = 1


def new_model(centroids: dict[int, list[float]], n_train: int) -> dict:
    """Provided. The model as a dict that json can save: format, version, when, how much data, the centroids.

    JSON keys are always text, so the labels become "0" ... "9"; the averages are rounded to 2 decimals
    (plenty for an average pixel, and the file is about 3 times smaller).
    """
    first = next(iter(centroids.values()), [])
    return {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_train": n_train,
        "image_size": round(len(first) ** 0.5) if first else 28,
        "centroids": {str(label): [round(value, 2) for value in centroids[label]] for label in sorted(centroids)},
    }


def centroids_of(model: dict) -> dict[int, list[float]]:
    """Provided. The centroids of a loaded model, with int labels again: {0: [...], 1: [...], ...}."""
    return {int(label): values for label, values in model["centroids"].items()}


def picture(image: list[int]) -> str:
    """Provided. A text picture of an image ('#' = ink, '.' = paper), then a one-line description of it."""
    width = 14 if len(image) == 196 else 28
    return render_digit(image, width=width) + "\n" + describe(image, width=width)


def load_model(path: Path) -> dict:
    """Provided. Read a model file saved by save_model (C2) and return the dict.

    Text that isn't JSON, or JSON whose "format" isn't MODEL_FORMAT: ValueError("not a valid model file: ...").
    A missing file: open() raises FileNotFoundError by itself (main() reports it). Read it: it shows how to
    turn one error (json.JSONDecodeError) into another that the user can act on.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        model = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"not a valid model file: {path} (it isn't JSON)") from None
    if not isinstance(model, dict) or model.get("format") != MODEL_FORMAT:
        raise ValueError(f"not a valid model file: {path} (its format isn't {MODEL_FORMAT})")
    check_model(model)              # ENG1 (OPTIONAL): wrong version, missing keys, wrong lengths
    return model


# ============================================================================= W · the worked example (provided)
def cmd_inspect(args: argparse.Namespace) -> str:
    """inspect [DATA] [--index N]: describe a digit file and draw one of its images (worked example W).

    Like every command, it RETURNS its text (main() prints it) and raises an error the user can fix
    (here ValueError for an index outside the file), never sys.exit().
    """
    images, labels = load_digits(args.data)
    if not 0 <= args.index < len(images):
        raise ValueError(f"--index must be from 0 to {len(images) - 1}, not {args.index}")
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    per_label = " · ".join(f"{label}: {counts[label]}" for label in sorted(counts))
    lines = [
        f"{Path(args.data).name}: {len(images):,} images of {len(images[0])} pixels",
        f"Images per label: {per_label}",
        f"Image {args.index} (label {labels[args.index]}):",
        picture(images[args.index]),
    ]
    return "\n".join(lines)


# ============================================================================= C1 · the command line
# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · C1: add the train and predict subcommands (inspect is already there), then save the file.
def build_parser() -> argparse.ArgumentParser:
    """The command line of digits.py: one subcommand per job.

    Typed in a terminal                                            args after parse_args
    train data/mnist_train_10k.csv.gz --out models/c.json         command='train', data=..., out='models/c.json',
                                                                   limit=None, force=False
    predict models/c.json samples/parcel_021.txt --show           command='predict', model=..., image=..., show=True
    """
    parser = argparse.ArgumentParser(
        prog="digits",
        description="Garabato's digit reader: train a model, then read handwritten digits from image files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect: the worked example W (already written: read it, then copy its pattern)
    inspect = subparsers.add_parser("inspect", help="describe a digit file and draw one of its images")
    inspect.add_argument("data", nargs="?", default=str(DEFAULT_DATA),
                         help="a digit file (default: data/mnist_test_2k.csv.gz)")
    inspect.add_argument("--index", type=int, default=0, help="which image to draw (default: 0)")

    # TODO 1: train DATA --out MODEL [--limit N] [--force]
    #   train = subparsers.add_parser("train", help="...")
    #   "data" is positional; --out is required; --limit is a whole number (default None); --force is a flag

    # TODO 2: predict MODEL IMAGE [--show]
    #   "model" and "image" are positional; --show is a flag
    return parser


# ============================================================================= C2 · a model is a file
# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · C2: replace `pass` in save_model, then save the file. (Its partner, load_model, is provided
# at the top of this file: read it.)
def save_model(model: dict, path: Path, force: bool = False) -> None:
    """Save `model` as a UTF-8 JSON file at `path`, creating its folder if needed.

    If the file already exists and force is False, raise FileExistsError(f"{path} exists; use --force to
    overwrite") and leave the file as it is. Returns None.
    """
    # TODO 1: path = Path(path); if path.exists() and not force: raise FileExistsError(...)
    # TODO 2: create the folder: path.parent.mkdir(parents=True, exist_ok=True)
    # TODO 3: open(path, "w", encoding="utf-8") and json.dump(model, f)
    pass


# ============================================================================= C3 · train
# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · C3: replace `pass`, then save the file.
def cmd_train(args: argparse.Namespace) -> str:
    """train DATA --out MODEL [--limit N] [--force]: learn the average image of each digit and save it.

    RETURN (don't print) one line such as
        Trained on 10,000 images in S.S s: saved models/centroids.json (K KB)
    """
    # TODO 1: start = time.perf_counter(); images, labels = load_digits(args.data, limit=args.limit)
    # TODO 2: model = new_model(fit_centroids(images, labels), n_train=len(images))   (both provided)
    # TODO 3: save_model(model, Path(args.out), force=args.force)                    (C2)
    # TODO 4: the seconds since start, the file size in KB (Path(args.out).stat().st_size / 1024),
    #         and return the message with an f-string
    pass


# ============================================================================= C4 · predict
# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · C4: replace `pass`, then save the file.
def cmd_predict(args: argparse.Namespace) -> str:
    """predict MODEL IMAGE [--show]: read the digit in one image file.

    RETURN "Prediction: 7"; with --show, picture(image) first, then a newline, then the prediction line.
    The reader depends on the file's suffix: .csv -> read_image_csv, .txt -> read_image_grid, anything else
    -> raise ValueError(f"unsupported image type: {suffix} (use .csv or .txt)").
    """
    # TODO 1: model = load_model(Path(args.model))                                   (C2)
    # TODO 2: suffix = Path(args.image).suffix; choose the reader (or raise ValueError)
    # TODO 3: label = predict_centroid(image, centroids_of(model))                   (both provided)
    # TODO 4: return the text: with args.show, picture(image) + "\n" + the prediction line
    pass


# ============================================================================= C5 · main
# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · C5: complete main (the first two lines are already there), then save the file.
def main(argv: list[str] | None = None) -> int:
    """Run digits.py with the words of a command line; RETURN the exit code.

    0: it worked: print the command's result (stdout). 1: a problem the user can fix: print ONE line,
    f"error: {err}", to stderr, no traceback. 2: the command line itself is wrong: parse_args already
    prints the usage and exits with 2, before your code runs.
    """
    parser = build_parser()
    args = parser.parse_args(argv)                   # argv=None: the words typed in the terminal
    # TODO 1: a dispatch dict: {"inspect": cmd_inspect, "train": cmd_train, "predict": cmd_predict}
    # TODO 2: call the command for args.command; print the string it returns; return 0
    # TODO 3: put the call in try/except (FileNotFoundError, FileExistsError, ValueError) as err:
    #         print(f"error: {err}", file=sys.stderr) and return 1


# ============================================================================= OPTIONAL tasks
# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · ENG1 (OPTIONAL): replace `pass` (the provided load_model already calls check_model).
def check_model(model: dict) -> None:
    """Raise ValueError unless `model` is a complete version-1 centroid model; return None if it is.

    Reject: something that isn't a dict; a missing key ("format", "version", "n_train", "image_size",
    "centroids"); a version other than MODEL_VERSION; centroids that aren't exactly "0" to "9"; a centroid
    that isn't a list of image_size * image_size numbers.
    """
    pass


# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · SCI1 (OPTIONAL): replace `pass`; add "evaluate" to build_parser and to main's dict.
def cmd_evaluate(args: argparse.Namespace) -> str:
    """evaluate DATA --model MODEL [--limit N] [--report FILE]: accuracy, per-digit accuracy and time.

    RETURN three lines (a fourth, "Report saved: FILE", with --report):
        Evaluated models/centroids.json on 2,000 images in S.S s
        Accuracy: NN.N% (n of 2,000)
        Per digit: 0: NN% · 1: NN% · ... · 9: NN%
    """
    pass


# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE · BLD1 (OPTIONAL): replace `pass`; add "batch" to build_parser and to main's dict.
def cmd_batch(args: argparse.Namespace) -> str:
    """batch INBOX --model MODEL [--out predictions.csv]: read every image in a folder, overnight.

    One CSV row per .csv/.txt file (sorted by name): file,prediction ("error: ..." for a file it can't read).
    An existing CSV is first copied to a timestamped backup (predictions_YYYYmmdd_HHMMSS.csv); the files it
    read move to INBOX/archive/YYYYmmdd_HHMMSS/. RETURN one summary line.
    """
    pass


# ============================================================================= Provided: the last line
# `python digits.py ...` runs this file as the program: its __name__ is "__main__", so main() runs with the
# words typed after digits.py, and SystemExit turns main's return value into the exit code (0, 1 or 2).
# `import digits` (the notebook, the tests) skips it: there, __name__ is "digits".
if __name__ == "__main__":
    raise SystemExit(main())
