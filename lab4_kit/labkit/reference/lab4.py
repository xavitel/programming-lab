"""Reference code for Lab 4 · Ship It to the Depot: a finished digits.py, the command-line tool.

Clear, commented versions of the worked example (cmd_inspect), every CORE task (C1 build_parser, C2
save_model, C3 cmd_train, C4 cmd_predict, C5 main; load_model is provided in digits.py, and is here next to
save_model) and every OPTIONAL task (ENG1 check_model,
SCI1 cmd_evaluate, BLD1 cmd_batch) of Lab 4. It is the same code as a finished digits.py, so you can read it
side by side with yours, or borrow one function when you are stuck (in the notebook):

    digits.build_parser = rescue("C1")       # your digits.py now uses the reference parser

The shape of every command-line tool in this course:

    the words typed in a terminal  ->  build_parser().parse_args(argv)  ->  args (a Namespace)
    ->  a command function cmd_xxx(args) that RETURNS a string  ->  main() prints it and returns an exit code:
        0 = it worked · 1 = a problem the user can fix (one line on stderr) · 2 = the command line itself is
        wrong (argparse prints the usage and exits with 2 on its own)

Missed the lab? Read it top to bottom: the provided helpers first, then the tasks in order.

The last section is not part of digits.py: it holds the class and the function of the catch-up notebook between
Lab 4 and Lab 5 (bridge_4to5.ipynb, tasks BR1-BR2), where Lab 1's two functions become one class.
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

# ============================================================================= provided (already in digits.py)
KIT_FOLDER = Path(__file__).resolve().parents[2]       # in digits.py: Path(__file__).resolve().parent
DEFAULT_DATA = KIT_FOLDER / "data" / "mnist_test_2k.csv.gz"
MODEL_FORMAT = "garabato-centroids"                      # the first thing load_model checks
MODEL_VERSION = 1


def new_model(centroids: dict[int, list[float]], n_train: int) -> dict:
    """The model as a dict that json can save: format, version, when, how much data, and the centroids.

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
    """The centroids of a loaded model, with int labels again: {0: [...], 1: [...], ...}."""
    return {int(label): values for label, values in model["centroids"].items()}


def picture(image: list[int]) -> str:
    """A text picture of an image ('#' = ink, '.' = paper), then a one-line description of it."""
    width = 14 if len(image) == 196 else 28
    return render_digit(image, width=width) + "\n" + describe(image, width=width)


# ============================================================================= W · the worked example
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
def build_parser() -> argparse.ArgumentParser:
    """The command line of digits.py: one subcommand per job (C1; evaluate and batch are OPTIONAL)."""
    parser = argparse.ArgumentParser(
        prog="digits",
        description="Garabato's digit reader: train a model, then read handwritten digits from image files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect: the worked example W (already in digits.py)
    inspect = subparsers.add_parser("inspect", help="describe a digit file and draw one of its images")
    inspect.add_argument("data", nargs="?", default=str(DEFAULT_DATA),
                         help="a digit file (default: data/mnist_test_2k.csv.gz)")
    inspect.add_argument("--index", type=int, default=0, help="which image to draw (default: 0)")

    # train (C1)
    train = subparsers.add_parser("train", help="learn a model from a digit file and save it")
    train.add_argument("data", help="a digit file, e.g. data/mnist_train_10k.csv.gz")
    train.add_argument("--out", required=True, help="where to save the model, e.g. models/centroids.json")
    train.add_argument("--limit", type=int, default=None, help="use only the first N images")
    train.add_argument("--force", action="store_true", help="overwrite the model file if it exists")

    # predict (C1)
    predict = subparsers.add_parser("predict", help="read the digit in one image file")
    predict.add_argument("model", help="a model file saved by train")
    predict.add_argument("image", help="a .csv image (one line of 784 numbers) or a .txt image (28 lines of 28)")
    predict.add_argument("--show", action="store_true", help="also draw the image")

    # evaluate (OPTIONAL SCI1)
    evaluate = subparsers.add_parser("evaluate", help="measure a model on a digit file")
    evaluate.add_argument("data", help="a digit file with labels, e.g. data/mnist_test_2k.csv.gz")
    evaluate.add_argument("--model", required=True, help="a model file saved by train")
    evaluate.add_argument("--limit", type=int, default=None, help="use only the first N images")
    evaluate.add_argument("--report", default=None, help="also save the results as a JSON file")

    # batch (OPTIONAL BLD1)
    batch = subparsers.add_parser("batch", help="read every image in a folder and write a CSV of predictions")
    batch.add_argument("inbox", help="a folder of .csv and .txt images")
    batch.add_argument("--model", required=True, help="a model file saved by train")
    batch.add_argument("--out", default="predictions.csv", help="the CSV to write (default: predictions.csv)")
    return parser


# ============================================================================= C2 · a model is a file
def save_model(model: dict, path: Path, force: bool = False) -> None:
    """Save `model` as a UTF-8 JSON file at `path`, creating its folder. Never overwrite an existing file
    unless force is True: raise FileExistsError instead (C2)."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)          # models/ may not exist yet
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f)


def load_model(path: Path) -> dict:
    """Read a model file saved by save_model. A file that isn't one raises ValueError("not a valid model
    file: ..."); a missing file raises FileNotFoundError (open() does that for us). Provided in digits.py."""
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


# ============================================================================= C3 · train
def cmd_train(args: argparse.Namespace) -> str:
    """train DATA --out MODEL [--limit N] [--force]: learn the average image of each digit and save it (C3).

    Returns e.g. "Trained on 10,000 images in S.S s: saved models/centroids.json (K KB)".
    """
    start = time.perf_counter()
    images, labels = load_digits(args.data, limit=args.limit)
    model = new_model(fit_centroids(images, labels), n_train=len(images))
    save_model(model, Path(args.out), force=args.force)
    seconds = time.perf_counter() - start
    size_kb = Path(args.out).stat().st_size / 1024
    return f"Trained on {len(images):,} images in {seconds:.1f} s: saved {args.out} ({size_kb:.0f} KB)"


# ============================================================================= C4 · predict
def cmd_predict(args: argparse.Namespace) -> str:
    """predict MODEL IMAGE [--show]: read the digit in one image file (C4).

    Returns "Prediction: 7"; with --show, the picture of the image comes first.
    """
    model = load_model(Path(args.model))
    suffix = Path(args.image).suffix                 # ".csv", ".txt", ".png", ... (with the dot)
    if suffix == ".csv":
        image = read_image_csv(args.image)
    elif suffix == ".txt":
        image = read_image_grid(args.image)
    else:
        raise ValueError(f"unsupported image type: {suffix} (use .csv or .txt)")
    label = predict_centroid(image, centroids_of(model))
    if args.show:
        return picture(image) + "\n" + f"Prediction: {label}"
    return f"Prediction: {label}"


# ============================================================================= C5 · main
def main(argv: list[str] | None = None) -> int:
    """Run digits.py with the words of a command line and return the exit code (C5).

    0: it worked (the result is on stdout). 1: a problem the user can fix (one line on stderr, no traceback).
    2: the command line itself is wrong: parse_args prints the usage and exits with 2 before we get here.
    """
    parser = build_parser()
    args = parser.parse_args(argv)                   # argv=None: the words typed in the terminal
    commands = {"inspect": cmd_inspect, "train": cmd_train, "predict": cmd_predict,
                "evaluate": cmd_evaluate, "batch": cmd_batch}       # the last two: OPTIONAL SCI1, BLD1
    try:
        output = commands[args.command](args)
    except (FileNotFoundError, FileExistsError, ValueError) as err:      # json.JSONDecodeError is a ValueError
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(output)
    return 0


# ============================================================================= ENG1 · bullet-proof (OPTIONAL)
def check_model(model: dict) -> None:
    """Raise ValueError unless `model` is a complete version-1 centroid model; return None if it is (ENG1).

    json.loads only says "this is JSON". These are the ways a JSON file can still be a broken model.
    """
    if not isinstance(model, dict):
        raise ValueError("a model must be a JSON object (a dict)")
    for key in ("format", "version", "n_train", "image_size", "centroids"):
        if key not in model:
            raise ValueError(f"the model has no {key!r}")
    if model["version"] != MODEL_VERSION:
        raise ValueError(f"model version {model['version']!r} is not supported (this digits.py reads version "
                         f"{MODEL_VERSION})")
    centroids = model["centroids"]
    if not isinstance(centroids, dict) or sorted(centroids) != [str(digit) for digit in range(10)]:
        raise ValueError('the model needs one centroid for each digit, "0" to "9"')
    size = model["image_size"]
    for label, values in centroids.items():
        if not isinstance(values, list) or len(values) != size * size:
            count = len(values) if isinstance(values, list) else "no list of"
            raise ValueError(f"centroid {label} has {count} values; a {size} x {size} image has {size * size}")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"centroid {label} holds {value!r}, which isn't a number")


# ============================================================================= SCI1 · evaluate (OPTIONAL)
def cmd_evaluate(args: argparse.Namespace) -> str:
    """evaluate DATA --model MODEL [--limit N] [--report FILE]: accuracy, per-digit accuracy and time (SCI1)."""
    start = time.perf_counter()
    centroids = centroids_of(load_model(Path(args.model)))
    images, labels = load_digits(args.data, limit=args.limit)
    seen, right = {}, {}                              # per label: how many images, how many read correctly
    for image, label in zip(images, labels):
        seen[label] = seen.get(label, 0) + 1
        if predict_centroid(image, centroids) == label:
            right[label] = right.get(label, 0) + 1
    correct = sum(right.values())
    accuracy = correct / len(labels) if labels else 0.0
    per_digit = {str(label): round(right.get(label, 0) / seen[label], 4) for label in sorted(seen)}
    seconds = time.perf_counter() - start
    lines = [f"Evaluated {args.model} on {len(labels):,} images in {seconds:.1f} s",
             f"Accuracy: {accuracy:.1%} ({correct:,} of {len(labels):,})",
             "Per digit: " + " · ".join(f"{label}: {share:.0%}" for label, share in per_digit.items())]
    if args.report:
        report = {"model": str(args.model), "data": str(args.data), "n": len(labels), "correct": correct,
                  "accuracy": round(accuracy, 4), "per_digit": per_digit, "seconds": round(seconds, 2)}
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        lines.append(f"Report saved: {args.report}")
    return "\n".join(lines)


# ============================================================================= BLD1 · the overnight batch (OPTIONAL)
def cmd_batch(args: argparse.Namespace) -> str:
    """batch INBOX --model MODEL [--out predictions.csv]: read every image in a folder, overnight (BLD1).

    One CSV row per .csv/.txt file (sorted by name): file,prediction ("error: ..." for a file it can't read).
    An existing CSV is first copied to a timestamped backup; the files it read move to
    INBOX/archive/YYYYmmdd_HHMMSS/, so the next run starts with an empty inbox. Returns one summary line.
    """
    inbox = Path(args.inbox)
    if not inbox.is_dir():
        raise FileNotFoundError(f"no such inbox folder: {inbox}")
    centroids = centroids_of(load_model(Path(args.model)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = sorted(list(inbox.glob("*.csv")) + list(inbox.glob("*.txt")))
    rows, unreadable = [], 0
    for file in files:
        try:
            image = read_image_csv(file) if file.suffix == ".csv" else read_image_grid(file)
            rows.append({"file": file.name, "prediction": predict_centroid(image, centroids)})
        except ValueError as err:                    # one bad file must not stop the overnight run
            rows.append({"file": file.name, "prediction": f"error: {err}"})
            unreadable += 1
    out = Path(args.out)
    backup = None
    if out.exists():
        backup = out.with_name(f"{out.stem}_{stamp}{out.suffix}")
        shutil.copy2(out, backup)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    archive = inbox / "archive" / stamp
    if files:
        archive.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.move(str(file), str(archive / file.name))
    summary = f"Read {len(files)} files: {len(files) - unreadable} predicted, {unreadable} unreadable · wrote {out}"
    if backup is not None:
        summary += f" (backup: {backup.name})"
    if files:
        summary += f" · moved them to {archive}"
    return summary


# ============================================================================= bridge 4 -> 5 (BR1-BR2)
# Not part of digits.py: the catch-up notebook between Lab 4 and Lab 5 (bridge_4to5.ipynb) defines these in the
# notebook. Borrow one there with  CentroidModel = rescue("BR1")  or  score = rescue("BR2").
class CentroidModel:
    """Lab 1's fit_centroids and predict_centroid as ONE object that remembers what it learnt (BR1).

        model = CentroidModel().fit(train_images, train_labels)     # learn: one average image per digit
        model.predict_one(test_images[0])                            # read one image with what it learnt
    """

    def __init__(self) -> None:
        self.centroids = {}                     # Step 1: nothing learnt yet; each new object gets its OWN dict

    def fit(self, images: list[list[int]], labels: list[int]) -> "CentroidModel":
        self.centroids = fit_centroids(images, labels)      # Step 2: keep what it learns ON the object (self.)
        return self                                         # so calls can be chained: CentroidModel().fit(...)

    def predict_one(self, image: list[int]) -> int:
        return predict_centroid(image, self.centroids)      # Step 3: the image first, then what fit() learnt


def score(model, images: list[list[int]], labels: list[int]) -> float:
    """The fraction of `images` that `model` reads right (BR2). Any object with predict_one(image) will do."""
    if not labels:                              # no images: nothing to divide by
        return 0.0
    right = 0
    for image, label in zip(images, labels):
        if model.predict_one(image) == label:   # ask the model itself: it never needs to know how it works
            right += 1
    return right / len(labels)


if __name__ == "__main__":
    raise SystemExit(main())
