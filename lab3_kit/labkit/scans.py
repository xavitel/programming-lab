"""Lab 3 · The Scanner Mess: the "garbage in, garbage out" harness (standard library only).

    from labkit.scans import naive_load, run_pipeline, retrain_and_compare, score_batch_b

    naive_images, naive_labels = naive_load("data/parcel_scans_raw.csv")     # the loader that broke the model
    clean_rows, report = run_pipeline(rows, clean_label, parse_pixels, fix_inversion)   # YOUR cleaners
    retrain_and_compare(naive=(naive_images, naive_labels), cleaned=clean_rows)         # retrain, test, compare
    score_batch_b(clean_rows, report)                                                   # the Challenge
    score_batch_c(clean_rows, report)                                                   # ... once, at the end

The model behind every number here is the "average digit" model (Lab 1, SCI1): one average image per
digit, and a new image gets the label of the closest average. It is simple, fast, and very sensitive
to bad training data, which is exactly what this lab needs to show.

You never need to change this file; open it if you're curious how the harness works.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import time
from itertools import repeat
from operator import add, mul
from pathlib import Path

from .data import DATA_DIR, KIT_DIR, cached_digits, downsample
from .text import say, short, type_name

__all__ = ["data_path", "naive_load", "read_rows", "run_pipeline", "retrain_and_compare", "model_table",
           "score_batch_b", "score_batch_c", "make_big_scans", "fit_averages", "average_accuracy",
           "nearest_accuracy", "as_training_set", "scan_key", "REASONS"]

RAW = "parcel_scans_raw.csv"
FIXTURE = "fixtures/scans_fixture.csv"
BATCH_B = "holdout/parcel_scans_batch_b.csv"
BATCH_C = "holdout/parcel_scans_batch_c.csv"
DEPOTS = "depots_reference.csv"
LAST_MONTH = "parcel_scans_last_month.csv"
COLUMNS = ["scan_id", "depot", "scanner", "scanned_at", "label_typed", "postcode", "box", "pixels",
           "notes", "operator_email"]
REASONS = ["duplicate", "bad label", "bad pixels"]     # the CORE rules, in order
LAST_MONTH_ROWS = 1500        # last month's model: the first 1,500 images of data/mnist_train_10k.csv.gz
TEST_ROWS = 2000              # every model is tested on the 2,000 test images of data/mnist_test_2k.csv.gz


# ============================================================================= files
def data_path(name: str | Path) -> Path:
    """Find a Lab 3 data file such as "parcel_scans_raw.csv" or "data/parcel_scans_raw.csv".

    Looks in data/ next to the notebook, then in the kit's data/ folder (and, on the instructor's
    build machine, in the lab's kit_extra/data folder). Raises FileNotFoundError with a fix.
    """
    rel = Path(str(name).replace("\\", "/"))
    if rel.parts and rel.parts[0] == "data":
        rel = Path(*rel.parts[1:])
    candidates = [Path(name), Path("data") / rel, DATA_DIR / rel,
                  KIT_DIR.parent / "labs" / "lab3_scanner_mess" / "kit_extra" / "data" / rel]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"data/{rel.as_posix()} is missing. Fix: re-extract the Lab 3 kit and open the "
                            "whole kit folder in VS Code (File > Open Folder).")


def read_rows(path: str | Path, encoding: str = "utf-8") -> tuple[list[dict[str, str]], list[int]]:
    """The worked example's reader (the same rules as read_scans): (good rows, malformed line numbers)."""
    rows, bad = [], []
    with open(data_path(path), encoding=encoding, newline="") as f:
        for line_number, row in enumerate(csv.DictReader(f), start=2):
            if None in row:
                bad.append(line_number)
            else:
                rows.append(row)
    return rows, bad


def naive_load(path: str | Path = "data/" + RAW) -> tuple[list[list[int]], list[int]]:
    """The loader that broke the model: it skips every row that crashes and trusts everything else.

    For each row: label = int(label_typed); pixels = int(float(v)) for every value. A row is skipped
    only if one of those raises, or if it doesn't have 784 pixels. No duplicates
    check, no range check, no inversion check: exactly what an export "looks fine" to.
    """
    images, labels = [], []
    with open(data_path(path), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                label = int(row["label_typed"])
                pixels = [int(float(value)) for value in row["pixels"].split()]
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
            if len(pixels) == 784:
                images.append(pixels)
                labels.append(label)
    return images, labels


# ============================================================================= the pipeline
def _stub_or_missing(fn: object) -> bool:
    from .check import is_stub
    return not callable(fn) or is_stub(fn)


def _task_hint(fn: object, fallback: str) -> str:
    """'clean_label (C1). Not there yet? Run clean_label = rescue("C1") ...'."""
    from . import cases
    from .env import current_lab

    name = getattr(fn, "__name__", None) or fallback
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    task = (lab_cases.by_name(name) or lab_cases.by_name(fallback)) if lab_cases else None
    if task is None:
        return name
    return f"{task.name} ({task.id}). Not there yet? Run {task.name} = rescue(\"{task.id}\") in a new cell"


def _not_passing(fn: object, fallback: str) -> str | None:
    """A note if fn doesn't pass its check yet (the numbers may then be off), else None."""
    from . import cases
    from .check import PASS, quiet_status
    from .env import current_lab

    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    name = getattr(fn, "__name__", None)
    task = lab_cases.by_name(name) if lab_cases and name else None
    if task is None:
        return None
    try:
        status = quiet_status(task, lab_cases)
    except Exception:
        return None
    if status == PASS:
        return None
    return (f"Note: {task.name} doesn't pass check(\"{task.id}\", {task.name}) yet, so the numbers below may be "
            "off. Run that check to see why.")


def _empty_report(total: int = 0) -> dict:
    return {"total": total, "valid": 0, "invalid": total, "reasons": {}}


def run_pipeline(rows: list[dict[str, str]], label_fn, pixels_fn, fix_fn, *, quiet: bool = False
                 ) -> tuple[list[dict], dict]:
    """Clean `rows` with YOUR three cleaners plugged into the CORE rules, in this order:

    1. the same scan_id seen before        -> "duplicate"   (ids compared after .strip().upper())
    2. label_fn(label_typed) is None       -> "bad label"
    3. pixels_fn(pixels) is None           -> "bad pixels"
    4. otherwise keep {scan_id, depot, label, pixels: fix_fn(pixels)}

    Returns (clean_rows, report) with report = {"total", "valid", "invalid", "reasons": {reason: count}}.
    If a cleaner isn't written yet or raises an error, it prints what to do and returns ([], report).
    """
    try:
        rows = list(rows)
    except TypeError:
        say(f"run_pipeline needs the list of rows from read_scans, not {type_name(rows)}.")
        return [], _empty_report()
    for fn, fallback in ((label_fn, "clean_label"), (pixels_fn, "parse_pixels"), (fix_fn, "fix_inversion")):
        if _stub_or_missing(fn):
            say(f"This cell needs {_task_hint(fn, fallback)}, then run this cell again.")
            return [], _empty_report(len(rows))
    if not quiet:
        for fn, fallback in ((label_fn, "clean_label"), (pixels_fn, "parse_pixels"), (fix_fn, "fix_inversion")):
            note = _not_passing(fn, fallback)
            if note:
                say(note)
    from .check import run_call, your_line

    clean_rows: list[dict] = []
    reasons: dict[str, int] = {}
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        scan_id = str(row.get("scan_id", "")).strip().upper()
        if scan_id in seen:
            reason = "duplicate"
        else:
            seen.add(scan_id)
            reason = None
            for fn, column in ((label_fn, "label_typed"), (pixels_fn, "pixels")):
                attempt = run_call(fn, (row.get(column, ""),), track=False)
                if attempt.error is not None:
                    _report_error(fn, attempt, row, column, number, your_line)
                    return [], _empty_report(len(rows))
                if attempt.got is None:
                    reason = "bad label" if column == "label_typed" else "bad pixels"
                    break
                problem = _wrong_type(fn, attempt.got, column)
                if problem:
                    say(problem, "Nothing was cleaned. Fix it and run its check, then run this cell again.")
                    return [], _empty_report(len(rows))
                if column == "label_typed":
                    label = attempt.got
                else:
                    pixels = attempt.got
            if reason is None:
                attempt = run_call(fix_fn, (pixels,), track=False)
                if attempt.error is not None:
                    _report_error(fix_fn, attempt, row, "pixels", number, your_line)
                    return [], _empty_report(len(rows))
                clean_rows.append({"scan_id": scan_id, "depot": str(row.get("depot", "")).strip(),
                                   "label": label, "pixels": attempt.got})
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
    report = {"total": len(rows), "valid": len(clean_rows), "invalid": len(rows) - len(clean_rows),
              "reasons": {r: reasons[r] for r in REASONS if r in reasons}}
    return clean_rows, report


def _wrong_type(fn, value: object, column: str) -> str | None:
    """A message if a cleaner returned something other than an int (label) or a list of ints (pixels)."""
    name = getattr(fn, "__name__", "your function")
    if column == "label_typed":
        if type(value) is int:
            return None
        return (f"{name} returned {short(value, 30)} ({type_name(value)}), not a digit (int) or None: "
                f"run check(\"C1\", {name}).")
    if isinstance(value, list) and all(type(p) is int for p in value):
        return None
    if isinstance(value, list) and value and isinstance(value[0], str):
        return (f"{name} returned text such as {value[0]!r}, not whole numbers (int): run "
                f"check(\"C2\", {name}). (split() gives strings: convert each one with int().)")
    return (f"{name} returned {short(value, 30)} ({type_name(value)}), not a list of whole numbers (int) or None: "
            f"run check(\"C2\", {name}).")


def _report_error(fn, attempt, row: dict, column: str, number: int, your_line) -> None:
    name = getattr(fn, "__name__", "your function")
    value = row.get(column, "")
    lines = [f"{name} raised {attempt.error_text} on row {number} (scan_id {str(row.get('scan_id', '')).strip()}, "
             f"{column} {short(value, 40)}). A cleaner should return None for a value it can't read, never crash."]
    place = your_line(attempt.error, fn, always_name=True)
    if place:
        lines.append(f"{place[0]} {place[1]}")
    lines.append("Nothing was cleaned. Fix it and run its check, then run this cell again.")
    say(*lines)


# ============================================================================= the model (average digits)
def as_training_set(data: object) -> tuple[list[list[int]], list[int]] | None:
    """(images, labels) from a pair of lists, or from a list of clean rows (dicts with 'pixels' and 'label').
    None (after printing why) if it is neither."""
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], list) and isinstance(data[1], list):
        images, labels = data
    elif isinstance(data, list) and all(isinstance(r, dict) for r in data):
        try:
            images = [r["pixels"] for r in data]
            labels = [r["label"] for r in data]
        except KeyError as error:
            say(f"Each clean row needs the keys 'pixels' and 'label' (missing: {error}).")
            return None
    else:
        say(f"Pass the training data as (images, labels) or as a list of clean rows, not {type_name(data)}.")
        return None
    good_images, good_labels = [], []
    for image, label in zip(images, labels):
        if isinstance(image, list) and len(image) == 784 and all(isinstance(p, (int, float)) for p in image):
            good_images.append(image)
            good_labels.append(label)
    if len(good_images) < len(images):
        say(f"Note: {len(images) - len(good_images)} training images are not lists of 784 numbers; they were left out.")
    return good_images, good_labels


def fit_averages(images: list[list[int]], labels: list) -> dict:
    """The average image of each label: {label: list of 784 floats} (Lab 1's fit_centroids, fast)."""
    totals: dict = {}
    counts: dict = {}
    for image, label in zip(images, labels):
        if label in totals:
            totals[label] = list(map(add, totals[label], image))
            counts[label] += 1
        else:
            totals[label] = list(image)
            counts[label] = 1
    return {label: [t / counts[label] for t in totals[label]] for label in totals}


def average_accuracy(averages: dict, images: list[list[int]], labels: list[int]) -> float:
    """Fraction of images whose closest average has the right label (the smallest squared distance:
    |x - c|^2 = |x|^2 - 2 x.c + |c|^2, and |x|^2 is the same for every c, so it can be left out)."""
    if not averages or not images:
        return 0.0
    keys = list(averages)
    rows = [averages[k] for k in keys]
    norms = [sum(map(mul, c, c)) for c in rows]
    right = 0
    for image, truth in zip(images, labels):
        best = min(range(len(keys)), key=lambda j: norms[j] - 2 * sum(map(mul, image, rows[j])))
        right += keys[best] == truth
    return right / len(images)


def nearest_accuracy(train_images: list[list[int]], train_labels: list, test_images: list[list[int]],
                     test_labels: list[int]) -> float:
    """1-nearest-neighbour accuracy (any image size; ties: the earlier training image).

    math.dist runs the whole distance in C; its square root never changes which image is closest."""
    if not train_images or not test_images:
        return 0.0
    right = 0
    for image, truth in zip(test_images, test_labels):
        distances = list(map(math.dist, repeat(image), train_images))
        right += train_labels[distances.index(min(distances))] == truth
    return right / len(test_images)


def _test_set(n: int = TEST_ROWS) -> tuple[list[list[int]], list[int]]:
    images, labels = cached_digits("mnist_test_2k.csv.gz", TEST_ROWS)
    return images[:n], labels[:n]


def _last_month() -> tuple[list[list[int]], list[int]]:
    return cached_digits("mnist_train_10k.csv.gz", LAST_MONTH_ROWS)


_RESULTS: dict = {}           # accuracy of training sets already measured in this session (by content)


def _fingerprint(images: list, labels: list) -> tuple:
    return (len(images), tuple(labels[:50]), tuple(sum(img) for img in images[:20]), sum(map(len, images[:5])))


def _accuracy_of(images: list, labels: list) -> float:
    key = _fingerprint(images, labels)
    if key not in _RESULTS:
        test_images, test_labels = _test_set()
        _RESULTS[key] = average_accuracy(fit_averages(images, labels), test_images, test_labels)
    return _RESULTS[key]


def _bar(value: float, width: int = 30) -> str:
    filled = max(0, min(width, round(value * width)))
    return "#" * filled + "." * (width - filled)


def retrain_and_compare(**training_sets: object) -> dict[str, float]:
    """Retrain the average-digit model on each training set and test it on the same 2,000 test images.

        retrain_and_compare(naive=(naive_images, naive_labels))
        retrain_and_compare(naive=(naive_images, naive_labels), cleaned=clean_rows)

    Each training set is (images, labels) or a list of clean rows. "last month" (1,500 clean images
    from the old scanner) is always measured first, for comparison. Returns {name: accuracy}.
    """
    sets: list[tuple[str, object]] = [("last month", _last_month())]
    sets += [(name.replace("_", " "), data) for name, data in training_sets.items()]
    results: dict[str, float] = {}
    rows = []
    start = time.perf_counter()
    for name, data in sets:
        pair = as_training_set(data)
        if pair is None:
            continue
        images, labels = pair
        if not images:
            say(f"{name}: no training images (empty), so there is nothing to retrain on.")
            continue
        results[name] = _accuracy_of(images, labels)
        rows.append((name, len(images), results[name]))
    if not rows:
        return results
    say("The average-digit model, retrained on each training set and tested on the same 2,000 test images:")
    width = max(len(name) for name, _, _ in rows)
    for name, n, accuracy in rows:
        say(f"  {name.ljust(width)}  {n:>6,} images  {_bar(accuracy)}  {accuracy:6.1%}")
    say(f"({time.perf_counter() - start:.1f} s)")
    return results


def model_table(**training_sets: object) -> dict[str, dict[str, float]]:
    """Which model suffers from which defect? For each training set: the average-digit model (28 x 28,
    2,000 test images) and 1-nearest-neighbour (14 x 14, the first 500 test images).
    Prints an estimate first (1-NN takes a few seconds per training set)."""
    pairs = []
    for name, data in training_sets.items():
        pair = as_training_set(data)
        if pair is not None and pair[0]:
            pairs.append((name.replace("_", " "), pair))
    if not pairs:
        say("model_table needs at least one training set, e.g. model_table(cleaned=clean_rows).")
        return {}
    total = sum(len(images) for _, (images, _) in pairs)
    say(f"Measuring {len(pairs)} training sets with two models (about {max(1, round(total * 500 * 3e-6))} s "
        "for the nearest-neighbour part)...")
    test_images, test_labels = _test_set()
    small_test = [downsample(image) for image in test_images[:500]]
    results: dict[str, dict[str, float]] = {}
    for name, (images, labels) in pairs:
        averages = _accuracy_of(images, labels)
        small = [downsample([int(p) for p in image]) for image in images]
        nearest = nearest_accuracy(small, labels, small_test, test_labels[:500])
        results[name] = {"average digit": averages, "nearest neighbour": nearest}
    width = max(len(name) for name in results)
    say(f"  {'training set'.ljust(width)}  {'images':>7}  {'average digit (28 x 28)':>24}  "
        f"{'nearest neighbour (14 x 14)':>28}")
    for name, (images, _) in pairs:
        r = results[name]
        say(f"  {name.ljust(width)}  {len(images):>7,}  {r['average digit']:>24.1%}  {r['nearest neighbour']:>28.1%}")
    return results


# ============================================================================= the Challenge: batches B and C
def scan_key(scan_id: object) -> str:
    """The form in which the answer keys store a scan id: the first 16 hex digits of
    sha256("lab3:" + the id, stripped and in capitals). The keys can be compared, not read."""
    return hashlib.sha256(("lab3:" + str(scan_id).strip().upper()).encode("utf-8")).hexdigest()[:16]


def _truth() -> dict:
    path = Path(__file__).resolve().parent / "cases" / "lab3_scans.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_batch(clean_rows: list[dict], report: dict, batch: str = "B") -> dict:
    """Score a pipeline run on data/holdout/parcel_scans_batch_b.csv (batch="B") or ..._batch_c.csv ("C").

    Rejections are compared, scan by scan, with what SHOULD be rejected (a duplicate copy,
    an unreadable label, broken pixels); everything else is a scan worth keeping, even if it needs
    repairing first. Precision: of the rows you rejected, how many deserved it. Recall: of the rows
    that deserved it, how many you rejected. Then the average-digit model is retrained on your clean
    rows and tested on the 2,000 test images. The truth holds scan_key(id) only, never the ids.
    """
    name = f"batch_{str(batch).strip().lower()}"
    try:
        truth = _truth()[name]
        rows_per_key: dict = truth["rows_per_key"]
        should: dict = truth["reject_per_key"]
    except (OSError, KeyError, ValueError):
        say(f"The batch {batch} truth (labkit/cases/lab3_scans.json) is missing: re-extract the kit.")
        return {}
    pair = as_training_set(clean_rows) if isinstance(clean_rows, list) else None
    if pair is None:
        say(f"score_{name}(clean_rows, report) needs the clean rows of YOUR pipeline run on batch {batch}.")
        return {}
    kept: dict[str, int] = {}
    for row in clean_rows:
        key = scan_key(row.get("scan_id", "") if isinstance(row, dict) else "")
        kept[key] = kept.get(key, 0) + 1
    unknown = sum(1 for k in kept if k not in rows_per_key)
    rejected_total = true_positive = should_total = 0
    for key, n_rows in rows_per_key.items():
        rejected = max(0, n_rows - kept.get(key, 0))
        deserved = should.get(key, 0)
        rejected_total += rejected
        should_total += deserved
        true_positive += min(rejected, deserved)
    precision = true_positive / rejected_total if rejected_total else 0.0
    recall = true_positive / should_total if should_total else 1.0
    images, labels = pair
    accuracy = _accuracy_of(images, labels) if images else 0.0
    say(f"Batch {batch}: {truth['rows']:,} well-formed rows; {should_total} deserve to be rejected.")
    say(f"  You kept {len(clean_rows):,} rows and rejected {rejected_total:,}.")
    say(f"  Precision of your rejections: {precision:.1%}  (of the rows you rejected, the share that deserved it)")
    say(f"  Recall of your rejections:    {recall:.1%}  (of the rows that deserved it, the share you rejected)")
    say(f"  Average-digit model retrained on your clean rows: {accuracy:.1%} on the 2,000 test images")
    if isinstance(report, dict) and report.get("total") not in (None, truth["rows"]):
        say(f"  Note: your report says total {report.get('total')}, but batch {batch} has {truth['rows']:,} "
            "well-formed rows. Did every row reach your cleaner?")
    if unknown:
        say(f"  Note: {unknown} kept scan_ids are not in batch {batch}: did you clean the right file?")
    return {"kept": len(clean_rows), "rejected": rejected_total, "should_reject": should_total,
            "precision": precision, "recall": recall, "accuracy": accuracy}


def score_batch_b(clean_rows: list[dict], report: dict) -> dict:
    """score_batch on data/holdout/parcel_scans_batch_b.csv: the batch you look at and improve on."""
    return score_batch(clean_rows, report, "B")


def score_batch_c(clean_rows: list[dict], report: dict) -> dict:
    """score_batch on data/holdout/parcel_scans_batch_c.csv: the same kind of export, other scans. Score it
    once, at the end, without tuning on it: it measures whether your improvements generalise."""
    return score_batch(clean_rows, report, "C")


def make_big_scans(path: str | Path = "output/big_scans.csv", n: int = 100_000, seed: int = 2026) -> Path:
    """Write a big scanner export (n rows, about 300 MB for 100,000) for the streaming alternative of the
    Challenge: rows of the raw export, repeated with new scan ids. Returns the path."""
    source = data_path(RAW)
    with open(source, encoding="utf-8", newline="") as f:
        lines = f.read().splitlines()
    header, body = lines[0], [line for line in lines[1:] if line]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\r\n")
        for i in range(n):
            line = rng.choice(body)
            first, rest = line.split(",", 1)
            f.write(f"PX-{9_000_000 + i:07d},{rest}\r\n")
    say(f"Wrote {out} ({n:,} rows, {out.stat().st_size / 1e6:.0f} MB).")
    return out


_LONG_FIELD = re.compile(r"(?<=,)[^,\"\r\n]{41,}(?=,|$)")


def preview_line(line: str, width: int = 110) -> str:
    """One CSV line exactly as it is in the file (quotes included), with the long pixels column shortened
    so the other columns stay readable. `width` is kept for compatibility; nothing else is cut."""
    def shorten(match: re.Match) -> str:
        value = match.group(0)
        if value.count(" ") <= 20:
            return value
        tokens = value.split()
        return " ".join(tokens[:4]) + f" ... ({len(tokens)} values)"

    return _LONG_FIELD.sub(shorten, line)
