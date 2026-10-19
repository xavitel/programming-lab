"""Helpers for the provided cells of Lab 2 · Read a Postcode, Pythonically (standard library only).

Views that use YOUR functions (they never crash: if a function isn't ready, they say what to do):
    show_neighbours(test_images[0], test_labels[0], train_images, train_labels, neighbours_fn=k_nearest)
    show_vote(majority_vote, test_images[4], train_images, train_labels, k=5, truth=test_labels[4])
    run_test(test_majority_vote, majority_vote, "your majority_vote")     # does a test catch a bug?

Measuring (Theory S4 idioms are not automatically faster: measure, don't guess):
    time_versions({"legacy": legacy.squared_distance, "yours": squared_distance}, image_a, image_b)
    show_line_counts({"legacy_reader.py": [legacy.predict], "yours": [predict_knn]})
    race_distances(squared_distance, fast_distance, test_images[:20], train_images, train_labels)
    compare_sizes(predict_knn, downsample, test_images[:20], test_labels[:20], train_images, train_labels)

The Postcode Machine (the Challenge):
    digit_scores(image, centroids)            # the distance from one digit image to each of the 10 averages
    strip_digits(strip, test_images)          # the 5 digit images of one postcode strip
    show_strip(images, "S001")                # a postcode strip as a text picture
    cost_table(decide, scored_strips, margins)  # the reject option: cost per 1,000 parcels

Other tools: ready(...) (is a function written yet?), show_checks(...) (do the checks of these functions pass?),
show_source(fn) (code with line numbers),
reader_predictions(...) (fast predictions for the report card), split_train(...) (a validation split).
You never need to change this file; open it if you're curious how the helpers work.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import math
import platform
import time

from . import cases
from .check import PASS, is_stub, run_call, run_task, your_line
from .data import cached_digits
from .env import current_lab
from .evaluate import count_code_lines
from .lab1_tools import split_train
from .show import join_pictures as _join, not_ready_message as _not_ready, passing_note as _passing
from .show import picture, pixels_of as _pixels, rescue_hint as _rescue_hint, task_for as _task_for
from .text import say, short, type_name
from .timing import best_time, format_duration, plain_distance

__all__ = ["ready", "show_checks", "show_source", "time_versions", "show_line_counts", "show_neighbours", "show_vote",
           "run_test", "reader_with", "race_distances", "compare_sizes", "reader_predictions", "digit_scores",
           "strip_digits", "show_strip", "cost_table", "split_train"]

# The second function of a two-function task: its task id.
PARTNERS = {"report_card": "BLD1"}
_ORDINALS = ["1st", "2nd", "3rd"] + [f"{n}th" for n in range(4, 21)]


# ============================================================================= is it written yet?
def _help_for(name: str) -> str:
    """'k_nearest (C2), or run: k_nearest = rescue("C2")' (or just the name outside a lab)."""
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    if lab_cases is None:
        return name
    if name in PARTNERS:
        task = lab_cases.find(PARTNERS[name])
        if task is None:
            return name
        return f"{name} ({task.id}), or run: from labkit.reference.lab{lab} import {name}"
    task = lab_cases.by_name(name)
    if task is None:
        return name
    if task.rescue:
        return f"{name} ({task.id}), or run: {name} = rescue(\"{task.id}\")"
    return f"{name} ({task.id})"


def ready(*functions: object) -> bool:
    """True if every function is defined and written (not a stub); otherwise print what's missing, return False.

        if ready("fast_distance", "predict_knn"):
            ...the experiment...

    Give the NAMES (strings): ready() looks them up in the notebook, so a function that was never defined
    (e.g. after a kernel restart, before its def cell was run again) gets a friendly line instead of a NameError.
    Function objects work too.
    """
    caller = inspect.currentframe()
    namespace = caller.f_back.f_globals if caller is not None and caller.f_back is not None else {}
    missing = []
    undefined = []
    for fn in functions:
        if isinstance(fn, str):
            name = fn
            if name not in namespace:
                undefined.append(_help_for(name))
                continue
            fn = namespace[name]
        if not callable(fn):
            missing.append(f"{short(fn, 30)} (not a function)")
        elif is_stub(fn):
            missing.append(_help_for(getattr(fn, "__name__", "a function")))
    if not missing and not undefined:
        return True
    lines = []
    if undefined:
        lines += ["This cell needs " + ("a function that is not defined yet" if len(undefined) == 1 else
                                        "functions that are not defined yet")
                  + " (after a restart, the kernel forgets everything):",
                  *[f"   - {item}" for item in undefined]]
    if missing:
        one = len(missing) == 1
        lines += [f"This cell needs {'a function that is still a stub' if one else 'functions that are still stubs'}:",
                  *[f"   - {item}" for item in missing]]
    total = len(missing) + len(undefined)
    if not missing:
        lines.append("Run the cell with its def (or the rescue line), then run this cell again." if total == 1 else
                     "Run the cells with their defs (or the rescue lines), then run this cell again.")
    else:
        lines.append("Write it, run its cell, then run this cell again." if total == 1 else
                     "Write them, run their cells, then run this cell again.")
    say(*lines)
    return False


def show_checks(*functions: object) -> bool:
    """Run the checks of these functions quietly and print one status line, e.g. for Putting It All Together:

        Your checks: C1 squared_distance PASS · C2 k_nearest NOT YET · ...

    A golden check on a few real images can agree even when a function still has a bug that only rare inputs
    (such as ties) show, so a NOT YET gets a warning. Returns True when every check passes.
    """
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    if lab_cases is None:
        return True
    parts, not_yet = [], []
    for fn in functions:
        task = lab_cases.by_name(getattr(fn, "__name__", ""))
        if task is None:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                status = run_task(task, fn, lab_cases)[0].status
        except KeyboardInterrupt:
            raise
        except Exception:
            status = "NOT YET"
        parts.append(f"{task.id} {task.name} {status}")
        if status != PASS:
            not_yet.append(task)
    lines = ["Your checks: " + " · ".join(parts) + "."]
    if not_yet:
        ids = " and ".join(task.id for task in not_yet)
        first = not_yet[0]
        lines.append(f"{ids} {'is' if len(not_yet) == 1 else 'are'} NOT YET: identical answers on a few test images "
                     "don't prove there is no bug. The checks found one, often in a rare case (such as a tie) that "
                     f"these images never hit: check(\"{first.id}\", {first.name}) shows it. Fix it, then run this "
                     "cell again.")
    say(*lines)
    return not not_yet


# ============================================================================= reading code
def show_source(obj: object) -> None:
    """Print the code of a function (or module) with line numbers, then how many lines of code it has."""
    try:
        lines, first = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        say(f"show_source: I can't read the code of {short(obj, 40)}.")
        return
    first = max(first, 1)
    width = len(str(first + len(lines)))
    say(*[f"{str(first + i).rjust(width)}  {line.rstrip()}" for i, line in enumerate(lines)])
    name = getattr(obj, "__name__", "this code")
    say(f"({name}: {count_code_lines(obj)} lines of code; blank lines, comments and docstrings not counted.)")


def show_line_counts(groups: dict) -> None:
    """Print the lines of code of several groups of functions, e.g. {"legacy_reader.py": [...], "yours": [...]}.

    A function that is still a stub is counted as it is (with a note).
    """
    totals: dict = {}
    rows = []
    for group, functions in groups.items():
        functions = list(functions) if isinstance(functions, (list, tuple)) else [functions]
        parts = []
        for fn in functions:
            n = count_code_lines(fn)
            note = " (still a stub)" if callable(fn) and is_stub(fn) else ""
            parts.append(f"{getattr(fn, '__name__', '?')} {n}{note}")
            totals[group] = totals.get(group, 0) + n
        rows.append((group, totals.get(group, 0), " + ".join(parts)))
    width = max(len(group) for group, _, _ in rows) + 2
    lines = ["Lines of code (blank lines, comments and docstrings not counted):"]
    for group, total, detail in rows:
        lines.append(f"  {group.ljust(width)}{total:>4}   = {detail}")
    say(*lines)


# ============================================================================= measuring
def time_versions(versions: dict, *args, compare_answers: bool = True, **kwargs) -> None:
    """Time one call of each version on the same arguments and check that they give the same answer.

    versions: {"a label": function, ...}. The first version's answer is the one the others must match
    (compare_answers=False skips that column, e.g. for a version that returns the square root on purpose).
    """
    rows, times = [], {}
    first_answer, have_first = None, False
    for label, fn in versions.items():
        if not callable(fn) or is_stub(fn):
            rows.append((label, "not written yet", _help_for(getattr(fn, "__name__", label))))
            continue
        attempt = run_call(fn, args, kwargs, track=False)
        if attempt.error is not None:
            place = your_line(attempt.error, fn, always_name=True)
            rows.append((label, "raised an error", f"{attempt.error_text}" + (f" · {place[0]} {place[1]}" if place else "")))
            continue
        seconds = best_time(fn, *args, budget_s=0.15, **kwargs)
        times[label] = seconds
        if not compare_answers:
            same = f"answer {short(attempt.got, 25)}"
        elif not have_first:
            first_answer, have_first = attempt.got, True
            same = "(the answer the others must give)"
        else:
            same = "yes" if attempt.got == first_answer else f"NO: {short(attempt.got, 25)} instead of {short(first_answer, 25)}"
        rows.append((label, format_duration(seconds), same))
    width = max(len(label) for label, _, _ in rows) + 2
    lines = [f"  {'Version'.ljust(width)}{'Time per call':>15}   {'Same answer?' if compare_answers else 'Answer'}"]
    lines += [f"  {label.ljust(width)}{when:>15}   {same}" for label, when, same in rows]
    if len(times) > 1:
        fastest = min(times, key=times.get)
        lines.append(f"Fastest on this laptop (Python {platform.python_version()}): {fastest}.")
    say(*lines)


def reader_with(distance_fn, image: list[int], train_images: list[list[int]], train_labels: list[int],
                k: int = 3) -> int:
    """A provided k-NN reader that uses the distance function you give it (same tie rules as the CORE:
    equal distances keep the training order, a tied vote goes to the closest tied neighbour)."""
    pairs = [(distance_fn(image, train), label) for train, label in zip(train_images, train_labels)]
    counts: dict = {}
    for _distance, label in sorted(pairs, key=lambda pair: pair[0])[:k]:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def _timed_predictions(predict, images: list) -> tuple[list | None, float, str | None]:
    """(predictions, seconds, problem) for predict(image) over the images; never raises."""
    out = []
    start = time.perf_counter()
    for i, image in enumerate(images):
        attempt = run_call(predict, (image,), track=False)
        if attempt.interrupted:
            return None, 0.0, "stopped (you pressed Interrupt)"
        if attempt.error is not None:
            place = your_line(attempt.error, None, always_name=True)
            return None, 0.0, (f"raised {attempt.error_text} on image #{i}" + (f" ({place[0]} {place[1]})" if place else ""))
        out.append(attempt.got)
    return out, time.perf_counter() - start, None


def race_distances(distance_a, distance_b, images: list, train_images: list, train_labels: list,
                   k: int = 3) -> None:
    """Read the same images with the k-NN reader twice, once per distance function: time per digit and
    identical answers. (Same answer first, faster second.)"""
    if not ready(distance_a, distance_b):
        return
    results = {}
    for fn in (distance_a, distance_b):
        name = getattr(fn, "__name__", "distance")
        predictions, seconds, problem = _timed_predictions(
            lambda image, fn=fn: reader_with(fn, image, train_images, train_labels, k), images)
        if problem:
            say(f"The reader with {name} {problem}. Fix {name} first (its check shows why).")
            return
        results[name] = (predictions, seconds)
    (name_a, (pred_a, sec_a)), (name_b, (pred_b, sec_b)) = results.items()
    same = sum(1 for p, q in zip(pred_a, pred_b) if p == q)
    n = len(images)
    ratio = f"   ({sec_a / sec_b:.1f} times as fast)" if sec_b > 0 and sec_a > sec_b else ""
    say(f"k-NN reader (k = {k}), {n} test images, each compared with {len(train_images):,} training images:",
        f"  with {name_a.ljust(18)}{format_duration(sec_a / n):>9} per digit",
        f"  with {name_b.ljust(18)}{format_duration(sec_b / n):>9} per digit{ratio}",
        f"Identical predictions: {same} of {n}")


def compare_sizes(predict_fn, shrink_fn, images: list, labels: list, train_images: list, train_labels: list,
                  k: int = 3) -> None:
    """The same reader on 28 x 28 images and on 14 x 14 images made by YOUR shrink function (SCI1):
    time per digit, accuracy on these images, and how many answers are identical."""
    if not ready(predict_fn, shrink_fn):
        return
    shrink_name = getattr(shrink_fn, "__name__", "your function")
    start = time.perf_counter()
    small_train, problem = [], None
    for image in train_images:
        attempt = run_call(shrink_fn, (image,), track=False)
        if attempt.error is not None or not isinstance(attempt.got, list) or len(attempt.got) != 196:
            problem = attempt.error_text or f"returned {short(attempt.got, 30)} ({type_name(attempt.got)})"
            break
        small_train.append(attempt.got)
    if problem:
        say(f"{shrink_name} {problem}, not 196 pixels. check(\"SCI1\", {shrink_name}) shows why.")
        return
    shrink_seconds = time.perf_counter() - start
    small_images = [shrink_fn(image) for image in images]
    rows = {}
    for size, test, train in ((28, images, train_images), (14, small_images, small_train)):
        predictions, seconds, problem = _timed_predictions(
            lambda image, train=train: predict_fn(image, train, train_labels, k), test)
        if problem:
            name = getattr(predict_fn, "__name__", "your reader")
            say(f"{name} {problem}. check(\"C4\", {name}) shows why.")
            return
        right = sum(1 for p, t in zip(predictions, labels) if p == t)
        rows[size] = (predictions, seconds / len(test), right / len(test))
    same = sum(1 for p, q in zip(rows[28][0], rows[14][0]) if p == q)
    n = len(images)
    lines = [f"{getattr(predict_fn, '__name__', 'reader')} (k = {k}) on {n} test images, "
             f"{len(train_images):,} training images:",
             f"  {'Images':<12}{'Pixels':>8}{'Time per digit':>16}{'Right':>8}"]
    for size in (28, 14):
        _, per_digit, acc = rows[size]
        lines.append(f"  {f'{size} x {size}':<12}{size * size:>8}{format_duration(per_digit):>16}"
                     f"{round(acc * n):>5} of {n}")
    lines.append(f"Identical answers: {same} of {n} · shrinking the {len(train_images):,} training images with "
                 f"{shrink_name} took {format_duration(shrink_seconds)} (once).")
    say(*lines)


def reader_predictions(n_test: int = 500, k: int = 3, n_train: int = 1000) -> tuple[list[int], list[int]]:
    """Fast predictions for the report card: the k-NN reader on 14 x 14 images with math.dist (provided).

    Returns (predictions, truth) for the first n_test test images (about 2 s for 500).
    """
    train_images, train_labels = cached_digits("mnist_train_10k_14x14.csv.gz", n_train)
    test_images, test_labels = cached_digits("mnist_test_2k_14x14.csv.gz", n_test)
    predictions = [reader_with(math.dist, image, train_images, train_labels, k) for image in test_images]
    return predictions, list(test_labels)


# ============================================================================= views of YOUR functions
def _closest_index(distance: object, label: object, train_labels: list, used: set, distances: list) -> int | None:
    """The first training image not used yet with this distance and label (None if there is none)."""
    for i, (d, lab) in enumerate(zip(distances, train_labels)):
        if i not in used and d == distance and lab == label:
            return i
    return None


def show_neighbours(image: object, truth: object, train_images: list, train_labels: list, *, neighbours_fn=None,
                    k: int = 3) -> None:
    """Show a test image, its true label, and the k nearest training images that YOUR k_nearest returns,
    each with its label and distance (or say what to do if k_nearest isn't ready)."""
    pixels, problem = _pixels(image)
    if pixels is None:
        say(f"show_neighbours: the test image: {problem}")
        return None
    message = _not_ready(neighbours_fn, "k_nearest")
    if message:
        say(message)
        return None
    task, name, _ = _task_for(neighbours_fn, "k_nearest")
    check_text = f"check(\"{task.id}\", {task.name}) shows why, or " if task is not None else ""
    attempt = run_call(neighbours_fn, (list(image), [list(t) for t in train_images], list(train_labels), k),
                       track=False)
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return None
    got = attempt.got
    if attempt.error is not None:
        lines = [f"{name} raised {attempt.error_text} on this image."]
        place = your_line(attempt.error, neighbours_fn, always_name=True)
        if place:
            lines.append(f"{place[0]} {place[1]}")
        say(*lines, f"Next step: {check_text}{_rescue_hint(task, name)}.")
        return None
    shape_ok = (isinstance(got, list) and got and all(isinstance(p, tuple) and len(p) == 2 for p in got))
    if not shape_ok:
        say(f"{name} returned {short(got, 50)} ({type_name(got)}); it should return a list of (distance, label) "
            f"tuples, the closest first.", f"Next step: {check_text}{_rescue_hint(task, name)}.")
        return None
    ok, note = _passing(neighbours_fn, "k_nearest")
    if not ok:
        say(note)
    distances = [plain_distance(pixels, list(t)) for t in train_images]
    used: set = set()
    found = []
    for distance, label in got:
        index = _closest_index(distance, label, train_labels, used, distances)
        if index is not None:
            used.add(index)
        found.append(index)
    truth_text = f"true label {truth}" if truth is not None else "true label unknown"
    shown = [(pixels, f"Test image ({truth_text})")]
    for n, ((distance, label), index) in enumerate(zip(got, found)):
        if index is not None:
            shown.append((list(train_images[index]), f"{_ORDINALS[n]} nearest: label {label}"))
    say(f"The test image and the {len(got)} nearest training images, by your {name}:")
    for start in range(0, len(shown), 2):
        pair = shown[start:start + 2]
        say(*_join([picture(values, 28 if len(values) == 784 else 14) for values, _ in pair],
                   [caption for _, caption in pair]))
    say(f"What your {name} returned (distance, label), the closest first:")
    for n, ((distance, label), index) in enumerate(zip(got, found)):
        where = f"training image #{index}" if index is not None else "no training image is this far away with this label"
        say(f"  {_ORDINALS[n]} nearest   {short((distance, label), 40)}   {where}")
    labels = [label for _distance, label in got]
    say(f"Their labels: {', '.join(map(str, labels))}. Which one wins the vote? That's C3.")


def show_vote(vote_fn, image: object = None, train_images: list | None = None, train_labels: list | None = None,
              *, k: int = 5, truth: object = None, neighbours: list | None = None) -> None:
    """Show a vote counted by YOUR majority_vote: the neighbours (closest first), the votes per label, and the
    winner. Give either an image (its k nearest training images are found for you) or neighbours=[...]."""
    message = _not_ready(vote_fn, "majority_vote")
    if message:
        say(message)
        return
    task, name, _ = _task_for(vote_fn, "majority_vote")
    if neighbours is None:
        pixels, problem = _pixels(image)
        if pixels is None or not train_images or train_labels is None:
            say(f"show_vote: give a test image with the training images and labels, or neighbours=[...] "
                f"({problem or 'no training data'}).")
            return
        pairs = [(plain_distance(pixels, list(t)), label) for t, label in zip(train_images, train_labels)]
        neighbours = sorted(pairs, key=lambda pair: pair[0])[:k]
        intro = f"The {len(neighbours)} training images nearest to the test image" + (
            f" (true label {truth})" if truth is not None else "") + ", closest first:"
    else:
        neighbours = list(neighbours)
        intro = "The neighbours, closest first:"
    counts: dict = {}
    for _distance, label in neighbours:
        counts[label] = counts.get(label, 0) + 1
    lines = [intro]
    for n, (distance, label) in enumerate(neighbours):
        lines.append(f"  {_ORDINALS[n]:>4}   distance {distance:>11,}   label {label}")
    lines.append("Votes per label (in the order they first appear):")
    for label, count in counts.items():
        lines.append(f"  label {label}: {'#' * count} {count}")
    top = max(counts.values())
    tied = [label for label, count in counts.items() if count == top]
    say(*lines)
    attempt = run_call(vote_fn, ([tuple(p) for p in neighbours],), track=False)
    check_text = f"check(\"{task.id}\", {task.name}) shows why, or " if task is not None else ""
    if attempt.error is not None:
        lines = [f"{name} raised {attempt.error_text} on these neighbours."]
        place = your_line(attempt.error, vote_fn, always_name=True)
        if place:
            lines.append(f"{place[0]} {place[1]}")
        say(*lines, f"Next step: {check_text}{_rescue_hint(task, name)}.")
        return
    ok, note = _passing(vote_fn, "majority_vote")
    if not ok:
        say(note)
    tie_text = (f" A tie between labels {', '.join(map(str, tied))}: the closest of the tied neighbours should win."
                if len(tied) > 1 else "")
    line = f"Your {name} says: {short(attempt.got, 30)}.{tie_text}"
    if truth is not None:
        line += f" The true label is {truth}."
    say(line)


def run_test(test_fn, fn, label: str = "", *, buggy: bool | None = None) -> None:
    """Run a test function on one version of a function and say what happened, in words.

    buggy=True: `fn` is known to be wrong, so the test should FAIL on it (then it "caught the bug").
    buggy=False: `fn` is known to be right, so the test should pass.
    """
    label = label or getattr(fn, "__name__", "this version")
    if not callable(test_fn) or is_stub(test_fn):
        say(_not_ready(test_fn, getattr(test_fn, "__name__", "your test")) or "Write the test first.")
        return
    if not callable(fn) or is_stub(fn):
        say(_not_ready(fn, getattr(fn, "__name__", "the function")) or f"{label} is not written yet.")
        return
    test_name = getattr(test_fn, "__name__", "your test")
    attempt = run_call(test_fn, (fn,), track=False)
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return
    if attempt.error is None:
        verdict = {True: "passed, so this test MISSED the bug: add an assert that this version gets wrong.",
                   False: "passed, as it should on a correct version."}.get(buggy, "passed (no assert failed).")
        say(f"{test_name} on {label}: {verdict}")
        return
    if isinstance(attempt.error, AssertionError):
        place = your_line(attempt.error, test_fn)
        what = f"FAILED at {place[0][:-1].lower()}: {place[1]}" if place else "FAILED"
        if not place and str(attempt.error):
            what += f" ({attempt.error})"
    else:
        what = f"FAILED with {attempt.error_text} (an error also counts as failing)"
    ending = {True: " The test caught the bug.",
              False: " On a correct version, that means this assert expects a wrong answer: work it out by hand."
              }.get(buggy, "")
    say(f"{test_name} on {label}: {what}.{ending}")


# ============================================================================= the Postcode Machine
def digit_scores(image: list[int], centroids: dict) -> list[float]:
    """The distance (math.dist) from one digit image to the average image of each digit 0-9.

    scores[d] is small when the image looks like the average d. Lower is better.
    """
    return [math.dist(image, centroids[digit]) for digit in range(10)]


def strip_digits(strip: dict, test_images: list) -> list:
    """The 5 digit images of a postcode strip (a dict from load_postcodes), left to right."""
    return [test_images[i] for i in strip["indices"]]


def show_strip(images: list, caption: str = "") -> None:
    """A postcode strip as one text picture: each digit shrunk to 14 x 14 and cut to its ink, side by side."""
    from .data import downsample

    small = []
    for image in images:
        pixels, problem = _pixels(image)
        if pixels is None:
            say(f"show_strip: {problem}")
            return
        small.append(downsample(pixels) if len(pixels) == 784 else pixels)
    rows = [r for r in range(14) if any(digit[r * 14 + c] for digit in small for c in range(14))] or list(range(14))
    columns = []
    for digit in small:
        ink = [c for c in range(14) if any(digit[r * 14 + c] for r in range(14))] or [6, 7]
        columns.append(range(max(0, ink[0] - 1), min(14, ink[-1] + 2)))
    lines = []
    for r in range(rows[0], rows[-1] + 1):
        parts = ["".join(" .:-=+*#%@"[digit[r * 14 + c] * 10 // 256] * 2 for c in cols)
                 for digit, cols in zip(small, columns)]
        lines.append("|" + "|".join(parts) + "|")
    border = "+" + "+".join("-" * (2 * len(cols)) for cols in columns) + "+"
    say(*([caption] if caption else []), border, *lines, border)


def cost_table(decide, scored_strips: list, margins: list, *, review_eur: float = 0.35,
               misroute_eur: float = 7.80) -> dict:
    """The reject option, in euros: for each margin, run decide(scores, margin) on every strip and count.

    scored_strips: [(scores, true postcode), ...]; decide returns a postcode (str) or None (send the parcel
    to a person: manual review). A wrong postcode is a misrouted parcel. Returns {margin: cost per 1,000}.
    """
    name = getattr(decide, "__name__", "your function")
    if not ready(decide):
        return {}
    n = len(scored_strips)
    lines = [f"{n} postcode strips · manual review €{review_eur:.2f} per parcel · misroute €{misroute_eur:.2f} per parcel",
             f"  {'Margin':>8}{'Right':>8}{'Misrouted':>11}{'Reviewed':>10}{'Cost per 1,000':>17}"]
    costs = {}
    for margin in margins:
        right = wrong = review = 0
        for scores, truth in scored_strips:
            attempt = run_call(decide, (scores, margin), track=False)
            if attempt.error is not None:
                place = your_line(attempt.error, decide, always_name=True)
                say(f"{name} raised {attempt.error_text} (margin {margin})" + (f" · {place[0]} {place[1]}" if place else ""))
                return costs
            if attempt.got is None:
                review += 1
            elif attempt.got == truth:
                right += 1
            else:
                wrong += 1
        cost = 1000 * (review * review_eur + wrong * misroute_eur) / n
        costs[margin] = cost
        lines.append(f"  {margin:>8g}{right:>8}{wrong:>11}{review:>10}{'€':>9}{cost:>8,.2f}")
    say(*lines)
    return costs
