"""Helpers for the provided cells of the two notebooks of the 3-week gap after Lab 2 (standard library only).

They run inside the Lab 2 kit. Each view uses YOUR function and never crashes: if the function isn't written
yet, or raises, it says what to do.

Both notebooks:
    checks_pass(save_predictions, load_predictions)              # one status line; True if every check passes

bridge_2to3.ipynb (files, csv, json and string methods: a preview of Theory S5-S6):
    save_with(save_predictions, predictions, truth, "output/predictions.csv")   # call it safely: True if it ran
    show_file("output/predictions.csv")                         # the first lines of a file, as text
    show_round_trip(load_predictions, "output/predictions.csv", predictions, truth)
    show_json("output/accuracy.json")                           # a JSON file, and what json.load gives back
    show_depot_sheet(parse_depot_line, DEPOT_SHEET)             # every line of a text, taken apart

holiday_quest.ipynb (the reject option of the Postcode Machine):
    show_code_scores(code_score, scores, ["28001", "28046"], "28046")   # a few candidates, the most alike first
    show_best_two(best_two, scored_strips, valid, strips, [0, 5])   # the two best candidates of a few strips
    show_rejects(read_or_reject, scored_strips, valid, margin=100)  # right, misrouted, sent to a person
    show_costs(costs, read_or_reject, scored_strips, valid)         # the cost table, with the counts
    nearest_digit_scores(image, train_images, train_labels)          # a sharper digit score (If you want more)

You never need to change this file; open it if you're curious how the helpers work.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path

from . import cases
from .check import PASS, run_call, run_task, your_line
from .env import current_lab
from .show import not_ready_message as _not_ready, passing_note as _passing, rescue_hint as _rescue_hint
from .show import task_for as _task_for
from .text import say, short, type_name

__all__ = ["DEPOT_SHEET", "checks_pass", "save_with", "show_file", "show_round_trip", "show_json", "show_depot_sheet",
           "show_code_scores", "show_best_two", "show_rejects", "show_costs", "nearest_digit_scores"]

# A page of Garabato's depot sheet, typed by hand (fictional, like the company; the depots agree with data/depots.json).
DEPOT_SHEET = """\
  28001 ; Retiro
28013;Chamartín
 28047 ;  Vallecas
28046 Chamartín
28050 ; Retiro ; Vallecas

  ; Retiro
28033 ; Retiro   """


# ============================================================================= small helpers
def _problem(fn, fallback: str, attempt, what: str) -> None:
    """Say that YOUR function raised, on which line, and what to do next."""
    task, name, _ = _task_for(fn, fallback)
    lines = [f"{name} raised {attempt.error_text} {what}."]
    place = your_line(attempt.error, fn, always_name=True)
    if place:
        lines.append(f"{place[0]} {place[1]}")
    check_text = f"check(\"{task.id}\", {task.name}) shows why, or " if task is not None else ""
    say(*lines, f"Next step: {check_text}{_rescue_hint(task, name)}.")


def _note_if_not_passing(fn, fallback: str) -> None:
    ok, note = _passing(fn, fallback)
    if not ok:
        say(note)


def checks_pass(*functions: object) -> bool:
    """Run the checks of these functions quietly and print one status line, e.g. for Putting It All Together:

        Your checks: BR1 save_predictions PASS · BR2 load_predictions NOT YET · ...

    Returns True when every check passes; otherwise names the first check to run.
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
        first = not_yet[0]
        lines.append(f"Fix {' and '.join(t.id for t in not_yet)} first: check(\"{first.id}\", {first.name}) shows "
                     "what is still wrong. Then run this cell again.")
    say(*lines)
    return not not_yet


def save_with(save_fn, *args) -> bool:
    """Call YOUR save function safely: True if it ran without an error (otherwise say where it failed)."""
    task, name, _ = _task_for(save_fn, getattr(save_fn, "__name__", "your function"))
    message = _not_ready(save_fn, name)
    if message:
        say(message)
        return False
    attempt = run_call(save_fn, args, track=False)
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return False
    if attempt.error is not None:
        _problem(save_fn, name, attempt, "while saving")
        return False
    return True


# ============================================================================= bridge 2 -> 3
def show_file(path: str, max_lines: int = 6) -> None:
    """Print the first lines of a text file exactly as they are stored (or say why there is nothing to show)."""
    file = Path(path)
    if not file.is_file():
        say(f"There is no file {path} yet. Run the cell that writes it first (and check that it PASSes).")
        return
    data = file.read_bytes()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    shown = lines[:max_lines]
    head = f"{path}: {len(lines):,} line{'s' if len(lines) != 1 else ''}, {len(data):,} bytes."
    say(head + (f" The first {len(shown)}:" if len(lines) > len(shown) else ""))
    for line in shown:
        say(f"  {line}")
    if len(lines) > len(shown):
        say(f"  ... and {len(lines) - len(shown):,} more")
    if b"\r\r\n" in data:
        say("Note: there is an empty line after every row. Open the file with newline=\"\" when you write it.")


def show_round_trip(load_fn, path: str, predictions: list, truth: list) -> None:
    """Read the predictions back with YOUR load function and compare them with the lists that were saved."""
    message = _not_ready(load_fn, "load_predictions")
    if message:
        say(message)
        return
    if not Path(path).is_file():
        say(f"There is no file {path} yet: run the save cell (BR1) first, then this cell again.")
        return
    attempt = run_call(load_fn, (path,), track=False)
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return
    if attempt.error is not None:
        _problem(load_fn, "load_predictions", attempt, f"while reading {path}")
        return
    got = attempt.got
    task, name, _ = _task_for(load_fn, "load_predictions")
    if not (isinstance(got, tuple) and len(got) == 2 and all(isinstance(part, list) for part in got)):
        say(f"{name} returned {short(got, 60)} ({type_name(got)}); it should return two lists: "
            "(predictions, truth).", f"Next step: check(\"BR2\", {name}) shows why.")
        return
    _note_if_not_passing(load_fn, "load_predictions")
    loaded_predictions, loaded_truth = got
    say(f"Read back from {path}: {len(loaded_predictions):,} predictions and {len(loaded_truth):,} true labels.")
    say(f"  The first five predictions: {short(loaded_predictions[:5], 60)}"
        + (f" ({type(loaded_predictions[0]).__name__} values)" if loaded_predictions else ""))
    same = loaded_predictions == list(predictions) and loaded_truth == list(truth)
    say(f"  The same as the two lists you saved: {'yes' if same else 'no'}.")
    if not same and any(isinstance(value, str) for value in loaded_predictions[:5]):
        say("  They came back as text ('7'), and '7' == 7 is False: turn each value into an int when you read it.")


def show_json(path: str) -> None:
    """Print a JSON file, then what json.load gives back (look at the keys) and how to get int keys again."""
    file = Path(path)
    if not file.is_file():
        say(f"There is no file {path} yet. Run the cell that writes it first (and check that it PASSes).")
        return
    show_file(path, max_lines=14)
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except ValueError as error:
        say(f"json.load can't read it: it is not JSON ({error}). Write the file with json.dump(..., f, indent=2).")
        return
    say(f"json.load gives back: {short(loaded, 90)}")
    if isinstance(loaded, dict) and loaded:
        key = next(iter(loaded))
        say(f"  The keys are text now ({key!r}, not {key}): JSON keys are always text.")
        if all(isinstance(k, str) and k.isdigit() for k in loaded):
            back = {int(digit): value for digit, value in loaded.items()}
            say("  A dict comprehension turns them back into ints:",
                "  {int(digit): accuracy for digit, accuracy in loaded.items()}",
                f"  -> {short(back, 90)}")


def show_depot_sheet(parse_fn, text: str) -> None:
    """Take every line of `text` apart with YOUR parse function and show what it gives for each one."""
    message = _not_ready(parse_fn, "parse_depot_line")
    if message:
        say(message)
        return
    _note_if_not_passing(parse_fn, "parse_depot_line")
    lines = text.split("\n")
    width = max(len(repr(line)) for line in lines)
    say(f"{len(lines)} lines of the depot sheet, as Python sees them, and what your function gives:")
    kept = 0
    for line in lines:
        attempt = run_call(parse_fn, (line,), track=False)
        if attempt.interrupted:
            say("Stopped (you pressed Interrupt).")
            return
        if attempt.error is not None:
            _problem(parse_fn, "parse_depot_line", attempt, f"on the line {line!r}")
            return
        kept += attempt.got is not None
        say(f"  {repr(line).ljust(width)}  ->  {short(attempt.got, 40)}")
    say(f"{kept} of {len(lines)} lines kept; the others gave None (a person checks them).")


# ============================================================================= holiday quest
def show_code_scores(score_fn, scores: list, codes: list, truth: str) -> None:
    """The total score of a few candidate codes by YOUR code_score, the most alike first."""
    message = _not_ready(score_fn, "code_score")
    if message:
        say(message)
        return
    _note_if_not_passing(score_fn, "code_score")
    rows = []
    for code in codes:
        attempt = run_call(score_fn, (scores, code), track=False)
        if attempt.interrupted:
            say("Stopped (you pressed Interrupt).")
            return
        if attempt.error is not None:
            _problem(score_fn, "code_score", attempt, f"on the code {code!r}")
            return
        rows.append((attempt.got, code))
    say(f"Candidate codes for this strip (true postcode {truth}), the smallest total first:")
    for total, code in sorted(rows, key=lambda row: row[0] if isinstance(row[0], (int, float)) else math.inf):
        note = "   <- the true postcode" if code == truth else ""
        shown = f"{total:>8,.0f}" if isinstance(total, (int, float)) else short(total, 20)
        say(f"  {code}   total {shown}{note}")


def show_best_two(best_two_fn, scored_strips: list, valid: set, strips: list, indices: list) -> None:
    """For a few strips: the true postcode, the two best valid codes by YOUR best_two, and the gap between them."""
    message = _not_ready(best_two_fn, "best_two")
    if message:
        say(message)
        return
    _note_if_not_passing(best_two_fn, "best_two")
    for i in indices:
        scores, truth = scored_strips[i]
        attempt = run_call(best_two_fn, (scores, set(valid)), track=False)
        if attempt.interrupted:
            say("Stopped (you pressed Interrupt).")
            return
        if attempt.error is not None:
            _problem(best_two_fn, "best_two", attempt, f"on strip {strips[i]['strip_id']}")
            return
        got = attempt.got
        try:
            (best, code), (second, other) = got
            gap = second - best
        except (TypeError, ValueError):
            say(f"best_two returned {short(got, 60)}; it should return two (total, code) tuples, the best first.")
            return
        verdict = "right" if code == truth else "WRONG"
        say(f"Strip {strips[i]['strip_id']} (true postcode {truth}):  best {code} ({best:,.0f}, {verdict}) · "
            f"second {other} ({second:,.0f}) · gap {gap:,.0f}")


def _counts(read_fn, scored_strips: list, valid: set, margin: float):
    """(right, misrouted, reviewed) for YOUR read_or_reject at one margin, or None after reporting a problem."""
    right = wrong = review = 0
    for scores, truth in scored_strips:
        attempt = run_call(read_fn, (scores, set(valid), margin), track=False)
        if attempt.interrupted:
            say("Stopped (you pressed Interrupt).")
            return None
        if attempt.error is not None:
            _problem(read_fn, "read_or_reject", attempt, f"(margin {margin:g})")
            return None
        if attempt.got is None:
            review += 1
        elif attempt.got == truth:
            right += 1
        else:
            wrong += 1
    return right, wrong, review


def show_rejects(read_fn, scored_strips: list, valid: set, margin: float) -> None:
    """How YOUR read_or_reject decides on every strip at one margin: right, misrouted, sent to a person."""
    message = _not_ready(read_fn, "read_or_reject")
    if message:
        say(message)
        return
    _note_if_not_passing(read_fn, "read_or_reject")
    counts = _counts(read_fn, scored_strips, valid, margin)
    if counts is None:
        return
    right, wrong, review = counts
    say(f"Margin {margin:g}, {len(scored_strips)} strips: {right} right · {wrong} misrouted · {review} sent to a person.")


def show_costs(costs: object, read_fn=None, scored_strips: list | None = None, valid: set | None = None) -> None:
    """The cost table: for each margin of YOUR costs dict, its cost per 1,000 parcels (and, with read_fn, how
    many strips were right, misrouted and sent to a person)."""
    if not isinstance(costs, dict) or not costs:
        say(f"There is no cost table to show yet (costs is {short(costs, 30)}). Write margin_costs (HQ5), run "
            "its cell and the cell that builds the table, then run this cell again.")
        return
    counts_ok = read_fn is not None and scored_strips is not None and valid is not None and not _not_ready(
        read_fn, "read_or_reject")
    head = f"  {'Margin':>8}" + (f"{'Right':>8}{'Misrouted':>11}{'Reviewed':>10}" if counts_ok else "")
    lines = [head + f"{'Cost per 1,000':>17}"]
    for margin, cost in costs.items():
        row = f"  {margin:>8g}" if isinstance(margin, (int, float)) else f"  {str(margin):>8}"
        if counts_ok:
            counts = _counts(read_fn, scored_strips, valid, margin)
            if counts is None:
                return
            row += f"{counts[0]:>8}{counts[1]:>11}{counts[2]:>10}"
        row += f"{'€':>9}{cost:>8,.2f}" if isinstance(cost, (int, float)) else f"{short(cost, 16):>17}"
        lines.append(row)
    say(*lines)


def nearest_digit_scores(image: list[int], train_images: list[list[int]], train_labels: list[int]) -> list[float]:
    """A sharper digit score (provided): for each digit 0-9, the distance (math.dist) to the CLOSEST training
    image of that digit, instead of to the average image. Lower is better, as with digit_scores."""
    best = [math.inf] * 10
    for train, label in zip(train_images, train_labels):
        distance = math.dist(image, train)
        if distance < best[label]:
            best[label] = distance
    return best
