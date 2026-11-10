"""Views for the provided cells of Lab 3 · The Scanner Mess (standard library only).

    from labkit.lab3_tools import (ready, show_file, show_rows, show_labels, show_pixels, show_fix,
                                   show_report, show_outputs, depot_table, find_suspects, show_suspects)

Each view runs YOUR function on real rows of the export and shows what it did, right after the task.
They never crash: while your function is still a stub, a view prints one line saying what to do (and
how to borrow a working version with rescue); if your function raises an error, it names your line
and the check to run.

You never need to change this file; open it if you're curious how the views work.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import cases
from .check import PASS, is_stub, quiet_status, run_call, your_line
from .env import current_lab
from .scans import REASONS, _truth, data_path, preview_line
from .show import side_by_side, show_digit
from .text import say, short, type_name

__all__ = ["ready", "show_file", "show_rows", "show_labels", "show_pixels", "show_fix", "show_report",
           "show_mine", "save_and_show", "show_outputs", "depot_table", "find_suspects", "show_suspects",
           "pick_rows", "profile_csv", "show_raw_line", "show_scanners", "show_inversion"]


FINALE = "Putting It All Together"


# ============================================================================= "is it ready?"
def _task_of(fn: object, fallback: str):
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    name = getattr(fn, "__name__", None) or fallback
    if lab_cases is None:
        return None, name, None
    return lab_cases.by_name(name) or lab_cases.by_name(fallback), name, lab_cases


def _rescue_text(task, name: str) -> str:
    if task is not None and task.rescue:
        return f"run {task.name} = rescue(\"{task.id}\") in a new cell, then run this cell again"
    return "finish it, run its cell, then run this cell again"


def _needs(fn: object, fallback: str) -> str | None:
    """'This cell needs clean_label (C1). Not there yet? Run ...' if fn is missing or a stub, else None."""
    task, name, _ = _task_of(fn, fallback)
    if callable(fn) and not is_stub(fn):
        return None
    label = f"{task.name} ({task.id})" if task is not None else name
    text = _rescue_text(task, name)
    return f"This cell needs {label}. Not there yet? {text[0].upper()}{text[1:]}."


def ready(*functions: object) -> bool:
    """True if every function is written (not a stub); otherwise print what's missing and return False."""
    for fn in functions:
        message = _needs(fn, getattr(fn, "__name__", "a function"))
        if message:
            say(message)
            return False
    return True


def _passing_note(fn: object, fallback: str) -> str | None:
    task, name, lab_cases = _task_of(fn, fallback)
    if task is None or getattr(fn, "__name__", None) != task.name:
        return None
    try:
        status = quiet_status(task, lab_cases)
    except Exception:
        return None
    if status == PASS:
        return None
    return (f"Note: {task.name} doesn't pass check(\"{task.id}\", {task.name}) yet, so some answers below may "
            "be wrong. Run that check to see why.")


def _call(fn: object, args: tuple, fallback: str, what: str):
    """Call the student's function safely: (True, result) or (False, None) after printing what happened."""
    attempt = run_call(fn, args, track=False)
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return False, None
    if attempt.error is not None:
        task, name, _ = _task_of(fn, fallback)
        lines = [f"{name} raised {attempt.error_text} on {what}."]
        place = your_line(attempt.error, fn, always_name=True)
        if place:
            lines.append(f"{place[0]} {place[1]}")
        check_text = f"check(\"{task.id}\", {task.name}) shows why, or " if task is not None else ""
        lines.append(f"Next step: {check_text}{_rescue_text(task, name)}.")
        say(*lines)
        return False, None
    return True, attempt.got


# ============================================================================= rows and files
def _shown(value: object, limit: int = 28) -> str:
    """A cell of a table: repr for text (so spaces and empty values show), shortened."""
    if isinstance(value, str) and len(value) > 40 and value.count(" ") > 20:
        tokens = value.split()
        return repr(" ".join(tokens[:3]) + " ...") + f" ({len(tokens)} values)"
    text = repr(value) if isinstance(value, str) else short(value, limit)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def show_file(path: str = "data/parcel_scans_raw.csv", n: int = 6, *, encoding: str = "utf-8") -> None:
    """Print the first n lines of a CSV file as they are, with the long pixels column shortened."""
    try:
        with open(data_path(path), encoding=encoding, newline="") as f:
            lines = [f.readline().rstrip("\r\n") for _ in range(n)]
    except (OSError, UnicodeDecodeError) as error:
        say(f"show_file: {type(error).__name__}: {error}")
        return
    say(f"The first {n} lines of {path} (the pixels column is shortened):")
    for number, line in enumerate(lines, start=1):
        say(f"  line {number}: {preview_line(line, 150)}")


def show_rows(rows: list, n: int = 5, columns: list[str] | None = None, *, start: int = 0) -> None:
    """Print rows (dicts) as a table: one row per line, text values in quotes so spaces and empty values show."""
    if not isinstance(rows, list) or not rows:
        say("show_rows: there are no rows to show (run the cell that reads the file first).")
        return
    chosen = [r for r in rows[start:start + n] if isinstance(r, dict)]
    if not chosen:
        say("show_rows: these rows are not dicts.")
        return
    columns = columns or [c for c in chosen[0] if c is not None]
    for r in chosen:
        say("  " + " · ".join(f"{c} {_shown(r.get(c))}" for c in columns))


def pick_rows(rows: list[dict], *, scanner: str | None = None, contains: str | None = None, n: int = 1) -> list[dict]:
    """The first n rows (in file order) from a given scanner and/or whose pixels contain a given text."""
    found = []
    for row in rows:
        if scanner is not None and row.get("scanner") != scanner:
            continue
        if contains is not None and contains not in str(row.get("pixels", "")):
            continue
        found.append(row)
        if len(found) == n:
            break
    return found


def _int_pixels(row: dict) -> list[int] | None:
    """The pixels of a row as ints, if they are 784 whole numbers from 0 to 255 (for the views only)."""
    tokens = str(row.get("pixels", "")).split()
    if len(tokens) != 784 or not all(t.isdigit() for t in tokens):
        return None
    values = [int(t) for t in tokens]
    return values if max(values) <= 255 else None


# ============================================================================= Meet the Data: a first look
# Text that people (or programs) type when a value is missing: a profile counts it apart from empty cells,
# because to Python it is ordinary text, not "nothing".
MISSING_WORDS = {"n/a", "na", "none", "null", "unknown", "-", "?", "nan"}


def _number_kind(token: str) -> str:
    """'whole' (an int() can read it), 'decimal' (only float() can), or 'text'."""
    try:
        int(token)
        return "whole"
    except ValueError:
        pass
    try:
        float(token)
    except ValueError:
        return "text"
    return "decimal"


def profile_csv(path: str = "data/parcel_scans_raw.csv", *, encoding: str = "utf-8") -> dict | None:
    """A first look at a CSV file (exploratory data analysis, standard library only).

    Prints how many data lines and columns it has, then for every column: how many cells are empty, how
    many hold text that MEANS missing ("N/A", "unknown", ...), how many different values there are, and the
    most common ones. The pixels column gets its own summary: values per row, whole numbers or decimals,
    the smallest and largest value, and how many images are mostly white (mean pixel above 127).
    Returns the numbers as a dict, or None if the file can't be read.
    """
    try:
        with open(data_path(path), encoding=encoding, newline="") as f:
            lines = list(csv.reader(f))
    except (OSError, UnicodeDecodeError) as error:
        say(f"profile_csv: {type(error).__name__}: {error}")
        return None
    if not lines:
        say(f"profile_csv: {path} is empty.")
        return None
    header, data = lines[0], lines[1:]
    odd = [n for n, values in enumerate(data, start=2) if len(values) != len(header)]
    say(f"{path}: {len(data):,} data lines (plus the header), {len(header)} columns.")
    if odd:
        shown = ", ".join(str(n) for n in odd[:5]) + (" ..." if len(odd) > 5 else "")
        say(f"  {len(odd)} line(s) do NOT have {len(header)} values (lines {shown}): they are left out below.")
    rows = [values for values in data if len(values) == len(header)]
    result: dict = {"lines": len(data), "columns": header, "odd_lines": odd, "per_column": {}}
    say(f"  {'column':<15}{'empty':>7}{'missing-like':>14}{'different':>11}   most common values")
    for i, name in enumerate(header):
        values = [values[i] for values in rows]
        if name == "pixels":
            continue
        empty = sum(1 for v in values if not v.strip())
        missing = sum(1 for v in values if v.strip().lower() in MISSING_WORDS)
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        common = sorted(counts, key=lambda v: -counts[v])[:3]
        shown = ", ".join(_shown(v, 22) for v in common)
        say(f"  {name:<15}{empty:>7,}{missing:>14,}{len(counts):>11,}   {shown}")
        result["per_column"][name] = {"empty": empty, "missing_like": missing, "different": len(counts)}
    if "pixels" in header:
        i = header.index("pixels")
        lengths: dict[int, int] = {}
        kinds = {"whole": 0, "decimal": 0, "text": 0}
        smallest = largest = None
        white = 0
        for values in rows:
            tokens = values[i].split()
            lengths[len(tokens)] = lengths.get(len(tokens), 0) + 1
            row_kinds = {_number_kind(t) for t in tokens}
            kind = "text" if "text" in row_kinds else ("decimal" if "decimal" in row_kinds else "whole")
            kinds[kind] += 1
            if kind == "whole" and tokens:
                numbers = [int(t) for t in tokens]
                smallest = min(numbers) if smallest is None else min(smallest, min(numbers))
                largest = max(numbers) if largest is None else max(largest, max(numbers))
                if sum(numbers) / len(numbers) > 127:
                    white += 1
        per_row = ", ".join(f"{n} values in {c:,} rows" for n, c in sorted(lengths.items(), key=lambda kv: -kv[1]))
        say(f"  pixels: {per_row}.")
        say(f"          rows of whole numbers {kinds['whole']:,} · rows with decimals {kinds['decimal']:,} · "
            f"rows with text in them {kinds['text']:,}")
        if smallest is not None:
            say(f"          in the whole-number rows: smallest value {smallest}, largest {largest}; "
                f"{white:,} images are mostly white (mean pixel above 127).")
        result["pixels"] = {"values_per_row": lengths, "kinds": kinds, "smallest": smallest, "largest": largest,
                            "mostly_white": white}
    return result


def show_raw_line(path: str = "data/parcel_scans_raw.csv", line_number: int = 143, *,
                  encoding: str = "utf-8") -> None:
    """One line of a CSV file three ways: as it is in the file, split at the commas next to the header's
    column names, and as the dict csv.DictReader makes of it (where an extra value goes under the key None)."""
    try:
        with open(data_path(path), encoding=encoding, newline="") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        say(f"show_raw_line: {type(error).__name__}: {error}")
        return
    if not 2 <= line_number <= len(lines):
        say(f"show_raw_line: {path} has lines 1 to {len(lines)}; line 1 is the header.")
        return
    header = next(csv.reader([lines[0]]))
    values = next(csv.reader([lines[line_number - 1]]))
    say(f"1. Line {line_number} of {path}, as it is in the file (lines = f.read().splitlines() gives it as "
        f"lines[{line_number - 1}], because a list counts from 0):")
    say(f"   {preview_line(lines[line_number - 1], 150)}")
    say(f"2. Split at every comma that is not inside quotes: {len(values)} values for {len(header)} columns.")
    notes = header.index("notes") if "notes" in header else len(header)
    for k, value in enumerate(values):
        column = header[k] if k < len(header) else "(no column left)"
        flag = ""
        if len(values) > len(header):
            if k == notes:
                flag = "   <- only the first part of the note"
            elif notes < k < len(header):
                flag = "   <- the rest of the note, in the wrong column"
            elif k >= len(header):
                flag = "   <- an extra value, with no column left"
        say(f"   {column:>16}: {_shown(value, 60)}{flag}")
    row = next(csv.DictReader([lines[0], lines[line_number - 1]]))
    say("3. What csv.DictReader makes of it (the last keys):")
    tail = [(k, v) for k, v in row.items()][-3:]
    for key, value in tail:
        say(f"   {key!r}: {_shown(value, 60)}")
    if None in row:
        say("   The note had a comma and no quotes around it, so the reader split it in two: every value after it "
            "moved one column, and the last one had no column left, so it went under the key None.")


def _first_with(rows: list[dict], scanner: str) -> dict | None:
    return next((r for r in rows if r.get("scanner") == scanner and len(str(r.get("pixels", "")).split()) == 784
                 and "NaN" not in str(r.get("pixels", ""))), None)


def show_scanners(rows: list[dict]) -> None:
    """One scan from each scanner, the way the naive loader read it (int(float(value)) for every pixel),
    with the first pixel values as they are in the file and the mean pixel."""
    picked = [(s, _first_with(rows, s)) for s in ("A", "B", "C")]
    picked = [(s, r) for s, r in picked if r is not None]
    if not picked:
        say("show_scanners: no rows from scanners A, B or C (run the cell that reads the file first).")
        return
    images, captions = [], []
    for scanner, row in picked:
        tokens = str(row["pixels"]).split()
        middle = tokens[14 * 28 + 10:14 * 28 + 14]           # four pixels from the middle of the image
        image = [int(float(t)) for t in tokens]
        mean = sum(image) / len(image)
        say(f"Scanner {scanner}, {row.get('scan_id', '').strip()} (depot {row.get('depot', '').strip()!r}, label typed "
            f"{row.get('label_typed', '')!r}): four pixels from the middle of the image: {' '.join(middle)} · "
            f"mean pixel as the loader read it {mean:.1f}")
        images.append(image)
        captions.append(f"scanner {scanner}")
    side_by_side(images, captions)


def show_inversion(rows: list[dict]) -> None:
    """The goal of C3, before any code: a scanner B scan as it arrives, the same scan with every pixel
    turned into 255 - pixel, and a scanner A scan for comparison, each with its mean pixel."""
    b, a = _first_with(rows, "B"), _first_with(rows, "A")
    if b is None or a is None or _int_pixels(b) is None or _int_pixels(a) is None:
        say("show_inversion: no clean scanner A and B rows found (run the cell that reads the file first).")
        return
    before, ok = _int_pixels(b), _int_pixels(a)
    after = [255 - p for p in before]
    for what, image in (("scanner B as it arrives", before), ("the same scan, 255 - pixel", after),
                        ("scanner A, for comparison", ok)):
        say(f"  {what:<28} mean pixel {sum(image) / len(image):6.1f}")
    side_by_side([before, after, ok], ["scanner B as it arrives", "255 - pixel", "scanner A"])


# ============================================================================= C1, C2, C3
def _is_word(text: str) -> bool:
    return len(text) > 1 and text.isalpha()


# the kinds of label_typed C1 is about, in the order show_labels prints them (one example of each)
LABEL_KINDS = [
    ("a plain digit", lambda t: len(t) == 1 and t in "0123456789"),
    ("a digit with spaces", lambda t: t != t.strip() and len(t.strip()) == 1 and t.strip() in "0123456789"),
    ("a word", lambda t: _is_word(t) and t.islower()),
    ("a word with a capital", lambda t: _is_word(t) and t.istitle()),
    ("a word in capitals", lambda t: _is_word(t) and t.isupper()),
    ("a word with spaces", lambda t: t != t.strip() and _is_word(t.strip())),
    ("a look-alike letter", lambda t: t.strip().lower() in ("o", "l", "i")),
    ("two digits", lambda t: len(t.strip()) == 2 and t.strip().isdigit()),
    ("a digit and a '?'", lambda t: t.strip().endswith("?")),
    ("a decimal number", lambda t: "." in t and t.replace(".", "", 1).strip().isdigit()),
    ("empty", lambda t: not t.strip()),
]


def _label_examples(rows: list[dict], n: int) -> list[str]:
    """One distinct label_typed of each kind (LABEL_KINDS, file order within a kind), then other unusual
    values in file order, at most n in all."""
    values = list(dict.fromkeys(str(row.get("label_typed", "")) for row in rows))
    chosen: list[str] = []
    for _, test in LABEL_KINDS:
        found = next((v for v in values if v not in chosen and test(v)), None)
        if found is not None:
            chosen.append(found)
    plain = LABEL_KINDS[0][1]
    for value in values:
        if len(chosen) >= n:
            break
        if value not in chosen and not plain(value) and not any(test(value) for _, test in LABEL_KINDS[1:6]):
            chosen.append(value)                       # e.g. 'n/a': none of the kinds above
    return chosen[:n]


def show_labels(clean_label: object, rows: list[dict], n: int = 12) -> None:
    """What YOUR clean_label makes of the export's label_typed values: one example of each kind of mess."""
    message = _needs(clean_label, "clean_label")
    if message:
        say(message)
        return
    note = _passing_note(clean_label, "clean_label")
    if note:
        say(note)
    chosen = _label_examples(rows, n)
    say("label_typed in the export   ->  your clean_label")
    for typed in chosen:
        ok, result = _call(clean_label, (typed,), "clean_label", f"the label {typed!r}")
        if not ok:
            return
        say(f"  {repr(typed):<24}  ->  {result!r}")


def show_pixels(parse_pixels: object, rows: list[dict]) -> None:
    """YOUR parse_pixels on five kinds of pixel text from the export, then the first good image, drawn."""
    message = _needs(parse_pixels, "parse_pixels")
    if message:
        say(message)
        return
    note = _passing_note(parse_pixels, "parse_pixels")
    if note:
        say(note)
    kinds = [("a scanner A row", pick_rows(rows, scanner="A")),
             ("a row with only 783 values", [r for r in rows if len(str(r.get("pixels", "")).split()) == 783][:1]),
             ("a row with 'NaN' in it", pick_rows(rows, contains="NaN")),
             ("a row with a 256 in it", [r for r in rows if " 256 " in f" {r.get('pixels', '')} "][:1]),
             ("a scanner C row (floats)", pick_rows(rows, scanner="C"))]
    first_good = None
    say("Pixel text from the export                   ->  your parse_pixels")
    for what, found in kinds:
        if not found:
            continue
        row = found[0]
        text = str(row.get("pixels", ""))
        tokens = text.split()
        ok, result = _call(parse_pixels, (text,), "parse_pixels", f"{what} ({row.get('scan_id', '').strip()})")
        if not ok:
            return
        if isinstance(result, list):
            kinds_in = " and ".join(sorted({type(v).__name__ for v in result}))
            shown = (f"a list of {len(result)} {kinds_in} values, from {min(result)} to {max(result)}"
                     if result and all(isinstance(v, (int, float)) for v in result) else f"a list: {short(result, 40)}")
            if first_good is None and len(result) == 784 and all(isinstance(v, int) for v in result):
                first_good = (row, result)
        else:
            shown = repr(result) if result is None else f"{short(result, 40)} ({type_name(result)})"
        say(f"  {what:<28} {len(tokens):>3} values  ->  {shown}")
    if first_good is not None:
        row, image = first_good
        say(f"The first good one, {row.get('scan_id', '').strip()} (label typed {row.get('label_typed', '')!r}), drawn:")
        show_digit(image)


def show_fix(fix_inversion: object, rows: list[dict]) -> None:
    """A scanner B image before and after YOUR fix_inversion, and a scanner A image (it must not change)."""
    message = _needs(fix_inversion, "fix_inversion")
    if message:
        say(message)
        return
    note = _passing_note(fix_inversion, "fix_inversion")
    if note:
        say(note)
    for scanner, what in (("B", "a scanner B scan (white paper)"), ("A", "a scanner A scan (already right)")):
        row = next((r for r in rows if r.get("scanner") == scanner and _int_pixels(r) is not None), None)
        if row is None:
            continue
        before = _int_pixels(row)
        ok, after = _call(fix_inversion, (list(before),), "fix_inversion", what)
        if not ok:
            return
        if not (isinstance(after, list) and len(after) == 784 and all(isinstance(p, (int, float)) for p in after)):
            say(f"fix_inversion returned {short(after, 40)} ({type_name(after)}) for {what}; it should return a "
                "list of 784 numbers.")
            return
        mean_before, mean_after = sum(before) / 784, sum(after) / 784
        say(f"{what[0].upper() + what[1:]}, {row.get('scan_id', '').strip()}: mean pixel {mean_before:.0f} before, "
            f"{mean_after:.0f} after your fix_inversion.")
        if scanner == "B" or after != before:
            side_by_side([before, [round(p) for p in after]], ["before", "after fix_inversion"])
        else:
            say("   Unchanged, as it should be: its mean is not above 127.")


# ============================================================================= C4, C5
def show_report(report: object, clean_rows: object = None) -> None:
    """The quality report as a table (one bar per reason), and the columns a clean row keeps."""
    if not isinstance(report, dict) or not all(k in report for k in ("total", "valid", "invalid", "reasons")):
        say(f"show_report: a report is a dict with total, valid, invalid and reasons; this is {short(report, 60)}.")
        return
    total = report["total"] or 0
    if isinstance(total, int) and total and all(isinstance(report[k], int) for k in ("valid", "invalid")):
        say(f"Quality report: {report['valid']:,} of {total:,} rows kept ({report['valid'] / total:.0%}), "
            f"{report['invalid']:,} rejected.")
    else:
        say(f"Quality report: total {report['total']!r}, valid {report['valid']!r}, invalid {report['invalid']!r}.")
        total = total if isinstance(total, int) else 0
    reasons = report.get("reasons") or {}
    if isinstance(reasons, dict) and total:
        width = max([len(str(r)) for r in reasons] + [6])
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1] if isinstance(item[1], int) else 0):
            if isinstance(count, int):
                bar = "#" * max(1, round(40 * count / total)) if count else ""
                say(f"  {str(reason).ljust(width)}  {count:>5}  {bar}")
    if isinstance(clean_rows, list) and clean_rows and isinstance(clean_rows[0], dict):
        first = clean_rows[0]
        say(f"A clean row keeps {len(first)} columns: {', '.join(str(k) for k in first)}.")


def _same_report(a: dict, b: dict) -> bool:
    """Two reports are the same when their numbers are, ignoring reasons with a count of 0."""
    def norm(r: dict) -> tuple:
        reasons = r.get("reasons") if isinstance(r.get("reasons"), dict) else {}
        return (r.get("total"), r.get("valid"), r.get("invalid"), {k: v for k, v in reasons.items() if v})
    return norm(a) == norm(b)


def _expected_report(rows: list) -> dict | None:
    """The expected report of the whole export (from the generator's facts, the baseline of check("C4")),
    if `rows` are the good rows of data/parcel_scans_raw.csv; else None."""
    try:
        facts = _truth()["raw"]
    except (OSError, KeyError, ValueError):
        return None
    counts = facts.get("reasons", {})
    total = sum(counts.values())
    if not isinstance(rows, list) or len(rows) != total:
        return None
    return {"total": total, "valid": counts["valid"], "invalid": total - counts["valid"],
            "reasons": {r: counts[r] for r in REASONS if counts.get(r)}}


def _first_difference(mine: dict, expected: dict) -> str:
    """'bad label 26 (expected 27)': the first number that differs between two reports."""
    mine_reasons = mine.get("reasons") if isinstance(mine.get("reasons"), dict) else {}
    for reason in REASONS + [r for r in mine_reasons if r not in REASONS]:
        if mine_reasons.get(reason, 0) != expected["reasons"].get(reason, 0):
            return f"{reason} {mine_reasons.get(reason, 0)!r} (expected {expected['reasons'].get(reason, 0)!r})"
    for key in ("valid", "invalid", "total"):
        if mine.get(key) != expected.get(key):
            return f"{key} {mine.get(key)!r} (expected {expected.get(key)!r})"
    return "the same numbers"


def show_mine(clean_scans: object, rows: list[dict], pipeline_report: dict | None = None):
    """Run YOUR clean_scans on the rows, show its report, and compare it with the expected numbers of the
    export (the same baseline as check("C4")) and with run_pipeline's report (which used YOUR cleaners).
    Returns (clean_rows, report), or None if clean_scans isn't ready or fails."""
    message = _needs(clean_scans, "clean_scans")
    if message:
        say(message)
        return None
    note = _passing_note(clean_scans, "clean_scans")
    if note:
        say(note)
    ok, result = _call(clean_scans, ([dict(r) for r in rows],), "clean_scans", "the rows of the export")
    if not ok:
        return None
    if not (isinstance(result, tuple) and len(result) == 2):
        say(f"clean_scans returned {short(result, 50)} ({type_name(result)}); it should return a pair: "
            "(clean_rows, report).")
        return None
    my_rows, my_report = result
    show_report(my_report, my_rows)
    if not isinstance(my_report, dict):
        return result
    expected = _expected_report(rows)
    same_pipeline = (_same_report(my_report, pipeline_report)
                     if isinstance(pipeline_report, dict) and pipeline_report.get("total") else None)
    if expected is not None:
        if _same_report(my_report, expected):
            say("The expected numbers for this export: yes, every count matches.")
        elif same_pipeline:
            say(f"The expected numbers for this export: no, {_first_difference(my_report, expected)}. Your "
                f"clean_scans agrees with run_pipeline in {FINALE}, which uses the same cleaners, so the "
                "difference is in one of them: run check(\"C1\", clean_label) and check(\"C2\", parse_pixels), "
                "then check(\"C4\", clean_scans).")
        else:
            say(f"The expected numbers for this export: no, {_first_difference(my_report, expected)}. Run "
                "check(\"C4\", clean_scans): it shows the first row your rules decide differently.")
    elif same_pipeline is not None:
        say("The same numbers as run_pipeline in Putting It All Together: " + ("yes." if same_pipeline else
            "no. Compare the two reports reason by reason, and run check(\"C4\", clean_scans)."))
    return result


def save_and_show(save_outputs: object, clean_rows: list, report: dict, out_dir: str = "output") -> None:
    """Call YOUR save_outputs on the clean rows and report, then show the two files it wrote."""
    message = _needs(save_outputs, "save_outputs")
    if message:
        say(message)
        return
    if not clean_rows:
        say("There are no clean rows to save yet: Putting It All Together needs C1, C2 and C3 first.")
        return
    note = _passing_note(save_outputs, "save_outputs")
    if note:
        say(note)
    ok, _ = _call(save_outputs, (clean_rows, report, out_dir), "save_outputs", f"the clean rows (out_dir {out_dir!r})")
    if ok:
        show_outputs(out_dir)


def show_outputs(out_dir: str = "output") -> None:
    """Show what save_outputs wrote: the first lines of scans_clean.csv and the whole quality_report.json."""
    folder = Path(out_dir)
    csv_file, json_file = folder / "scans_clean.csv", folder / "quality_report.json"
    if not csv_file.is_file() and not json_file.is_file():
        say(f"There is nothing in {folder}/ yet: run save_outputs first (C5).")
        return
    if csv_file.is_file():
        data = csv_file.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            say(f"{csv_file} is not UTF-8 text: open it with encoding=\"utf-8\" when you write it.")
            text = data.decode("latin-1")
        lines = text.splitlines()
        say(f"{csv_file}: {len(data):,} bytes, {len(lines)} lines (a header and {max(0, len(lines) - 1)} rows). "
            "The first 3 lines, pixels shortened:")
        for line in lines[:3]:
            say("  " + preview_line(line, 150))
        if b"\r\r\n" in data:
            say("  Note: some lines end with \\r\\r\\n, so Excel shows an empty row after each row: open the file "
                "with newline=\"\" when you write it.")
    if json_file.is_file():
        text = json_file.read_text(encoding="utf-8", errors="replace")
        say(f"{json_file}:", *text.rstrip("\n").split("\n")[:40])
        try:
            json.loads(text)
        except ValueError as error:
            say(f"  Note: this is not valid JSON ({error}).")


# ============================================================================= BLD1
def depot_table(rows: list[dict], clean_depot: object, aliases: dict, clean_rows: list[dict] | None = None) -> None:
    """Per depot (as YOUR clean_depot reads it): rows of the export, the scanner they came from, and how many
    rows clean_rows kept (counted in clean_rows itself, so a duplicate copy is never counted as kept and the
    column adds up to len(clean_rows))."""
    message = _needs(clean_depot, "clean_depot")
    if message:
        say(message)
        return
    table: dict[str, dict] = {}
    for row in rows:
        ok, depot = _call(clean_depot, (row.get("depot", ""), aliases), "clean_depot", f"the depot {row.get('depot')!r}")
        if not ok:
            return
        name = depot if isinstance(depot, str) else "(not recognised)"
        entry = table.setdefault(name, {"rows": 0, "kept": 0, "scanners": {}})
        entry["rows"] += 1
        scanner = row.get("scanner", "?")
        entry["scanners"][scanner] = entry["scanners"].get(scanner, 0) + 1
    for row in clean_rows or []:
        if not isinstance(row, dict):
            continue
        ok, depot = _call(clean_depot, (str(row.get("depot", "")), aliases), "clean_depot",
                          f"the depot {row.get('depot')!r}")
        if not ok:
            return
        name = depot if isinstance(depot, str) else "(not recognised)"
        table.setdefault(name, {"rows": 0, "kept": 0, "scanners": {}})["kept"] += 1
    say(f"{'depot':<18}{'rows':>6}{'kept by the CORE':>18}   scanners")
    for name in sorted(table, key=lambda k: (k.startswith("("), k)):
        entry = table[name]
        scanners = ", ".join(f"{s}: {c}" for s, c in sorted(entry["scanners"].items()))
        share = f" ({entry['kept'] / entry['rows']:.0%})" if entry["rows"] else ""
        kept = f"{entry['kept']}{share}" if clean_rows is not None else "-"
        say(f"{name:<18}{entry['rows']:>6}{kept:>18}   {scanners}")


# ============================================================================= SCI2
def find_suspects(suspects: object, clean_rows: list[dict]) -> list | None:
    """YOUR suspects(clean_rows). If check("SCI2") has just run this same function on the same rows (same
    ids, labels and pixels), its answer is reused instead of waiting a second time. None if it fails."""
    message = _needs(suspects, "suspects")
    if message:
        say(message)
        return None
    from .cases.lab3 import LAST_SUSPECTS, suspects_key

    last = LAST_SUSPECTS.get("run")
    if last is not None and last[0] is getattr(suspects, "__code__", None) and last[1] == suspects_key(clean_rows):
        say("(Reusing the answer of your suspects from check(\"SCI2\"): same function, same rows.)")
        return list(last[2])
    ok, found = _call(suspects, ([dict(r) for r in clean_rows],), "suspects", "the clean rows")
    return found if ok else None


def show_suspects(clean_rows: list[dict], scan_ids: list, n: int = 4) -> None:
    """Draw the first n suspects: the label someone typed next to the image, for a human to decide."""
    if not isinstance(scan_ids, (list, tuple, set)) or not scan_ids:
        say("No suspects to show (your function returned an empty list or no list).")
        return
    by_id = {str(r.get("scan_id", "")).strip().upper(): r for r in clean_rows if isinstance(r, dict)}
    shown = 0
    for scan_id in list(scan_ids):
        row = by_id.get(str(scan_id).strip().upper())
        if row is None or not isinstance(row.get("pixels"), list):
            continue
        say(f"{row['scan_id']}: typed label {row.get('label')!r}. Would you keep it, relabel it, or drop it?")
        show_digit(row["pixels"])
        shown += 1
        if shown == n:
            break
    if not shown:
        say("None of these scan ids is in clean_rows.")
