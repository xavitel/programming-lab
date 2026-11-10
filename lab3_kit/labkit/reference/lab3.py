"""Reference code for Lab 3 · The Scanner Mess.

Clear, commented versions of the worked example, every CORE task and every OPTIONAL task of Lab 3,
written in plain style. Read them to catch up on the lab, to compare with your own code, or borrow
one with rescue() when you are stuck:

    clean_label = rescue("C1")

The pattern behind every cleaner in this lab: normalise -> map -> validate -> a clean value, or None
for "I can't trust this value". None never crashes the pipeline and never pretends to be a real value;
the pipeline decides what to do with it (here: reject the row and record the reason).

The data contract (the same in every lab): an image is a list of 784 ints from 0 (paper) to 255
(ink), row by row from the top-left corner; a label is an int from 0 to 9.
"""
from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path

# ============================================================================= provided constants
# The words people typed instead of a digit (C1). Keys are lower case: look them up after .lower().
LABEL_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
# Letters that look like a digit (C1): the letter o for 0, a lower-case L or a capital i for 1.
LOOK_ALIKES = {"o": 0, "l": 1, "i": 1}
# The columns a clean row keeps (C4, C5): everything else is dropped (data minimisation: the model needs
# only the label and the pixels; operator_email and notes hold personal data, and it never needs them).
CLEAN_COLUMNS = ["scan_id", "depot", "label", "pixels"]


# ============================================================================= W · worked example
def read_scans(path: str) -> tuple[list[dict[str, str]], list[int]]:
    """Read a scanner export: return (good rows, line numbers of malformed rows).

    A malformed row has MORE values than the header, because a comma inside a note was not quoted.
    csv.DictReader puts the extra values under the key None, so `None in row` finds those rows.
    Line 1 of the file is the header, so the first data row is line 2.
    """
    rows = []
    bad_lines = []
    with open(path, encoding="utf-8", newline="") as f:      # utf-8: the export's encoding; newline="": csv's rule
        reader = csv.DictReader(f)                            # one dict per row, keys from the header line
        for line_number, row in enumerate(reader, start=2):
            if None in row:                                   # an unquoted comma: the row has an extra value
                bad_lines.append(line_number)
            else:
                rows.append(row)
    return rows, bad_lines


# ============================================================================= C1
def clean_label(raw: str) -> int | None:
    """Return the digit a person typed (an int 0-9), or None if you can't trust it.

    Normalise (strip spaces, lower case), then map: one digit -> int, a word -> LABEL_WORDS,
    a look-alike letter -> LOOK_ALIKES. Anything else ("77", "7?", "", "n/a") -> None.

    >>> clean_label(" Seven ")
    7
    >>> clean_label("77") is None
    True
    """
    text = raw.strip().lower()                 # " SEVEN " -> "seven"
    if len(text) == 1 and text in "0123456789":
        return int(text)                       # one digit: safe to convert
    if text in LABEL_WORDS:
        return LABEL_WORDS[text]               # "seven" -> 7
    if text in LOOK_ALIKES:
        return LOOK_ALIKES[text]               # "o" -> 0, "l" or "i" -> 1
    return None                                # anything else: we can't trust it


# ============================================================================= C2
def parse_pixels(raw: str) -> list[int] | None:
    """Turn the pixels column into 784 ints from 0 to 255, or None if anything is wrong.

    Wrong = not exactly 784 values, a value int() can't read ("NaN", or a float such as "0.4627"),
    or a value outside 0-255.
    """
    tokens = raw.split()                       # split on any whitespace; extra spaces don't matter
    if len(tokens) != 784:
        return None                            # truncated (783) or too long
    pixels = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError:                     # "NaN", "0.4627", "": not a whole number
            return None
        if value < 0 or value > 255:
            return None                        # 256 or -1: not a pixel
        pixels.append(value)
    return pixels


# ============================================================================= C3
def fix_inversion(pixels: list[int]) -> list[int]:
    """Return the image with paper 0 and ink 255: if the mean pixel is above 127 (white paper),
    every pixel becomes 255 - pixel. Always a NEW list: the input stays as it was.

    >>> fix_inversion([255, 255, 0, 245])
    [0, 0, 255, 10]
    """
    mean = sum(pixels) / len(pixels)
    if mean > 127:                             # mostly white: scanner B's inverted images
        return [255 - p for p in pixels]
    return list(pixels)                        # a copy, so the caller's list is never shared


# ============================================================================= C4
def clean_scans(rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
    """Clean every row with the rules below, in this order; return (clean_rows, report).

    1. the normalised id (strip, upper) was seen before    -> reject: "duplicate"
       (the id is added to `seen` right after this test, whatever happens next)
    2. clean_label(label_typed) is None                    -> reject: "bad label"
    3. parse_pixels(pixels) is None                        -> reject: "bad pixels"
    4. otherwise keep ONLY scan_id, depot, label and pixels (fixed by fix_inversion)

    A row is rejected for the FIRST rule it breaks, so each row has exactly one reason. Duplicates come
    first: a second copy is the same scan again, whatever its label says, so it is counted once, as a copy.

    The report: {"total": rows read, "valid": rows kept, "invalid": rows rejected,
                 "reasons": {reason: how many rows}}.
    """
    clean_rows = []
    reasons = {}
    seen = set()
    for row in rows:
        reason = None
        scan_id = row["scan_id"].strip().upper()
        if scan_id in seen:
            reason = "duplicate"                    # the same scan again: counted once, as a copy
        else:
            seen.add(scan_id)                       # added before the next tests, whatever they say
            label = clean_label(row["label_typed"])
            pixels = parse_pixels(row["pixels"])
            if label is None:                       # `is None`: a label 0 is a real label
                reason = "bad label"
            elif pixels is None:
                reason = "bad pixels"
            else:
                clean_rows.append({"scan_id": scan_id,          # a NEW dict: operator_email and notes
                                   "depot": row["depot"].strip(),   # are not copied (data minimisation)
                                   "label": label,
                                   "pixels": fix_inversion(pixels)})
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
    report = {"total": len(rows), "valid": len(clean_rows), "invalid": len(rows) - len(clean_rows),
              "reasons": reasons}
    return clean_rows, report


# ============================================================================= C5
def save_outputs(clean_rows: list[dict], report: dict, out_dir: str = "output") -> None:
    """Write out_dir/scans_clean.csv (scan_id, depot, label, pixels space-joined) and
    out_dir/quality_report.json (indented, UTF-8 text kept readable). Creates out_dir if needed."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "scans_clean.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLEAN_COLUMNS)
        writer.writeheader()
        for row in clean_rows:
            writer.writerow({"scan_id": row["scan_id"], "depot": row["depot"], "label": row["label"],
                             "pixels": " ".join(str(p) for p in row["pixels"])})   # 784 ints -> one text field
    with open(folder / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)      # ensure_ascii=False keeps "í" as "í"


# ============================================================================= OPTIONAL · ENG1
def clean_postcode(raw: str, valid: set[str]) -> str | None:
    """Return a clean Madrid postcode such as "28013", or None.

    Remove spaces and dots, drop an "E-" or "Madrid" prefix (any case), then the rest must be one of
    the postcodes Garabato delivers to (`valid`). No regular expression needed: `text in valid` already
    compares the whole text. A regex is for FINDING a postcode inside longer text (the gauntlet).
    """
    text = raw.strip().replace(" ", "").replace(".", "")    # " Madrid 28.013" -> "Madrid28013"
    if text.upper().startswith("E-"):
        text = text[2:]
    elif text.upper().startswith("MADRID"):
        text = text[len("MADRID"):]
    if text in valid:                                          # a set: one instant lookup
        return text
    return None


# ============================================================================= OPTIONAL · SCI1
def rescale_floats(raw: str) -> list[int] | None:
    """Scanner C writes each pixel as a fraction from 0.0 to 1.0: turn exactly 784 of them into
    ints from 0 to 255 with round(v * 255), or return None if anything is wrong."""
    tokens = raw.split()
    if len(tokens) != 784:
        return None
    pixels = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            return None
        if not 0.0 <= value <= 1.0:            # also False for nan and inf
            return None
        pixels.append(round(value * 255))
    return pixels


# ============================================================================= OPTIONAL · BLD1
def clean_depot(raw: str, aliases: dict[str, str]) -> str | None:
    """Return the official depot name ("Chamartín", "Retiro" or "Vallecas"), or None.

    Strip spaces, repair mojibake ("ChamartÃ­n" is UTF-8 read as Latin-1), normalise to NFC (one
    code point per accented letter), casefold, then look the result up in `aliases`
    ({casefolded alias: official name}, from data/depots_reference.csv).
    """
    text = raw.strip()
    try:
        text = text.encode("latin-1").decode("utf-8")       # undo UTF-8 read as Latin-1
    except UnicodeError:                                     # not mojibake (or not Latin-1 at all)
        pass                                                 # keep the text as it was
    text = unicodedata.normalize("NFC", text)                # "i" + combining accent -> "í"
    return aliases.get(text.casefold())


def load_depot_aliases(path: str = "data/depots_reference.csv") -> dict[str, str]:
    """Read data/depots_reference.csv (written by an old Windows program: cp1252, not UTF-8) into
    {casefolded alias: official depot name}."""
    aliases = {}
    with open(path, encoding="cp1252", newline="") as f:
        for row in csv.DictReader(f):
            aliases[row["alias"].strip().casefold()] = row["depot"].strip()
    return aliases


# ============================================================================= helpers for checks and views
def decisions(rows: list[dict[str, str]]) -> list[str]:
    """What clean_scans does with each row, in order: "valid" or the reason it was rejected."""
    out = []
    seen = set()
    for row in rows:
        scan_id = row["scan_id"].strip().upper()
        if scan_id in seen:
            out.append("duplicate")
        else:
            seen.add(scan_id)
            if clean_label(row["label_typed"]) is None:
                out.append("bad label")
            elif parse_pixels(row["pixels"]) is None:
                out.append("bad pixels")
            else:
                out.append("valid")
    return out
