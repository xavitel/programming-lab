"""Checks for Lab 3 · The Scanner Mess (blueprint §5.3, plan_labs2-5.md §2).

What each task is checked against. Expected values that come from the reference are stored in
cases/lab3_data.json by generators/gen_cases.py; the facts about the data files (the expected decision
of every fixture row, the reason counts of the whole export, and the answer keys of SCI2 and the Challenge,
stored as scan_key(id) hashes) are in cases/lab3_scans.json, written by generators/make_scanner_mess.py from
the defects it injected. So this file holds no solution code: only inputs, toy answers, and the
misconception detectors ("Likely cause").

Tasks, in notebook order: W, C1, C2, C3, C4 (CORE), C5, ENG1, SCI1, BLD1 (OPTIONAL, checked like CORE),
ENG2, SCI2, BLD2, SCI3 (ADVANCED: sanity checks only, no rescue).

C4 is checked in three steps: toy rows (one rule at a time), then the 30-row fixture row by row (the first
row that differs is shown), then the whole export reason by reason (a table of expected vs yours).
"""
from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import sys
import textwrap
import types
from pathlib import Path

from ..check import PASS, is_stub, run_task
from ..data import load_valid_postcodes
from ..fuzz import fuzz_strings
from ..scans import FIXTURE, RAW, REASONS, data_path, preview_line, read_rows, run_pipeline, scan_key
from ..spec import Case, Detector, Task, custom
from ..text import short, type_name

FINALE = "Putting It All Together"      # the name of the lab's final CORE run (check_all, start)
CLEAN_COLUMNS = ["scan_id", "depot", "label", "pixels"]
NEXT_HINT_2 = "open Hint 2 below (the plan), fix your code, run the cell with def again, then this check."

_CACHE: dict = {}


class _Said:
    """An expectation written in words: the engine prints it as it is, without the quotes of a str."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


# ============================================================================= data for the cases
def _facts() -> dict:
    if "facts" not in _CACHE:
        with open(Path(__file__).with_name("lab3_scans.json"), encoding="utf-8") as f:
            _CACHE["facts"] = json.load(f)
    return _CACHE["facts"]


def _rows(name: str) -> list[dict]:
    """The good rows of a data file (read once), as fresh copies: a check never shares them."""
    if name not in _CACHE:
        _CACHE[name] = read_rows(name)[0]
    return [dict(row) for row in _CACHE[name]]


def _row_where(name: str, test) -> dict:
    return next(row for row in _rows(name) if test(row))


def _int_pixels(row: dict) -> list[int]:
    return [int(v) for v in row["pixels"].split()]


def _valid_postcodes() -> set[str]:
    if "valid" not in _CACHE:
        _CACHE["valid"] = load_valid_postcodes("data/valid_postcodes.txt")
    return set(_CACHE["valid"])


def _aliases() -> dict[str, str]:
    """data/depots_reference.csv, read the way the BLD1 card says (cp1252)."""
    if "aliases" not in _CACHE:
        with open(data_path("depots_reference.csv"), encoding="cp1252", newline="") as f:
            _CACHE["aliases"] = {r["alias"].strip().casefold(): r["depot"].strip() for r in csv.DictReader(f)}
    return dict(_CACHE["aliases"])


def _pixel_text(values) -> str:
    return " ".join(str(v) for v in values)


# 784-pixel toy images: a vertical bar of ink (paper 0, ink 200), and the same image inverted
_BAR = [200 if 10 <= i % 28 <= 13 and 4 <= i // 28 <= 23 else 0 for i in range(784)]
PIX_OK = _pixel_text(_BAR)
PIX_INVERTED = _pixel_text(255 - p for p in _BAR)
PIX_SHORT = _pixel_text(_BAR[:783])
PIX_NAN = _pixel_text(["NaN" if i == 400 else p for i, p in enumerate(_BAR)])


def _int(got: object, expected: object) -> bool:
    """An int equal to the expected one (True/False and floats don't count); None must be None."""
    if expected is None:
        return got is None
    return type(got) is int and got == expected


INT_OR_NONE = custom(_int, "")


def _no_call(t) -> bool:
    """A string method used without (): raw.strip().lower instead of raw.strip().lower()."""
    return t.raised(TypeError, AttributeError) and "builtin_function_or_method" in str(t.error)


NO_CALL = Detector(_no_call, "you wrote a method without its (): text.lower is the method itself, text.lower() "
                             "calls it. Add the brackets to every .strip, .lower, .upper and .split, e.g. "
                             "raw.strip().lower().")


def _test(ctx, ok: bool, *, title: str, case: str, expected: str, got: str, cause: str | None = None,
          passed: str | None = None) -> bool:
    """One check from a condition. On NOT YET it prints `got`; on PASS, `passed` (default: the expected text)."""
    shown = (expected if passed is None else passed) if ok else got
    return ctx.require(ok, case=case, title=title, expected=expected, got=shown, cause=cause)


# ============================================================================= W · read_scans
def _path(name: str) -> str:
    """data/<name> as the student's code should open it (in the kit: a relative path)."""
    return str(data_path(name))


def _read_count(fn):
    rows, bad = fn(_path(FIXTURE))
    return len(rows)


def _read_bad(fn):
    return fn(_path(FIXTURE))[1]


def _read_keys(fn):
    rows, _ = fn(_path(FIXTURE))
    return sorted({key for row in rows for key in row}, key=str)


def _read_raw(fn):
    rows, bad = fn(_path(RAW))
    return len(rows), len(bad)


W = Task(
    "W", "read_scans", title="Read the scanner export",
    practise="with open(..., encoding, newline), csv.DictReader and enumerate (Theory S5–S6)",
    expected_label="Expected",
    cases=[
        Case(call=_read_count, key="fixture-rows", highlight=True,
             title="The fixture: every well-formed row comes back as a dict",
             show="len(read_scans(\"data/fixtures/scans_fixture.csv\")[0])", expected_label="Expected rows"),
        Case(call=_read_bad, key="fixture-bad", highlight=True,
             title="The fixture: the row with an unquoted comma is reported by its line number",
             show="read_scans(\"data/fixtures/scans_fixture.csv\")[1]", expected_label="Expected lines"),
        Case(call=_read_keys, key="fixture-keys",
             title="Every good row has exactly the 10 columns of the header (no None key)",
             show="the keys of the rows read from data/fixtures/scans_fixture.csv"),
        Case(call=_read_raw, key="raw", highlight=True,
             title="The whole export: (good rows, malformed rows)",
             show="read_scans(\"data/parcel_scans_raw.csv\"): (len(rows), len(bad_lines))"),
    ],
    detectors=[
        Detector(lambda t: t.raised(UnicodeDecodeError),
                 "the file is UTF-8, but it was opened with another encoding. Pass encoding=\"utf-8\" to open()."),
        Detector(lambda t: isinstance(t.got, list) and None in t.got,
                 "some rows still have a None key: skip the rows where `None in row`, and keep their line numbers."),
    ],
)


# ============================================================================= C1 · clean_label
def _c1(raw: str, expected, title: str, *, highlight: bool = False) -> Case:
    return Case((raw,), expected, compare=INT_OR_NONE, title=title, highlight=highlight)


def _strip_low(t) -> str:
    return t.args[0].strip().lower() if t.args and isinstance(t.args[0], str) else ""


C1 = Task(
    "C1", "clean_label", title="Which digit did they type?",
    practise="string methods, dict lookup, and None for a value you can't trust (Theory S6)",
    minimum_path=True,
    input_labels=("Typed label",), expected_label="Expected",
    cases=[
        _c1("7", 7, "A clean digit", highlight=True),
        _c1(" 7", 7, "Spaces around a digit: strip them first", highlight=True),
        _c1("3\n", 3, "A digit followed by a line break"),
        _c1("0", 0, "Zero is a real label, not a missing one"),
        _c1("seven", 7, "A word: look it up in LABEL_WORDS", highlight=True),
        _c1("SEVEN", 7, "A word in capitals: lower-case it before the lookup"),
        _c1(" Nine ", 9, "A word with capitals and spaces"),
        _c1("O", 0, "The letter O typed for a zero (LOOK_ALIKES)"),
        _c1("l", 1, "A lower-case L typed for a one (LOOK_ALIKES)"),
        _c1("I", 1, "A capital i typed for a one (LOOK_ALIKES)"),
        _c1("77", None, "Two digits: not a label, so None"),
        _c1("00", None, "Two digits, even 00: None (int(\"00\") would give 0)"),
        _c1("+7", None, "A plus sign and a digit: None (int(\"+7\") would give 7)"),
        _c1("7?", None, "A digit and a question mark: None"),
        _c1("7.0", None, "A decimal number: None (int(\"7.0\") would crash)"),
        _c1("-1", None, "A negative number: None"),
        _c1("", None, "Empty text: None"),
        _c1("   ", None, "Only spaces: None"),
        _c1("12", None, "Two different digits: \"12\" is a piece of \"0123456789\", still not a label"),
        _c1("n/a", None, "n/a: None"),
        _c1("seventy", None, "A word that is not a digit: None"),
    ],
    detectors=[
        NO_CALL,
        Detector(lambda t: t.raised(ValueError) and t.args and isinstance(t.args[0], str) and not t.args[0].strip(),
                 "the empty text reached int(). If you tested `text in \"0123456789\"`: on a string, `in` looks "
                 "for a PIECE of it, so '' (and '01') are 'in' \"0123456789\". Test the length too: "
                 "len(text) == 1 and text in \"0123456789\".", next_step=NEXT_HINT_2),
        Detector(lambda t: t.raised(ValueError) and "invalid literal for int()" in str(t.error),
                 "int() crashes on text that isn't one whole number, such as '7?' or 'seven'. Look at the text "
                 "first: convert it only when it is one digit, look words up in the dicts, and return None for "
                 "anything else.", next_step=NEXT_HINT_2),
        Detector(lambda t: t.raised(KeyError),
                 "LABEL_WORDS[text] raises KeyError when the text is not in the dict. Test `text in LABEL_WORDS` "
                 "first (or use LABEL_WORDS.get(text)), and return None when it is nowhere."),
        Detector(lambda t: isinstance(t.got, str),
                 "you returned text (a str). Return the number itself: int(text), LABEL_WORDS[text] or "
                 "LOOK_ALIKES[text]."),
        Detector(lambda t: t.expected is None and type(t.got) is int and len(_strip_low(t)) > 1
                 and _strip_low(t) in "0123456789",
                 "on a string, `in` looks for a PIECE of it, so '12' (like '') is 'in' \"0123456789\" although it "
                 "is two digits. Test the length too: len(text) == 1 and text in \"0123456789\".",
                 next_step=NEXT_HINT_2),
        Detector(lambda t: t.expected is None and type(t.got) is int and len(_strip_low(t)) > 1
                 and _strip_low(t).lstrip("+-").replace(".", "").isdigit(),
                 "a label is ONE digit from 0 to 9, so text such as '77', '00', '+7', '7.0' or '-1' must give "
                 "None, even when int() can read it. Check the text first: len(text) == 1 and text in "
                 "\"0123456789\"."),
        Detector(lambda t: t.expected is not None and t.got is None and t.args[0] != t.args[0].strip()
                 and t.args[0].strip().lower() == t.args[0].strip(),
                 "the text has spaces (or a line break) around it: call .strip() before you look at it."),
        Detector(lambda t: t.expected is not None and t.got is None and _strip_low(t) in ("o", "l", "i"),
                 "the letters O, l and I are look-alikes for 0 and 1: look the lower-case text up in "
                 "LOOK_ALIKES too."),
        Detector(lambda t: t.expected is not None and t.got is None and t.args[0].strip() != _strip_low(t),
                 "capitals: 'SEVEN' is not a key of LABEL_WORDS, but 'seven' is. Call .lower() before the lookup."),
        Detector(lambda t: t.expected == 0 and t.got is None,
                 "0 is a real label. If you wrote `if label:` or `or None`, remember that 0 counts as False: "
                 "test `is None` instead."),
        Detector(lambda t: isinstance(t.got, bool),
                 "you returned True or False. Return the digit itself (an int), or None."),
    ],
    returns=(int, type(None)),
)


# ============================================================================= C2 · parse_pixels
def _c2(text_fn, expected, title: str, show_text: str, *, key: str | None = None, highlight: bool = False) -> Case:
    return Case(lambda: (text_fn(),), expected, key=key, title=title, highlight=highlight,
                show_inputs={"Pixel text": show_text})


def _real_a() -> str:
    return _row_where(RAW, lambda r: r["scanner"] == "A" and len(r["pixels"].split()) == 784
                      and r["pixels"].replace(" ", "").isdigit())["pixels"]


def _real_c() -> str:
    return _row_where(RAW, lambda r: r["scanner"] == "C")["pixels"]


C2 = Task(
    "C2", "parse_pixels", title="784 numbers, or nothing",
    practise="split, int() inside try/except, and checking length and range (Theory S2, S6)",
    minimum_path=True,
    expected_label="Expected",
    cases=[
        Case(lambda: (PIX_OK,), key="clean", highlight=True, title="A clean scan: 784 whole numbers separated by spaces",
             show_inputs={"Pixel text": "'0 0 0 ... 200 200 ... 0' (784 values)"}),
        Case(lambda: ("  " + PIX_OK.replace(" ", "  ") + " \n",), key="spaces",
             title="Extra spaces and a line break between the numbers are fine",
             show_inputs={"Pixel text": "'  0  0  0 ... 0 \\n' (784 values, two spaces between them)"}),
        _c2(lambda: PIX_SHORT, None, "Only 783 numbers (a truncated export): None", "'0 0 0 ... 0' (783 values)",
            highlight=True),
        _c2(lambda: PIX_OK + " 0", None, "785 numbers: None", "'0 0 0 ... 0 0' (785 values)"),
        _c2(lambda: PIX_NAN, None, "A 'NaN' among the numbers: None (int('NaN') raises ValueError)",
            "'0 0 ... NaN ... 0' (784 values)", highlight=True),
        _c2(lambda: _pixel_text([256] + _BAR[1:]), None, "256 is out of range: None", "'256 0 0 ... 0' (784 values)"),
        _c2(lambda: _pixel_text(_BAR[:-1] + [-1]), None, "-1 is out of range: None", "'0 0 ... 0 -1' (784 values)"),
        Case(lambda: (_pixel_text([0, 255] * 392),), key="limits", title="0 and 255 are the limits: both allowed",
             show_inputs={"Pixel text": "'0 255 0 255 ...' (784 values)"}),
        _c2(lambda: _pixel_text(["0.4627"] * 784), None,
            "Scanner C floats such as 0.4627: None in the CORE (SCI1 rescues them)", "'0.4627 0.4627 ...' (784 values)"),
        _c2(lambda: "", None, "Empty text: None", "''"),
        Case(lambda: (_real_a(),), key="real-a", title="A real scanner A row of the export",
             show_inputs={"Pixel text": "the pixels column of the first scanner A row of data/parcel_scans_raw.csv"}),
        _c2(_real_c, None, "A real scanner C row of the export: None",
            "the pixels column of the first scanner C row of data/parcel_scans_raw.csv"),
    ],
    detectors=[
        NO_CALL,
        Detector(lambda t: t.raised(ValueError),
                 "int() raised ValueError on a value that isn't a whole number ('NaN', '0.4627'). Put the int() "
                 "call inside try/except ValueError, and return None in the except block.", next_step=NEXT_HINT_2),
        Detector(lambda t: isinstance(t.got, list) and isinstance(t.expected, list) and len(t.got) == 1
                 and t.args and isinstance(t.args[0], str) and len(t.args[0].split()) > 1,
                 "your function returned after the first value: `return pixels` is inside the loop. Un-indent it "
                 "so it runs once, after the loop has converted all 784 values."),
        Detector(lambda t: isinstance(t.got, list) and t.expected is None and t.args and isinstance(t.args[0], str)
                 and "NaN" in t.args[0],
                 "int('NaN') raised ValueError and your function still returned a list. The except block must "
                 "`return None`: with `pass` (or `continue`) the loop just carries on, skipping the value or "
                 "repeating the previous one."),
        Detector(lambda t: isinstance(t.got, list) and t.got and all(isinstance(v, str) for v in t.got),
                 "you returned the numbers as text ('0', '200'): split() gives strings. Convert each one with "
                 "int(), and return the list of ints."),
        Detector(lambda t: isinstance(t.got, list) and t.expected is None and t.got and
                 all(isinstance(v, int) for v in t.got) and "0.4627" in str(t.args[0])[:20],
                 "int(float(x)) turns 0.4627 into 0, so a scanner C image becomes blank paper. In the CORE, "
                 "int(token) must fail on a float: return None (SCI1 rescues these rows properly)."),
        Detector(lambda t: isinstance(t.got, list) and t.expected is None and len(t.got) != 784,
                 "check the number of values: exactly 784, otherwise return None (a truncated scan would "
                 "otherwise slip through)."),
        Detector(lambda t: isinstance(t.got, list) and t.expected is None and t.got
                 and any(isinstance(v, int) and not 0 <= v <= 255 for v in t.got),
                 "check the range: every pixel must be from 0 to 255, so 256 or -1 means None."),
        Detector(lambda t: t.got is None and isinstance(t.expected, list),
                 "a good scan gave None. Use split() with no argument (split(\" \") makes empty strings out of "
                 "double spaces), and compare the length with 784."),
        Detector(lambda t: isinstance(t.got, (tuple, map)) or inspect.isgenerator(t.got),
                 "return a list (put the values in a list, or wrap them with list(...))."),
    ],
    returns=(list, type(None)),
)


# ============================================================================= C3 · fix_inversion
def _real_b() -> list[int]:
    row = _row_where(RAW, lambda r: r["scanner"] == "B" and len(r["pixels"].split()) == 784
                     and r["pixels"].replace(" ", "").isdigit() and max(map(int, r["pixels"].split())) <= 255)
    return _int_pixels(row)


def _real_a_pixels() -> list[int]:
    return [int(v) for v in _real_a().split()]


def _new_list(fn) -> bool:
    image = [0, 0, 255, 10]
    return fn(image) is not image


C3 = Task(
    "C3", "fix_inversion", title="White paper, black ink?",
    practise="a mean, a condition and a list comprehension (Theory S2, S4)",
    minimum_path=True,
    input_labels=("Pixels",), expected_label="Expected",
    cases=[
        Case(([0, 0, 255, 10],), [0, 0, 255, 10], title="A normal scan (mean 66): the same pixels, unchanged",
             highlight=True),
        Case(([255, 255, 0, 245],), [0, 0, 255, 10],
             title="An inverted scan (mean 189): every pixel becomes 255 - pixel", highlight=True),
        Case(([127, 127],), [127, 127], title="Mean exactly 127: not above 127, so unchanged"),
        Case(([127, 128],), [128, 127], title="Mean 127.5: above 127, so inverted"),
        Case(lambda: (_real_b(),), key="real-b", title="A real scanner B image of the export: inverted",
             show="fix_inversion(the pixels of the first good scanner B row)"),
        Case(lambda: (_real_a_pixels(),), key="real-a", title="A real scanner A image: unchanged",
             show="fix_inversion(the pixels of the first good scanner A row)"),
        Case(call=_new_list, expected=True, key="new-list", expected_label="Expected",
             title="It returns a NEW list, even when nothing changes",
             show="fix_inversion(image) is not image"),
    ],
    detectors=[
        Detector(lambda t: t.raised(TypeError) and "'int' and 'list'" in str(t.error),
                 "inside the comprehension, use the loop variable: [255 - p for p in pixels] (p is one pixel; "
                 "pixels is the whole list)."),
        Detector(lambda t: t.got is False and t.expected is True,
                 "you returned the very list you were given. Return a copy instead: list(pixels) or pixels[:]. "
                 "Otherwise a later change to one of them changes both."),
        Detector(lambda t: isinstance(t.got, list) and t.args and t.args[0] == [127, 127] and t.got == [128, 128],
                 "a mean of exactly 127 is not above 127: use > 127, not >=."),
        Detector(lambda t: isinstance(t.got, list) and t.args and isinstance(t.args[0], list)
                 and t.expected == t.args[0] and t.got == [255 - p for p in t.args[0]],
                 "you inverted a normal scan. Invert only when the mean pixel is above 127 (white paper)."),
        Detector(lambda t: isinstance(t.got, list) and t.args and t.args[0] == [127, 128] and t.got == [127, 128],
                 "the mean of [127, 128] is 127.5. If you used //, it became 127: compute the mean with "
                 "sum(pixels) / len(pixels)."),
        Detector(lambda t: isinstance(t.got, list) and t.args and isinstance(t.args[0], list) and t.got == t.args[0]
                 and t.expected != t.args[0],
                 "the inverted scan came back unchanged. Compute the mean (sum / len); when it is above 127, "
                 "return [255 - p for p in pixels]."),
        Detector(lambda t: t.got is None and t.error is None,
                 "your function returned None. Does every path end with a return? (Also when nothing changes: "
                 "return a copy.)"),
    ],
    returns=list,
)


# ============================================================================= C4 · clean_scans
def _row(scan_id: str, label: str = "7", pixels: str = PIX_OK, depot: str = "Retiro") -> dict:
    return {"scan_id": scan_id, "depot": depot, "scanner": "A", "scanned_at": "2026-11-02T14:05",
            "label_typed": label, "postcode": "28017", "box": "5", "pixels": pixels,
            "notes": "", "operator_email": "op.amber17@example.org"}


def _report_text(report: object) -> str:
    if not isinstance(report, dict):
        return f"{short(report, 40)} ({type_name(report)}, not a dict)"
    parts = []
    for key in ("total", "valid", "invalid"):
        parts.append(f"{key} {report[key]!r}" if key in report else f"no '{key}'")
    reasons = report.get("reasons")
    if not isinstance(reasons, dict):
        parts.append(f"reasons {short(reasons, 30)}" if "reasons" in report else "no 'reasons'")
    else:
        for reason in REASONS + [r for r in reasons if r not in REASONS]:
            count = reasons.get(reason, 0)
            if count:
                parts.append(f"{reason} {count!r}")
    return " · ".join(parts)


def _kept_text(row: object) -> str:
    if not isinstance(row, dict):
        return f"{short(row, 30)} (not a dict)"
    bits = [f"label {row.get('label')!r}"]
    pixels = row.get("pixels")
    if isinstance(pixels, list) and pixels and all(isinstance(p, int) for p in pixels):
        bits.append(f"mean pixel {sum(pixels) / len(pixels):.0f}")
    else:
        bits.append(f"pixels {short(pixels, 24)}")
    depot = row.get("depot")
    if isinstance(depot, str) and depot != depot.strip():
        bits.append(f"depot {depot!r}")
    extra = [str(k) for k in row if k not in CLEAN_COLUMNS]
    missing = [k for k in CLEAN_COLUMNS if k not in row]
    if extra:
        bits.append("extra columns: " + ", ".join(extra))
    if missing:
        bits.append("missing columns: " + ", ".join(missing))
    return f"{row.get('scan_id', '?')!s} ({', '.join(bits)})"


def _outcome(result: object) -> str:
    """One line that says what clean_scans did: the rows it kept, then its report."""
    if not (isinstance(result, tuple) and len(result) == 2):
        return f"returned {short(result, 40)} ({type_name(result)}); expected a pair (clean_rows, report)"
    clean_rows, report = result
    if not isinstance(clean_rows, list):
        return f"clean_rows is {type_name(clean_rows)}, not a list | {_report_text(report)}"
    kept = "; ".join(_kept_text(r) for r in clean_rows) or "nothing"
    return f"kept {kept} | {_report_text(report)}"


def _describe(rows: list[dict]) -> str:
    """The toy rows of a C4 test, in a few words each."""
    parts = []
    for r in rows:
        bits = [f"label_typed {r['label_typed']!r}"]
        if r["pixels"] is PIX_INVERTED:
            bits.append("an inverted image")
        elif r["pixels"] is not PIX_OK:
            bits.append("broken pixels")
        if r["depot"] != "Retiro":
            bits.append(f"depot {r['depot']!r}")
        scan_id = r["scan_id"] if r["scan_id"] == r["scan_id"].strip() else repr(r["scan_id"])
        parts.append(f"{scan_id} ({', '.join(bits)})")
    return "clean_scans([" + "; ".join(parts) + "])"


# The toy tests of C4, one rule at a time: (key, title, rows, what clean_scans should give, written by _outcome).
# The first three are the ones a PASS prints.
C4_TOYS = [
    ("one", "One clean row: kept, with only scan_id, depot, label and pixels", [_row("PX-000001")],
     "kept PX-000001 (label 7, mean pixel 20) | total 1 · valid 1 · invalid 0"),
    ("duplicate", "The same scan twice: the second copy is a 'duplicate'", [_row("PX-000007"), _row("PX-000007")],
     "kept PX-000007 (label 7, mean pixel 20) | total 2 · valid 1 · invalid 1 · duplicate 1"),
    ("bad-label", "A label clean_label can't read: rejected as 'bad label'",
     [_row("PX-000001"), _row("PX-000002", label="7?")],
     "kept PX-000001 (label 7, mean pixel 20) | total 2 · valid 1 · invalid 1 · bad label 1"),
    ("two-bad", "Two rows with a bad label: invalid counts ROWS, so 2",
     [_row("PX-000017", label="77"), _row("PX-000018", label="n/a"), _row("PX-000019")],
     "kept PX-000019 (label 7, mean pixel 20) | total 3 · valid 1 · invalid 2 · bad label 2"),
    ("duplicate-variant", "The same id written differently: still a 'duplicate'",
     [_row("PX-000008"), _row(" px-000008 ")],
     "kept PX-000008 (label 7, mean pixel 20) | total 2 · valid 1 · invalid 1 · duplicate 1"),
    ("first-rule", "A bad label AND bad pixels: the first rule that fails wins ('bad label')",
     [_row("PX-000009", label="77", pixels=PIX_SHORT)], "kept nothing | total 1 · valid 0 · invalid 1 · bad label 1"),
    ("dup-of-bad", "A copy of a row rejected for its label is still a 'duplicate'",
     [_row("PX-000010", label="77"), _row("PX-000010")],
     "kept nothing | total 2 · valid 0 · invalid 2 · duplicate 1 · bad label 1"),
    ("dup-bad-copy", "A copy with a bad label is a 'duplicate': rule 1 comes before rule 2",
     [_row("PX-000016"), _row("PX-000016", label="77")],
     "kept PX-000016 (label 7, mean pixel 20) | total 2 · valid 1 · invalid 1 · duplicate 1"),
    ("messy", "A messy label and spaces around the depot: label 7, depot 'Retiro'",
     [_row("PX-000011", label=" Seven ", depot=" Retiro ")],
     "kept PX-000011 (label 7, mean pixel 20) | total 1 · valid 1 · invalid 0"),
    ("zero", "Label 0 is a real label: kept", [_row("PX-000012", label="0")],
     "kept PX-000012 (label 0, mean pixel 20) | total 1 · valid 1 · invalid 0"),
    ("inverted", "An inverted scan: kept, and fixed by fix_inversion", [_row("PX-000013", pixels=PIX_INVERTED)],
     "kept PX-000013 (label 7, mean pixel 20) | total 1 · valid 1 · invalid 0"),
    ("bad-pixels", "Broken pixels: rejected as 'bad pixels'", [_row("PX-000014", pixels=PIX_NAN)],
     "kept nothing | total 1 · valid 0 · invalid 1 · bad pixels 1"),
    ("empty", "No rows: nothing kept, and a report of zeros", [], "kept nothing | total 0 · valid 0 · invalid 0"),
]


def _c4_toy_cause(key: str, result: object, got: str, expected: str, n_rows: int) -> str:
    """The likely cause of a wrong answer on one toy test (most specific first)."""
    pair = isinstance(result, tuple) and len(result) == 2
    rows_out = result[0] if pair and isinstance(result[0], list) else None
    report = result[1] if pair else None
    if rows_out is None:
        return "clean_scans must return a pair: the list of clean rows and the report (return clean_rows, report)."
    kept = [r for r in rows_out if isinstance(r, dict)]
    if len(kept) != len(rows_out):
        return "each clean row is a dict with the keys scan_id, depot, label and pixels."
    if any(set(r) - set(CLEAN_COLUMNS) for r in kept):
        return ("a clean row keeps ONLY scan_id, depot, label and pixels. Build a NEW dict with those four keys: "
                "operator_email and notes are personal or unneeded data, so they stay behind (data minimisation).")
    if any(c not in r for r in kept for c in CLEAN_COLUMNS):
        return "a clean row needs all four keys: scan_id, depot, label and pixels."
    if not isinstance(report, dict) or any(k not in report for k in ("total", "valid", "invalid", "reasons")):
        return "the report is a dict with four keys: total, valid, invalid and reasons (a dict {reason: count})."
    unknown = [r for r in report["reasons"] if r not in REASONS] if isinstance(report["reasons"], dict) else []
    if unknown:
        return ("use exactly these reason texts: 'duplicate', 'bad label', 'bad pixels' "
                f"(you wrote {unknown[0]!r}): the report counts rows by these exact keys.")
    if any(isinstance(r.get("label"), str) for r in kept):
        return "the label you kept is text. Keep what clean_label returns (an int)."
    if "duplicate" in got and "duplicate" not in expected:
        return ("a row this call has never seen was called a duplicate. Is `seen` created outside clean_scans? "
                "Create it inside, at the start, so every call starts with an empty set.")
    if key in ("bad-label", "two-bad") and "bad label" not in got:
        return ("a row whose label clean_label can't read was kept. Reject it when clean_label returns None "
                "(rule 2), and test it with `is None`.")
    if key == "duplicate-variant" and "duplicate" not in got:
        return ("' px-000008 ' is the same scan as 'PX-000008'. Normalise the id with .strip().upper() before you "
                "test it and before you add it to seen.")
    if key == "dup-bad-copy" and "bad label" in got:
        return ("rule 1 (duplicate) comes BEFORE rule 2 (label): test `scan_id in seen` before you look at the "
                "label. The first rule that fails gives the reason.")
    if key == "dup-of-bad" and "duplicate" not in got:
        return ("add the id to seen right after the duplicate test, whatever happens next: a copy of a row with a "
                "bad label is a duplicate too.")
    if key == "duplicate" and "duplicate" not in got:
        return ("the second copy of a scan was kept. Keep a set `seen` of normalised ids: reject an id that is "
                "already in it, otherwise add it.")
    if key == "first-rule" and "bad pixels" in got:
        return ("the first rule that fails gives the reason: test the label (rule 2) before the pixels (rule 3), "
                "in one if / elif chain.")
    if key == "zero" and "kept nothing" in got:
        return "the row with label 0 was rejected. `if not label:` treats 0 as missing: test `label is None`."
    if key == "inverted" and "mean pixel 235" in got:
        return "the inverted scan was kept as it is. Pass the pixels through fix_inversion (C3) before you keep them."
    if key == "messy" and "depot" in got.split("|")[0]:
        return "strip the depot too: row[\"depot\"].strip()."
    if key == "bad-pixels" and "kept PX" in got:
        return "a row with broken pixels was kept. Reject it when parse_pixels returns None (rule 3)."
    if key in ("first-rule", "dup-of-bad") and "kept PX" in got:
        return "a row whose label clean_label can't read was kept. Reject it when clean_label returns None (rule 2)."
    reasons = report["reasons"] if isinstance(report["reasons"], dict) else {}
    if report.get("invalid") == len(reasons) and report.get("invalid") != sum(reasons.values()):
        return ("invalid counts the rejected ROWS, not the different reasons: len(reasons) is the number of "
                "reasons. Use invalid = total - valid.")
    if report.get("total") != n_rows:
        return "total is the number of rows clean_scans was given: len(rows)."
    if report.get("valid") != len(rows_out) or report.get("invalid") != report.get("total", 0) - report.get("valid", 0):
        return "the report's numbers don't add up: valid = len(clean_rows) and invalid = total - valid."
    return "compare your rules, and their order, with the table on the card."


def _counts(result: object) -> dict | None:
    """{"valid": n, reason: n} from what clean_scans returned, or None if it isn't (list, report dict)."""
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list)
            and isinstance(result[1], dict) and isinstance(result[1].get("reasons"), dict)):
        return None
    counts = {"valid": len(result[0])}
    for reason, count in result[1]["reasons"].items():
        counts[str(reason)] = count
    return counts


_DECISION_CAUSES = [
    (lambda want, got, row: want == "duplicate" and got == "valid" and row["scan_id"] != row["scan_id"].strip().upper(),
     "this id is written differently from the first copy ({id!r}). Normalise it with .strip().upper() before "
     "you look it up in seen (and before you add it)."),
    (lambda want, got, row: want == "duplicate" and got == "bad label",
     "this row is a copy of an earlier row. Rule 1 (duplicate) comes BEFORE rule 2 (label): test `scan_id in "
     "seen` before you look at the label, and add the id to seen right after that test, for every row that "
     "reaches it (not only the rows you keep)."),
    (lambda want, got, row: want == "duplicate" and got == "valid",
     "this row is a second copy of a scan you already met: keep a set of the ids you have seen, and reject an id "
     "that is already in it."),
    (lambda want, got, row: want == "bad label",
     "label_typed is {label!r}: clean_label should return None for it. Run check(\"C1\", clean_label)."),
    (lambda want, got, row: want == "valid" and got == "bad label",
     "label_typed is {label!r}, which clean_label can read. Do you call clean_label (C1) on row[\"label_typed\"]?"),
    (lambda want, got, row: want == "bad pixels" and got == "valid",
     "the pixels of this row are broken; parse_pixels returns None for them. Test it with `is None`."),
    (lambda want, got, row: want == "valid" and got == "bad pixels",
     "the pixels of this row are fine. Do you call parse_pixels (C2) on row[\"pixels\"], and test its result "
     "with `is None`?"),
]


def _decision_cause(want: str, got: str, row: dict) -> str:
    for test, message in _DECISION_CAUSES:
        try:
            if test(want, got, row):
                return message.format(id=row["scan_id"], label=row["label_typed"])
        except Exception:
            continue
    return "compare your rules, and their order, with the card: duplicate, label, pixels."


def _numbers_problem(result: object, n_rows: int) -> tuple[str, str] | None:
    """(what is wrong, likely cause) if the report's total / valid / invalid don't fit the rows, else None."""
    clean_rows, report = result
    reasons = report.get("reasons") if isinstance(report.get("reasons"), dict) else {}
    total, valid, invalid = report.get("total"), report.get("valid"), report.get("invalid")
    if total != n_rows:
        return f"total {total!r} (expected {n_rows:,})", "total is the number of rows clean_scans was given: len(rows)."
    if valid != len(clean_rows):
        return (f"valid {valid!r}, but clean_rows has {len(clean_rows):,} rows",
                "valid is the number of rows you kept: len(clean_rows).")
    if invalid != n_rows - len(clean_rows) or invalid != sum(v for v in reasons.values() if isinstance(v, int)):
        cause = ("invalid counts the rejected ROWS, not the different reasons (len(reasons)). "
                 if invalid == len(reasons) else "")
        return (f"invalid {invalid!r} (expected {n_rows - len(clean_rows):,})",
                cause + "invalid = total - valid, and it equals the sum of the reason counts.")
    return None


def _c4_check(fn, ctx) -> None:
    """Toy rows (one rule at a time), then the fixture row by row, then the whole export reason by reason."""
    if fn is None:
        return
    for key, title, toy_rows, expected in C4_TOYS:
        call = _describe(toy_rows)
        attempt = ctx.call(fn, [dict(r) for r in toy_rows], show=call)     # tracked: are the rows left unchanged?
        if attempt.error is not None or attempt.interrupted:
            attempt.title, attempt.inputs = title, [("Call:", call)]
            ctx.expect(attempt, _Said(expected))
            return
        got = _outcome(attempt.got)
        if not _test(ctx, got == expected, title=title, case=call, expected=expected, got=got,
                     cause=_c4_toy_cause(key, attempt.got, got, expected, len(toy_rows))):
            return
    # the rows read_scans gave must stay as they were (the engine also checks this, in less detail)
    variant = next(rows for key, _, rows, _ in C4_TOYS if key == "duplicate-variant")
    toy_rows = [dict(r) for r in variant] + [_row("PX-000015", label=" Seven ", depot=" Retiro ")]
    before = [dict(r) for r in toy_rows]
    attempt = ctx.call(fn, toy_rows, track=False, show=_describe(before))
    changed = [(i, key) for i, (old, new) in enumerate(zip(before, toy_rows)) for key in old
               if new.get(key, "(removed)") != old[key]]
    where = ""
    if changed:
        i, key = changed[0]
        where = f"row {i + 1}'s {key!r} was {before[i][key]!r}, after the call it is {toy_rows[i].get(key, '(removed)')!r}"
    if not _test(ctx, attempt.error is None and not changed, title="The rows you were given stay as they were",
                 case=_describe(before), expected="every row unchanged", got=where or "your function raised an error",
                 cause="build the cleaned values in new variables and put them in a NEW dict; change nothing in "
                       "`row` itself (no row[...] = ..., no row.pop(...)): the caller may still need the original."):
        return
    facts = _facts()
    # 1. the fixture, row by row: the k-th call cleans the first k rows; whichever count grew is that row's fate
    rows = _rows(FIXTURE)
    truth = facts["fixture"]["decisions"]
    title = "The 30-row fixture (data/fixtures/scans_fixture.csv), row by row: every kind of defect"
    previous: dict = {}
    for k in range(1, len(rows) + 1):
        attempt = ctx.call(fn, [dict(r) for r in rows[:k]], track=False, show=f"clean_scans(fixture_rows[:{k}])")
        if attempt.error is not None or attempt.interrupted:
            attempt.title = title
            ctx.expect(attempt, _Said("a decision for every row"))
            return
        counts = _counts(attempt.got)
        if counts is None:
            ctx.require(False, case=attempt.show, title=title, expected="(clean_rows, report)",
                        got=short(attempt.got, 60), cause="clean_scans must return a pair: the list of clean rows "
                        "and the report dict (with a 'reasons' dict).")
            return
        grew = [name for name in set(counts) | set(previous) if counts.get(name, 0) > previous.get(name, 0)]
        got = grew[0] if len(grew) == 1 else ("nothing" if not grew else " and ".join(sorted(grew)))
        sid, line, want = truth[k - 1]
        if got != want:
            where = f"line {line} ({sid.strip()})"
            got_text = "kept (valid)" if got == "valid" else (f"rejected, '{got}'" if got in REASONS else got)
            want_text = "kept (valid)" if want == "valid" else f"rejected, '{want}'"
            ctx.require(False, case="clean_scans(fixture_rows)", title=title, expected=f"{where}: {want_text}",
                        got=f"{where}: {got_text}", cause=_decision_cause(want, got, rows[k - 1]))
            return
        previous = counts
    problem = _numbers_problem(attempt.got, len(rows))
    if problem:
        ctx.require(False, case="clean_scans(fixture_rows)", title=title, expected=f"total {len(rows)} · valid = "
                    "len(clean_rows) · invalid = total - valid", got=problem[0], cause=problem[1])
        return
    ctx.require(True, case="clean_scans(fixture_rows)", title=title,
                expected=f"the right decision for all {len(rows)} rows", got=f"all {len(rows)} right")
    # 2. the whole export: one call, then expected vs yours, reason by reason
    rows = _rows(RAW)
    title = f"The whole export (data/parcel_scans_raw.csv, {len(rows):,} rows), reason by reason"
    attempt = ctx.call(fn, rows, track=False, show="clean_scans(rows)")
    if attempt.error is not None or attempt.interrupted:
        attempt.title = title
        ctx.expect(attempt, _Said("a report"))
        return
    counts = _counts(attempt.got)
    expected = facts["raw"]["reasons"]
    if counts is None:
        ctx.require(False, case=attempt.show, title=title, expected="(clean_rows, report)", got=short(attempt.got, 60))
        return
    names = ["valid"] + REASONS + [r for r in counts if r not in expected]
    differs = [n for n in names if counts.get(n, 0) != expected.get(n, 0)]
    if differs:
        ctx.note("The whole export, reason by reason:")
        ctx.note(f"   {'':<22}{'expected':>9}{'yours':>9}")
        for name in names:
            mark = "   <- differs" if name in differs else ""
            ctx.note(f"   {name:<22}{expected.get(name, 0):>9,}{counts.get(name, 0):>9,}{mark}")
    problem = None if differs else _numbers_problem(attempt.got, len(rows))
    if problem:
        _test(ctx, False, case="clean_scans(rows)", title=title, expected=f"total {len(rows):,} · valid "
              f"{expected.get('valid', 0):,} · invalid {len(rows) - expected.get('valid', 0):,}", got=problem[0],
              cause=problem[1])
        return
    first = differs[0] if differs else None
    cause = None
    if differs:
        cause = _cleaner_cause(fn, ctx, rows, differs) or (
            "the toy rows and the fixture pass, so look at what the export has that they don't: find the reason "
            "that differs in the table below.")
    _test(ctx, not differs, case="clean_scans(rows)", title=title,
                expected=f"{first} {expected.get(first, 0):,}" if first else "the expected count for every reason",
                got=f"{first} {counts.get(first, 0):,}" if first else "the same counts", cause=cause)


def _cleaner_of(fn, name: str):
    """The student's cleaner `name` as clean_scans sees it (the notebook's globals), or None."""
    found = getattr(fn, "__globals__", {}).get(name)
    return found if callable(found) else None


def _cleaner_cause(fn, ctx, rows: list[dict], differs: list[str]) -> str | None:
    """When the export's counts differ: the first export value on which the student's clean_label or
    parse_pixels disagrees with the reference (the usual reason when the toys and the fixture pass)."""
    def disagree(mine, ref, values):
        for value in values:
            want = ref(value)
            try:
                got = mine(value)
            except Exception as error:                        # noqa: BLE001 (a student's cleaner)
                return value, f"raises {type(error).__name__}", want
            if got != want or type(got) is not type(want):
                return value, got, want
        return None

    if any(name in differs for name in ("bad label", "valid")):
        mine = _cleaner_of(fn, "clean_label")
        if mine is not None:
            values = list(dict.fromkeys(r["label_typed"] for r in rows))
            found = disagree(mine, ctx.reference("clean_label"), values)
            if found:
                value, got, want = found
                got_text = got if isinstance(got, str) and got.startswith("raises") else f"gives {got!r}"
                return (f"the difference starts in your clean_label: for the export's label_typed {value!r} "
                        f"it {got_text}, and it should give {want!r} (a label is ONE digit, a word, "
                        "or a look-alike). Fix clean_label (C1) and run its cell again, then this check.")
    if any(name in differs for name in ("bad pixels", "valid")):
        mine = _cleaner_of(fn, "parse_pixels")
        if mine is not None:
            found = disagree(mine, ctx.reference("parse_pixels"), list(dict.fromkeys(r["pixels"] for r in rows)))
            if found:
                value, got, want = found
                row = next(r for r in rows if r["pixels"] == value)
                got_text = got if isinstance(got, str) and got.startswith("raises") else (
                    "returns None" if got is None else f"returns {short(got, 30)}")
                want_text = "None" if want is None else "a list of 784 ints"
                return (f"the difference starts in your parse_pixels: it {got_text} for the pixels of "
                        f"{row['scan_id'].strip()} (scanner {row['scanner']}), and it should return {want_text}. "
                        "Fix parse_pixels (C2) and run its cell again, then this check.")
    return None


C4 = Task(
    "C4", "clean_scans", title="One pipeline, rules in order",
    practise="composing your cleaners, a set for 'seen before?', and a report dict (Theory S3, S5)",
    depends_on=("C1", "C2", "C3"),
    custom=_c4_check,
    detectors=[                                   # for errors; wrong answers get their cause from _c4_toy_cause
        NO_CALL,
        Detector(lambda t: t.raised(KeyError) and str(t.error).strip("'\"") in REASONS,
                 "the first time a reason appears it isn't in the dict yet, so reasons[reason] += 1 fails. Count "
                 "with reasons[reason] = reasons.get(reason, 0) + 1 (Theory S3, counting with a dict)."),
        Detector(lambda t: t.raised(KeyError) and "None" not in str(t.error),
                 "a key isn't there. The rows from read_scans use the header's names: 'scan_id', "
                 "'label_typed', 'pixels', 'depot'. The report needs 'total', 'valid', 'invalid' and 'reasons'."),
        Detector(lambda t: t.raised(TypeError) and "unhashable" in str(t.error),
                 "only simple values such as text go into a set: add the normalised scan_id to seen, not the row."),
        Detector(lambda t: t.raised(AttributeError) and "'set' object has no attribute 'append'" in str(t.error),
                 "a set grows with .add(), not .append()."),
        Detector(lambda t: t.raised(AttributeError) and "'NoneType'" in str(t.error),
                 "a cleaner returned None and the next line used it anyway. Test `is None` first, and reject the "
                 "row with its reason."),
    ],
)


# ============================================================================= C5 · save_outputs
_C5_ROWS = [{"scan_id": "PX-000001", "depot": "Chamartín", "label": 7, "pixels": _BAR},
            {"scan_id": "PX-000002", "depot": "Retiro", "label": 0, "pixels": [0] * 784},
            {"scan_id": "PX-000004", "depot": "Vallecas", "label": 3, "pixels": [255 - p for p in _BAR]}]
_C5_REPORT = {"total": 4, "valid": 3, "invalid": 1, "reasons": {"duplicate": 1},
              "depots": {"Chamartín": 1, "Retiro": 1, "Vallecas": 1}}


def _pixels_text_cause(got: object) -> str:
    """Why the pixels column of scans_clean.csv is not '0 0 0 ...' (784 numbers separated by single spaces)."""
    text = got if isinstance(got, str) else ""
    if text.startswith("["):
        return ("a list written to a CSV becomes its repr '[0, 0, ...]'. Write "
                "\" \".join(str(p) for p in row[\"pixels\"]) instead.")
    separators = sorted({c for c in text if not c.isdigit() and c != " "})
    if separators or "  " in text:
        used = f"you used {separators[0]!r}" if separators else "you used more than one space"
        return ("the pixels must be separated by single spaces (\" \".join(...)), the format parse_pixels reads "
                f"back; {used}.")
    return "write the 784 pixels of the row, each as text, separated by single spaces: \" \".join(...)."


def _c5_files(fn, ctx) -> None:
    """Call save_outputs in a scratch folder, read the files back, and check them one property at a time."""
    rows = [dict(r, pixels=list(r["pixels"])) for r in _C5_ROWS]
    report = json.loads(json.dumps(_C5_REPORT))
    with ctx.temp_dir(chdir=True) as folder:
        show = "save_outputs(clean_rows, report, \"results\")"
        attempt = ctx.call(fn, rows, report, "results", show=show)
        if attempt.error is not None or attempt.interrupted:
            attempt.title = "Writing the two files into a new folder, results/"
            ctx.expect(attempt, _Said("two files in results/"))
            return
        out = folder / "results"
        csv_file, json_file = out / "scans_clean.csv", out / "quality_report.json"
        title = "It creates the folder and both files"
        if not _test(ctx, csv_file.is_file() and json_file.is_file(), case=show, title=title,
                           expected="results/scans_clean.csv and results/quality_report.json",
                           got=", ".join(sorted(p.name for p in out.iterdir())) if out.is_dir() else "no folder results/",
                           cause="create the folder with Path(out_dir).mkdir(parents=True, exist_ok=True), then "
                                 "open both files inside it: Path(out_dir) / \"scans_clean.csv\"."):
            return
        data = csv_file.read_bytes()
        try:
            text = data.decode("utf-8")
            utf8 = True
        except UnicodeDecodeError:
            text, utf8 = data.decode("latin-1"), False
        if not _test(ctx, utf8, case=show, title="scans_clean.csv is UTF-8 text ('Chamartín' survives)",
                           expected="UTF-8", got="another encoding (probably your system's default)",
                           cause="pass encoding=\"utf-8\" to open() when you write the file."):
            return
        if not _test(ctx, b"\r\r\n" not in data, case=show, title="No empty line after each row",
                           expected="one line per row", got="lines ending in \\r\\r\\n (an empty row after each row)",
                           cause="open the CSV file with newline=\"\": the csv module writes its own line ends."):
            return
        lines = list(csv.reader(text.splitlines()))
        header = lines[0] if lines else []
        if not _test(ctx, header == CLEAN_COLUMNS, case=show, title="The header is scan_id,depot,label,pixels",
                           expected=",".join(CLEAN_COLUMNS),
                           got=preview_line(text.splitlines()[0])[:90] if lines else "(an empty file)",
                           cause="use csv.DictWriter(f, fieldnames=[\"scan_id\", \"depot\", \"label\", \"pixels\"]) "
                                 "and call writer.writeheader() once, before the rows."):
            return
        if not _test(ctx, len(lines) == 1 + len(rows), case=show, title="One line per clean row",
                           expected=f"{len(rows)} rows after the header", got=f"{len(lines) - 1} rows",
                           cause="call writer.writerow(...) once for each clean row."):
            return
        first = dict(zip(header, lines[1]))
        want = " ".join(str(p) for p in _BAR)
        if not _test(ctx, first.get("pixels") == want, case=show,
                           title="The pixels column holds 784 numbers separated by spaces",
                           expected="'0 0 0 ... 200 200 ... 0' (784 numbers)", got=short(first.get("pixels"), 50),
                           cause=_pixels_text_cause(first.get("pixels"))):
            return
        if not _test(ctx, first.get("depot") == "Chamartín" and first.get("label") == "7", case=show,
                           title="The other columns are written as they are",
                           expected="PX-000001, Chamartín, 7", got=f"{first.get('scan_id')}, {first.get('depot')}, "
                           f"{first.get('label')}"):
            return
        raw_json = json_file.read_bytes().decode("utf-8", errors="replace")
        try:
            loaded = json.loads(raw_json)
        except ValueError as error:
            ctx.require(False, case=show, title="quality_report.json is valid JSON", expected="valid JSON",
                        got=f"not JSON ({error})", cause="write it with json.dump(report, f, ...).")
            return
        if not _test(ctx, loaded == _C5_REPORT, case=show, title="quality_report.json holds the report you were given",
                           expected=short(_C5_REPORT, 60), got=short(loaded, 60),
                           cause="json.dump the report itself, unchanged."):
            return
        if not _test(ctx, "Chamartín" in raw_json, case=show, title="Accented names stay readable in the JSON",
                           expected="\"Chamartín\"", got="\"Chamart\\u00edn\"",
                           cause="json.dump escapes every non-ASCII letter by default: pass ensure_ascii=False "
                                 "(and open the file with encoding=\"utf-8\")."):
            return
        indents = [len(line) - len(line.lstrip()) for line in raw_json.splitlines() if line.lstrip().startswith('"')]
        indent = min(indents) if indents else 0
        flat = "one key per line, but not indented" if "\n" in raw_json.strip() else "everything on one line"
        if not _test(ctx, indent > 0, case=show, title="The JSON is indented, one key per line",
                           expected="indented (indent=2)", got=flat, passed="indented, one key per line",
                           cause="pass indent=2 to json.dump, so people can read the report."):
            return
    with ctx.temp_dir(chdir=True) as folder:
        attempt = ctx.call(fn, [dict(r, pixels=list(r["pixels"])) for r in _C5_ROWS], dict(_C5_REPORT),
                           show="save_outputs(clean_rows, report)", track=False)
        if attempt.error is not None:
            attempt.title = "Without out_dir, the files go to output/"
            ctx.expect(attempt, _Said("two files in output/"))
            return
        _test(ctx, (folder / "output" / "scans_clean.csv").is_file()
                    and (folder / "output" / "quality_report.json").is_file(),
                    case="save_outputs(clean_rows, report)", title="Without out_dir, the files go to output/",
                    expected="output/scans_clean.csv and output/quality_report.json",
                    got="not there", cause="give out_dir the default value \"output\" in the def line.")


C5 = Task(
    "C5", "save_outputs", title="Save the clean data and the report",
    practise="csv.DictWriter, json.dump and pathlib (Theory S5)",
    optional=True, not_needed_later=True,     # OPTIONAL since 10 Nov 2026: the CORE ends with C4
    custom=_c5_files,
    detectors=[
        Detector(lambda t: t.raised(ValueError) and "fields not in fieldnames" in str(t.error),
                 "a row has a key that is not in your fieldnames: either fieldnames misses one of scan_id, depot, "
                 "label, pixels, or the rows you write still have a column you meant to drop."),
        Detector(lambda t: t.raised(TypeError) and "JSON serializable" in str(t.error),
                 "json.dump can write dicts, lists, str, int, float, bool and None, nothing else: dump the report "
                 "dict itself."),
        Detector(lambda t: t.raised(FileNotFoundError),
                 "the folder doesn't exist yet: create it first with Path(out_dir).mkdir(parents=True, exist_ok=True)."),
        Detector(lambda t: t.raised(AttributeError) and "object has no attribute 'write'" in str(t.error),
                 "json.dump and csv.DictWriter write into an open FILE, not a path: with open(path, \"w\", "
                 "encoding=\"utf-8\") as f: then json.dump(report, f, ...) or csv.DictWriter(f, ...)."),
    ],
)


# ============================================================================= OPTIONAL · ENG1 clean_postcode
def _pc(raw: str, expected, title: str, *, highlight: bool = False) -> Case:
    return Case(lambda: (raw, _valid_postcodes()), expected, title=title, highlight=highlight,
                key=f"pc-{raw!r}", show_inputs={"Raw postcode": repr(raw),
                                                "Valid postcodes": "the 55 of data/valid_postcodes.txt (28001 to 28055)"})


ENG1 = Task(
    "ENG1", "clean_postcode", title="Postcodes and the gauntlet",
    practise="string methods and a set lookup (Theory S3, S6)",
    expected_label="Expected",
    cases=[
        _pc("28013", "28013", "A clean postcode", highlight=True),
        _pc(" 28013 ", "28013", "Spaces around it"),
        _pc("28.013", "28013", "A dot inside: remove dots", highlight=True),
        _pc("28 013", "28013", "A space inside: remove spaces"),
        _pc("E-28013", "28013", "The country prefix E-"),
        _pc("e-28013", "28013", "The prefix in lower case"),
        _pc("Madrid 28013", "28013", "A Madrid prefix", highlight=True),
        _pc("MADRID 28055", "28055", "The prefix in capitals, and the highest postcode"),
        _pc("2813", None, "A digit missing: None"),
        _pc("08013", None, "A Barcelona postcode: None"),
        _pc("28099", None, "28 and three digits, but not a postcode Garabato delivers to: None"),
        _pc("280130", None, "Six digits: None"),
        _pc("28O13", None, "A letter O instead of a zero: None"),
        _pc("", None, "Empty: None"),
        _pc("Madrid", None, "Only the prefix: None"),
    ],
    detectors=[
        Detector(lambda t: t.expected is None and t.got == "28099",
                 "28099 looks like a Madrid postcode, but it isn't one Garabato delivers to: also test `text in "
                 "valid`."),
        Detector(lambda t: isinstance(t.got, str) and t.expected is None and len(t.got) != 5,
                 "a postcode has exactly 5 digits: return the cleaned text only if the WHOLE text is in valid "
                 "(`text in valid`); searching for 28 and three digits inside it also accepts a longer number."),
        Detector(lambda t: t.got is None and isinstance(t.expected, str) and "." in t.args[0],
                 "remove the dots too: text.replace(\".\", \"\")."),
        Detector(lambda t: t.got is None and isinstance(t.expected, str) and t.args[0].strip()[:1] in "eEmM",
                 "drop the prefix in any case: compare text.upper().startswith(\"E-\") or "
                 "text.upper().startswith(\"MADRID\"), after removing spaces."),
        Detector(lambda t: isinstance(t.got, int),
                 "return the postcode as text (\"28013\"), not a number: a leading zero would vanish in 08013."),
    ],
    returns=(str, type(None)),
)


# ============================================================================= OPTIONAL · SCI1 rescale_floats
def _floats(values) -> str:
    return " ".join("0.0" if v == 0 else f"{v:.4f}" for v in values)


SCI1 = Task(
    "SCI1", "rescale_floats", title="Rescue scanner C",
    practise="float(), round() and validation, the same pattern as C2 (Theory S2, S6)",
    expected_label="Expected",
    cases=[
        Case(lambda: (_floats([0.0] * 392 + [0.4627] * 392),), key="toy", highlight=True,
             title="784 fractions: each becomes round(v * 255)",
             show_inputs={"Pixel text": "'0.0 0.0 ... 0.4627 0.4627' (784 values)"}),
        Case(lambda: (_floats([1.0, 0.0] * 392),), key="limits", title="0.0 and 1.0 are the limits: 0 and 255",
             show_inputs={"Pixel text": "'1.0000 0.0 1.0000 0.0 ...' (784 values)"}),
        Case(lambda: (_real_c(),), key="real-c", highlight=True, title="A real scanner C row of the export",
             show_inputs={"Pixel text": "the pixels column of the first scanner C row of data/parcel_scans_raw.csv"}),
        Case(lambda: (_floats([0.5] * 783),), None, title="783 values: None",
             show_inputs={"Pixel text": "'0.5000 ...' (783 values)"}),
        Case(lambda: (" ".join(["0.5"] * 783 + ["NaN"]),), None, title="A NaN: None (NaN is not between 0 and 1)",
             show_inputs={"Pixel text": "'0.5 ... NaN' (784 values)"}),
        Case(lambda: (" ".join(["0.5"] * 783 + ["1.2"]),), None, title="1.2 is above 1: None",
             show_inputs={"Pixel text": "'0.5 ... 1.2' (784 values)"}),
        Case(lambda: (" ".join(["0.5"] * 783 + ["-0.1"]),), None, title="-0.1 is below 0: None",
             show_inputs={"Pixel text": "'0.5 ... -0.1' (784 values)"}),
        Case(lambda: (PIX_OK,), None, title="Whole numbers up to 200 (scanner A): None, they are not fractions",
             show_inputs={"Pixel text": "'0 0 ... 200 ... 0' (784 values)"}),
    ],
    detectors=[
        Detector(lambda t: t.raised(ValueError) and "NaN" in str(t.error),
                 "a NaN got through your range test, then round() failed on it. `v < 0 or v > 1` is False for NaN; "
                 "test the other way round: if not 0.0 <= v <= 1.0, return None."),
        Detector(lambda t: t.raised(ValueError),
                 "float() raised ValueError: put it inside try/except ValueError and return None."),
        Detector(lambda t: isinstance(t.got, list) and t.got and isinstance(t.got[0], float),
                 "return ints from 0 to 255: round(v * 255) gives an int."),
        Detector(lambda t: isinstance(t.got, list) and t.expected is None and "NaN" in str(t.args[0])[-5:],
                 "NaN passes no comparison, but `v < 0 or v > 1` is False for NaN too. Test the range the other "
                 "way round: if not 0.0 <= v <= 1.0, return None."),
        Detector(lambda t: isinstance(t.got, list) and t.expected is None,
                 "every value must be a fraction from 0.0 to 1.0, and there must be exactly 784 of them."),
        Detector(lambda t: isinstance(t.got, list) and isinstance(t.expected, list) and t.got
                 and max(t.got) <= 1,
                 "you kept the fractions. Multiply each one by 255 and round it: round(v * 255)."),
    ],
    returns=(list, type(None)),
)


# ============================================================================= OPTIONAL · BLD1 clean_depot
def _dp(raw: str, expected, title: str, *, highlight: bool = False) -> Case:
    return Case(lambda: (raw, _aliases()), expected, title=title, highlight=highlight, key=f"dp-{raw!r}",
                show_inputs={"Raw depot": repr(raw), "Aliases": "data/depots_reference.csv, read with cp1252"})


BLD1 = Task(
    "BLD1", "clean_depot", title="Depot names",
    practise="Unicode normalisation, encodings and casefold (Theory S6)",
    expected_label="Expected",
    cases=[
        _dp("Chamartín", "Chamartín", "The official name", highlight=True),
        _dp(" Retiro ", "Retiro", "Spaces around it"),
        _dp("CHAMARTÍN", "Chamartín", "Capitals: casefold before the lookup"),
        _dp("vallecas", "Vallecas", "Lower case"),
        _dp("VLC", "Vallecas", "An alias"),
        _dp("chm", "Chamartín", "An alias in lower case"),
        _dp("Chamartín", "Chamartín", "The i and its accent as TWO code points (NFD): normalise to NFC",
            highlight=True),
        _dp("ChamartÃ­n", "Chamartín", "Mojibake: 'ChamartÃ­n' is UTF-8 read as Latin-1", highlight=True),
        _dp("Chamartin", "Chamartín", "Without the accent (an alias in the reference file)"),
        _dp("Getafe", None, "A depot that doesn't exist: None"),
        _dp("", None, "Empty: None"),
    ],
    detectors=[
        Detector(lambda t: t.raised(UnicodeDecodeError, UnicodeEncodeError),
                 "the mojibake repair raises an error on text that isn't mojibake. Put it inside try/except "
                 "UnicodeError and keep the text as it was when it fails."),
        Detector(lambda t: t.got is None and t.args and "́" in t.args[0],
                 "this 'í' is two code points: an i and a combining accent. unicodedata.normalize(\"NFC\", text) "
                 "joins them into one."),
        Detector(lambda t: t.got is None and t.args and "Ã" in t.args[0],
                 "'ChamartÃ­n' is UTF-8 bytes read as Latin-1. Undo it with text.encode(\"latin-1\").decode(\"utf-8\")."),
        Detector(lambda t: t.got is None and isinstance(t.expected, str),
                 "the aliases are casefolded: strip and casefold the text before the lookup (casefold() is "
                 "lower() for every alphabet)."),
        Detector(lambda t: isinstance(t.got, str) and t.got != t.expected,
                 "return the official name, the VALUE of the aliases dict, not the text you looked up."),
    ],
    returns=(str, type(None)),
)


# ============================================================================= ADVANCED · ENG2 cleaners.py (sanity)
def _module_prints(module) -> list[int]:
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return []
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "print"]


def _eng2_module(module, ctx) -> None:
    if not isinstance(module, types.ModuleType):
        ctx.not_started("Import your module first (import cleaners), then run check(\"ENG2\", cleaners).")
        return
    # Check the file as it is saved now. A fresh import also keeps the line numbers right: on IPython 9,
    # autoreload patches functions in place and the tracebacks then name the wrong lines.
    try:
        module = importlib.reload(module)
    except (ImportError, TypeError):
        pass                                          # not a reloadable file module: check it as it is
    except Exception as error:                        # noqa: BLE001 (an error in the student's file)
        where = f" (line {error.lineno})" if isinstance(error, SyntaxError) and error.lineno else ""
        ctx.require(False, case=f"import {module.__name__}", title=f"{module.__name__}.py can be imported",
                    expected="no error", got=f"{type(error).__name__}{where}: {error}",
                    cause=f"Python stops at this error when it reads {module.__name__}.py: fix that line, save the "
                          "file, then run this check again.")
        return
    lab_cases = ctx.lab_cases
    needed = [("clean_label", "C1"), ("parse_pixels", "C2")]
    for name, _ in needed:
        fn = getattr(module, name, None)
        if fn is None or is_stub(fn):
            ctx.not_started(f"cleaners.py has no working {name} yet: paste your function into it, save the file, "
                            "run the import cell above again (importlib.reload), then this check.")
            return
    for name, task_id in needed:
        task = lab_cases.find(task_id) if lab_cases is not None else None
        if task is None:
            continue
        report, _ = run_task(task, getattr(module, name), lab_cases)
        _test(ctx, report.status == PASS, case=f"check(\"{task_id}\", cleaners.{name})",
                    title=f"cleaners.{name} still passes the {task_id} check",
                    expected="PASS", got=f"{report.status} ({report.passed} of {report.total})",
                    cause=f"run check(\"{task_id}\", cleaners.{name}) to see which test fails.")
    functions = [(n, f) for n, f in vars(module).items() if inspect.isfunction(f) and not n.startswith("_")
                 and getattr(f, "__module__", "") == module.__name__]
    missing_doc = [n for n, f in functions if not (f.__doc__ or "").strip()]
    _test(ctx, not missing_doc, case="cleaners.py", title="Every public function has a docstring",
                expected="a docstring in each function", got="no docstring in " + ", ".join(missing_doc),
                cause="a module is read by people who never saw your notebook: one line saying what the function "
                      "returns, and when it returns None.")
    prints = _module_prints(module)
    _test(ctx, not prints, case="cleaners.py", title="No print() inside the module",
                expected="no print()", got=f"print() on line {', '.join(map(str, prints))}",
                cause="a cleaner returns a value or None; printing is the caller's job (Theory S13, module rules).")
    extra = {"clean_postcode": (_valid_postcodes(),), "clean_depot": (_aliases(),)}
    texts = fuzz_strings(200)
    for name, fn in functions:
        if name not in ("clean_label", "parse_pixels", "rescale_floats", "clean_postcode", "clean_depot") or is_stub(fn):
            continue
        for text in texts:
            attempt = ctx.call(fn, text, *extra.get(name, ()), track=False, show=f"cleaners.{name}({short(text, 30)})")
            if attempt.error is not None:
                attempt.title = f"cleaners.{name} never raises, on 200 tricky strings (fuzz_strings(200))"
                ctx.expect(attempt, _Said("a value or None"), next_step=ENG2_NEXT)
                return
        ctx.require(True, case=f"cleaners.{name}(text) for text in fuzz_strings(200)",
                    title=f"cleaners.{name} never raises, on 200 tricky strings", expected="no error", got="no error")


ENG2_NEXT = ("fix it in cleaners.py and save the file, run the import cell above again (it reloads the module), "
             "then this check.")


def _non_ascii_digit(t) -> bool:
    text = t.args[0] if t.args and isinstance(t.args[0], str) else ""
    return any(c.isdigit() and c not in "0123456789" for c in text)


ENG2 = Task(
    "ENG2", "cleaners", kind="custom", title="Module and properties",
    practise="a module of your own, and property tests",
    rescue=False, custom=_eng2_module, quality=False,
    detectors=[
        Detector(lambda t: t.raised(ValueError) and _non_ascii_digit(t),
                 "isdigit() is True for more characters than 0-9: '²' passes it, but int('²') raises ValueError. "
                 "Test `len(text) == 1 and text in \"0123456789\"` instead (or put int() inside try/except ValueError).",
                 next_step=ENG2_NEXT),
        Detector(lambda t: t.raised(ValueError) and "invalid literal for int()" in str(t.error),
                 "int() raised ValueError on a tricky string: put it inside try/except ValueError and return None.",
                 next_step=ENG2_NEXT),
    ],
)


# ============================================================================= ADVANCED · SCI2 suspects (sanity)
LAST_SUSPECTS: dict = {}       # the check's last suspects() run, for labkit.lab3_tools.find_suspects


def suspects_key(rows: list[dict]) -> tuple:
    """What a suspects() answer depends on: every row's id, label and pixel sum."""
    return tuple((str(r.get("scan_id")), r.get("label"), sum(r.get("pixels") or ())) for r in rows)


def _sci2_score(fn, ctx) -> None:
    clean_rows, _ = run_pipeline(_rows(RAW), ctx.reference("clean_label"), ctx.reference("parse_pixels"),
                                 ctx.reference("fix_inversion"), quiet=True)
    ids = {r["scan_id"] for r in clean_rows}
    liar_keys = set(_facts()["liar_keys"])
    liars = {i for i in ids if scan_key(i) in liar_keys}
    rows_key = suspects_key(clean_rows)
    attempt = ctx.call(fn, [dict(r, pixels=list(r["pixels"])) for r in clean_rows], track=False,
                       show="suspects(clean_rows)")
    attempt.title = f"suspects(clean_rows) on the {len(clean_rows):,} clean rows of the export"
    if attempt.error is not None or attempt.interrupted:
        ctx.expect(attempt, _Said("a list of scan ids"))
        return
    got = attempt.got
    if not _test(ctx, isinstance(got, (list, tuple, set)) and all(isinstance(s, str) for s in got),
                       case="suspects(clean_rows)", title="It returns a list of scan ids (text)",
                       expected="['PX-001234', ...]", got=f"{short(got, 40)} ({type_name(got)})",
                       passed=f"{len(got)} scan ids, e.g. {short(list(got)[:2], 40)}"):
        return
    if attempt.error is None and isinstance(got, (list, tuple, set)):
        LAST_SUSPECTS["run"] = (getattr(fn, "__code__", None), rows_key, list(got))   # find_suspects reuses it
    flagged = {s.strip().upper() for s in got}
    unknown = flagged - ids
    if not _test(ctx, not unknown, case="suspects(clean_rows)", title="Every suspect is a scan id of clean_rows",
                       expected="ids from clean_rows", got=f"{len(unknown)} unknown, e.g. {sorted(unknown)[:1]}"):
        return
    if not _test(ctx, len(flagged) <= 0.15 * len(ids), case="suspects(clean_rows)",
                       title="A short list a person can review (at most 15% of the rows)",
                       expected=f"at most {int(0.15 * len(ids))} suspects", got=f"{len(flagged)} suspects",
                       passed=f"{len(flagged)} suspects",
                       cause="stage 2 (5 nearest neighbours) should clear most of stage 1's suspects."):
        return
    hits = len(flagged & liars)
    precision = hits / len(flagged) if flagged else 0.0
    recall = hits / len(liars) if liars else 1.0
    ctx.note(f"Your {len(flagged)} suspects contain {hits} of the {len(liars)} rows whose typed label is wrong: "
             f"precision {precision:.0%}, recall {recall:.0%}.")
    _test(ctx, precision >= 0.25 and recall >= 0.5, case="suspects(clean_rows)",
                title="It finds wrong labels: precision of at least 25%, recall of at least 50%",
                expected="precision >= 25% and recall >= 50%", got=f"precision {precision:.0%}, recall {recall:.0%}",
          passed=f"precision {precision:.0%}, recall {recall:.0%}",
                cause="stage 1 flags rows whose own label's average is not the nearest; stage 2 keeps only those "
                      "none of whose 5 nearest neighbours (14 x 14) carries its label.")


SCI2 = Task(
    "SCI2", "suspects", title="Find the liars",
    practise="averages, nearest neighbours and precision/recall",
    rescue=False, custom=_sci2_score, quality=False,
)


# ============================================================================= ADVANCED · BLD2 parse_timestamp (sanity)
def _partner(first: object, name: str):
    main = sys.modules.get("__main__")
    found = getattr(main, name, None) if main is not None else None
    if callable(found):
        return found
    found = getattr(first, "__globals__", {}).get(name)
    return found if callable(found) else None


def _bld2_report(fn, ctx) -> None:
    """quality_report_md, the second function: a sanity check on the Markdown it returns."""
    md = _partner(fn, "quality_report_md")
    if md is None or is_stub(md):
        ctx.require(False, case="quality_report_md(report, clean_rows)",
                    title="quality_report_md returns a Markdown report", expected="a Markdown text",
                    got="not written yet", cause="parse_timestamp works: now write quality_report_md (the cell below).")
        return
    report = {"total": 10, "valid": 6, "invalid": 4, "reasons": {"duplicate": 3, "bad label": 1}}
    rows = [{"scan_id": f"PX-00000{i}", "depot": d, "label": i, "pixels": [0] * 784}
            for i, d in enumerate(["Chamartín", "Retiro", "Retiro", "Vallecas", "Chamartín", "Retiro"])]
    attempt = ctx.call(md, report, rows, show="quality_report_md(report, clean_rows)", track=False)
    attempt.title = "quality_report_md returns a Markdown report"
    if attempt.error is not None:
        ctx.expect(attempt, _Said("a Markdown text"))
        return
    text = attempt.got
    ok = (isinstance(text, str) and text.lstrip().startswith("#") and "|" in text and "duplicate" in text
          and "Chamartín" in text and "10" in text)
    _test(ctx, ok, case="quality_report_md(report, clean_rows)", title="quality_report_md returns a Markdown report",
                expected="a # title, a | table | with every reason, the total, and the rows kept per depot",
                got=short(text, 60), cause="start with a '# ' title line, then a table: | reason | rows |.")


BLD2 = Task(
    "BLD2", "parse_timestamp", title="A report for humans",
    practise="datetime.strptime with several formats, and Markdown",
    rescue=False, custom=_bld2_report, quality=False, expected_label="Expected",
    input_labels=("Raw timestamp",),
    cases=[
        Case(("2026-11-02T14:05",), "2026-11-02T14:05", title="Already ISO 8601: unchanged", highlight=True),
        Case(("02/11/2026 14:05",), "2026-11-02T14:05", title="Day first, with a time: 2 November", highlight=True),
        Case(("2026/11/02",), "2026-11-02", title="A date only, with slashes"),
        Case(("02/11/2026",), "2026-11-02", title="A date only, day first"),
        Case(("",), None, title="Empty: None"),
        Case(("N/A",), None, title="N/A: None"),
        Case(("31/02/2026 10:00",), None, title="31 February doesn't exist: None"),
    ],
    returns=(str, type(None)),
)


# ============================================================================= ADVANCED · SCI3 compare_training_sets (sanity)
class _MajorityModel:
    """A stand-in for a scikit-learn model, with the same three methods: fit, predict and score.
    It predicts the most common training label; it also records what it was given."""

    made = 0

    def __init__(self) -> None:
        type(self).made += 1
        self.fitted = 0
        self.label = None
        self.seen_max = None

    def fit(self, X, y):
        self.fitted += 1
        counts: dict = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        self.label = max(sorted(counts), key=counts.get) if counts else None
        self.seen_max = max((max(row) for row in X), default=0)
        return self

    def predict(self, X):
        return [self.label for _ in X]

    def score(self, X, y):
        if (self.seen_max is not None and self.seen_max > 1.0) or max((max(row) for row in X), default=0) > 1.0:
            return -1.0
        return sum(1 for label in y if label == self.label) / len(y) if y else 0.0


def _sci3_toy(fn, ctx) -> None:
    sets = {"naive": ([[255, 0], [0, 255], [255, 255]], [1, 1, 7]), "cleaned": ([[0, 255], [255, 0]], [7, 7])}
    test_images, test_labels = [[0, 255], [255, 0], [128, 128], [0, 0]], [7, 7, 1, 2]
    _MajorityModel.made = 0
    show = "compare_training_sets(make_model, training_sets, test_images, test_labels)"
    attempt = ctx.call(fn, _MajorityModel, sets, test_images, test_labels, show=show, track=True)
    attempt.title = "Two training sets, a stand-in model that always predicts the most common training label"
    if attempt.error is not None:
        ctx.expect(attempt, _Said("{'naive': 0.25, 'cleaned': 0.5}"))
        return
    got = attempt.got
    if not _test(ctx, isinstance(got, dict) and set(got) == set(sets), case=show,
                       title="It returns {name: test accuracy} for every training set",
                       expected="{'naive': ..., 'cleaned': ...}", got=f"{short(got, 50)} ({type_name(got)})", passed=short(got, 50)):
        return
    if not _test(ctx, _MajorityModel.made == len(sets), case=show, title="A NEW model for every training set",
                       expected=f"make_model() called {len(sets)} times", got=f"make_model() called {_MajorityModel.made} time{'' if _MajorityModel.made == 1 else 's'}",
                       cause="call make_model() inside the loop: a model trained on one set must never be reused "
                             "(or trained further) on the next one."):
        return
    if not _test(ctx, all(v != -1.0 for v in got.values()), case=show,
                       title="Pixels are scaled to 0.0-1.0 before fit and score",
                       expected="pixel / 255 everywhere", got="pixels from 0 to 255",
                       cause="neural networks learn best from small inputs: divide every pixel by 255, for the "
                             "training images AND the test images."):
        return
    _test(ctx, got == {"naive": 0.25, "cleaned": 0.5}, case=show, title="The scores come from model.score on the test set",
                expected="{'naive': 0.25, 'cleaned': 0.5}", got=short(got, 50),
                cause="score each model on the (scaled) TEST images and labels: model.score(X_test, test_labels).")


SCI3 = Task(
    "SCI3", "compare_training_sets", title="Does a real neural network care?",
    practise="the fit / score interface of scikit-learn, and a fair comparison",
    rescue=False, custom=_sci3_toy, quality=False,
)


# notebook order: CORE (W, C1-C4), OPTIONAL (C5, ENG1, SCI1, BLD1), ADVANCED (ENG2, SCI2, BLD2, SCI3)
TASKS = [W, C1, C2, C3, C4, C5, ENG1, SCI1, BLD1, ENG2, SCI2, BLD2, SCI3]
