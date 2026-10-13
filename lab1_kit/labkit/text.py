"""Small text helpers shared by every labkit module.

- say() prints, and never crashes on a console that cannot show symbols such as → or ×.
- field() builds one aligned line of a check report ("   Expected:      25").
- short() gives a one-line preview of any value, so a 784-pixel image never floods the screen.
- hide_user_folder() replaces your user folder (it often contains your name) with ~ in a traceback.
"""
from __future__ import annotations

import math
import os
import re
import sys

INDENT = "   "          # every detail line of a report starts with 3 spaces
LABEL_WIDTH = 15        # "Failing case:  " and "Expected:      " both take 15 characters

# Plain-text stand-ins, used only when the console cannot print the real symbol.
_PLAIN = {
    "→": "->", "–": "-", "—": "-", "·": "|", "×": "x", "µ": "u", "≈": "~", "±": "+/-",
    "…": "...", "⭐ ": "", "🛟 ": "", "💡 ": "", "⏱ ": "", "🎯 ": "", "🚀 ": "", "✍️ ": "",
    "⭐": "*", "🛟": "", "💡": "", "⏱": "", "🎯": "", "🚀": "", "✍️": "",
}


def say(*lines: str) -> None:
    """Print each argument on its own line.

    In Jupyter every symbol prints fine. A Windows terminal that uses an old code page
    cannot print some symbols, so we swap them for plain text instead of crashing.
    """
    text = "\n".join(str(line) for line in lines)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        for symbol, plain in _PLAIN.items():
            text = text.replace(symbol, plain)
        print(text.encode(encoding, "replace").decode(encoding), flush=True)


# A user folder in a path: C:\Users\<name> (also the short form C:\Users\ANARUI~1, and with \\ or / as
# separators, as in an error message), /Users/<name> (macOS) and /home/<name> (Linux).
_USER_FOLDER = re.compile(r"[A-Za-z]:(?:\\\\|\\|/)(?:Users|Documents and Settings)(?:\\\\|\\|/)[^\\/:*?\"'<>|\r\n]+"
                          r"|(?<![\w.~])/(?:Users|home)/[^/\"'\r\n]+")


def hide_user_folder(text: str) -> str:
    """Replace the user folder in `text` with ~, because a folder name is often the person's real name.

    Use it before showing a traceback or a path that someone may paste into an AI tool or a forum:

    >>> hide_user_folder('File "C:\\\\Users\\\\Ana Ruiz\\\\lab1_kit\\\\labkit\\\\check.py", line 3')
    'File "~\\\\lab1_kit\\\\labkit\\\\check.py", line 3'
    """
    text = str(text)
    homes = {value.rstrip("\\/") for value in (os.path.expanduser("~"), os.environ.get("USERPROFILE"),
                                                os.environ.get("HOME")) if value}
    for home in sorted((h for h in homes if len(h) > 3), key=len, reverse=True):   # never "/" or "C:"
        for form in (home, home.replace("\\", "\\\\")):         # as printed, and as repr() shows it
            text = re.sub(re.escape(form), "~", text, flags=re.IGNORECASE)
    return _USER_FOLDER.sub("~", text)


def field(label: str, text: str) -> str:
    """Return one aligned report line, e.g. field("Expected:", "25") -> '   Expected:      25'."""
    return INDENT + (label + " ").ljust(LABEL_WIDTH) + str(text)


def thousands(n: float) -> str:
    """1000 -> '1,000' (whole numbers only)."""
    return f"{int(n):,}"


def plural(n: int, word: str, many: str | None = None) -> str:
    """plural(1, "image") -> '1 image'; plural(3, "image") -> '3 images'."""
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def type_name(value: object) -> str:
    """'an int', 'a str', 'a list', 'None', ... for messages such as 'Got: 25.0 (a float)'."""
    if value is None:
        return "None"
    name = type(value).__name__
    article = "an" if name[:1].lower() in "aeiou" else "a"
    return f"{article} {name}"


def is_number(value: object) -> bool:
    """True for int and float values (bool does not count as a number here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception:  # an object whose __repr__ fails, or an int too long to print
        if isinstance(value, int):
            return f"<a very large int, about {int(value.bit_length() * 0.30103):,} digits>"
        return f"<{type(value).__name__} object>"


def short(value: object, limit: int = 70) -> str:
    """A one-line preview of `value`, at most about `limit` characters.

    >>> short([0] * 784)
    '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …] (784 values)'
    """
    if isinstance(value, float):
        return _safe_repr(value)
    if isinstance(value, str):
        text = _safe_repr(value)
        if len(text) <= limit:
            return text
        lines = value.split("\n")
        head = _safe_repr(lines[0][: limit - 10] if len(lines) > 1 else value[: limit - 10])
        if len(lines) > 1:
            return f"{head[:-1]}…' ({len(lines)} lines, {len(value):,} characters)"
        return f"{head[:-1]}…' ({len(value):,} characters)"
    if isinstance(value, (list, tuple, set, frozenset)):
        text = _safe_repr(value)
        if len(text) <= limit:
            return text
        items = list(value)
        open_, close = {list: "[]", tuple: "()"}.get(type(value), "{}")
        if items and all(isinstance(v, (list, tuple)) for v in items[:50]):
            sizes = {len(v) for v in items[:50]}
            inner = f" of {sizes.pop():,} values" if len(sizes) == 1 else ""
            first = short(items[0], 30)
            return f"{open_}{first}, …{close} ({len(items):,} lists{inner})"
        shown = []
        for item in items:
            piece = short(item, 20)
            if len(", ".join(shown + [piece])) > limit - 25:
                break
            shown.append(piece)
        return f"{open_}{', '.join(shown)}, …{close} ({len(items):,} values)"
    if isinstance(value, dict):
        text = _safe_repr(value)
        if len(text) <= limit:
            return text
        shown = []
        for key, item in value.items():
            piece = f"{short(key, 15)}: {short(item, 20)}"
            if len(", ".join(shown + [piece])) > limit - 20:
                break
            shown.append(piece)
        return "{" + ", ".join(shown) + f", …}} ({len(value):,} keys)"
    text = _safe_repr(value)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def first_difference(expected: object, got: object) -> str | None:
    """Point at the first place where two long strings or lists differ, or return None.

    Used under Expected/Got when the values are too long to compare by eye.
    """
    if isinstance(expected, str) and isinstance(got, str) and expected != got:
        if len(expected) <= 40 and len(got) <= 40:
            return None
        exp_lines, got_lines = expected.split("\n"), got.split("\n")
        if len(exp_lines) != len(got_lines):
            return f"{plural(len(exp_lines), 'line')} expected, {len(got_lines):,} found"
        for row, (e, g) in enumerate(zip(exp_lines, got_lines), start=1):
            if e != g:
                if len(e) != len(g):
                    return f"line {row} should have {plural(len(e), 'character')}, it has {len(g):,}"
                col = next(i for i, (a, b) in enumerate(zip(e, g)) if a != b)
                return f"line {row}, character {col + 1}: expected {e[col]!r}, got {g[col]!r}"
        return None
    if isinstance(expected, (list, tuple)) and isinstance(got, (list, tuple)) and len(expected) > 8:
        if len(expected) != len(got):
            return f"{plural(len(expected), 'value')} expected, {len(got):,} found"
        for index, (e, g) in enumerate(zip(expected, got)):
            try:
                same = bool(e == g)
            except Exception:
                same = False
            if not same:
                return f"first difference at index {index}: expected {short(e, 30)}, got {short(g, 30)}"
    return None


def preview_pair(expected: object, got: object) -> tuple[str, str]:
    """One-line previews of an expected and an actual value, chosen so that they differ on screen.

    Two long multi-line strings (e.g. a rendered digit) are shown by their first differing line;
    if two different values still look the same, their types are added.
    """
    if (isinstance(expected, str) and isinstance(got, str) and "\n" in expected + got
            and max(len(expected), len(got)) > 60):
        exp_lines, got_lines = expected.split("\n"), got.split("\n")
        for row in range(max(len(exp_lines), len(got_lines))):
            e = exp_lines[row] if row < len(exp_lines) else None
            g = got_lines[row] if row < len(got_lines) else None
            if e != g:
                return (f"line {row + 1}: {short(e, 60)}  ({len(exp_lines)} lines in all)",
                        f"line {row + 1}: {short(g, 60) if g is not None else '(no such line)'}  "
                        f"({len(got_lines)} lines in all)")
    shown_expected, shown_got = short(expected), short(got)
    if shown_expected == shown_got:
        shown_got += f" ({type_name(got)}, expected {type_name(expected)})"
    return shown_expected, shown_got


def is_finite_number(value: object) -> bool:
    """True for a real number that is not inf or nan."""
    return is_number(value) and math.isfinite(value)
