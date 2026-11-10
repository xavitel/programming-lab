"""cleaners.py: the cleaners of Lab 3 · The Scanner Mess, as a module (ADVANCED task ENG2).

Every function takes the raw text of one value and returns a clean value, or None when the value can't
be trusted. Module rules (Theory S13, modules):
- no print() and no input(): a cleaner returns, the caller decides what to show;
- None, and only None, means "I can't trust this value" (never "", 0 or -1);
- a docstring in every function: someone who never saw your notebook will call these.

Use it from the notebook:
    import cleaners
    cleaners.clean_label(" Seven ")         # 7
    check("ENG2", cleaners)

After every save, run the notebook's import cell again (it reloads this module): no restart needed, and
check("ENG2", cleaners) always reads the file as it is saved.
"""
from __future__ import annotations

import unicodedata

# ===== YOUR CODE HERE =====
# ✏️ WRITE CODE: paste your functions from the notebook below each docstring (replace `pass`), then save.
LABEL_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
LOOK_ALIKES = {"o": 0, "l": 1, "i": 1}


def clean_label(raw: str) -> int | None:
    """Return the digit a person typed (an int from 0 to 9), or None if you can't trust it (C1)."""
    pass


def parse_pixels(raw: str) -> list[int] | None:
    """Return 784 ints from 0 to 255, or None if the text is anything else (C2)."""
    pass


def rescale_floats(raw: str) -> list[int] | None:
    """Return 784 fractions from 0.0 to 1.0 as ints from 0 to 255 (round(v * 255)), or None (SCI1)."""
    pass


def clean_postcode(raw: str, valid: set[str]) -> str | None:
    """Return a clean Madrid postcode such as "28013" that is in `valid`, or None (ENG1)."""
    pass


def clean_depot(raw: str, aliases: dict[str, str]) -> str | None:
    """Return the official depot name for any spelling in `aliases` ({casefolded alias: name}), or None (BLD1)."""
    pass
