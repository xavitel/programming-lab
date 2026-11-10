"""fuzz_strings(n): tricky text for property tests (Lab 3, ENG2).

    from labkit.fuzz import fuzz_strings

    for text in fuzz_strings(200):
        clean_label(text)              # a cleaner must never raise: a value, or None

The list starts with the classics that break cleaners (empty text, spaces, "NaN", "1e3", look-alike
letters, accents written two ways, digits from other alphabets, very long text, ...), then random
mixes. The same seed always gives the same list, so a failure can be reproduced.

You never need to change this file; open it if you're curious how it works.
"""
from __future__ import annotations

import random

__all__ = ["fuzz_strings", "CLASSICS"]

CLASSICS = [
    "", " ", "   ", "\t", "\n", " \t\n ", "\x00", "'", '"', ",", ";", "\\", "?", "-", ".", "..",
    "0", "7", " 7", "7 ", " 7 ", "07", "77", "-1", "+7", "7.0", "7,0", "1e3", "1_000", "0x1F", "nan", "NaN",
    "inf", "-inf", "None", "null", "n/a", "N/A", "seven", "SEVEN", " Seven ", "seventy", "O", "o", "l", "I",
    "i", "İ", "ı", "ǀ", "٧", "７", "²", "​7", "7 ", " ", "Chamartín",
    "ChamartÃ­n", "CHAMARTÍN", "Retiro", "VLC", "sí", "sí", "SÍ", "yes", "Y", "true", "1",
    "28013", " 28013", "28.013", "E-28013", "Madrid 28013", "2813", "08013", "28O13", "28 013",
    "0 " * 784, "255 " * 784, "256 " * 784, "0.5 " * 784, "1.0 " * 784, ("0 " * 783).strip(),
    " ".join(["0"] * 392 + ["NaN"] + ["0"] * 391), "a" * 10_000, "7" * 1000, "é" * 50,
]

_ALPHABET = ("0123456789" * 3 + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + " " * 6
             + ".,;:-_+/\\'\"?!@#()[]" + "\t\n" + "áéíóúñçÁÉÍÓÚÑ" + "́ ​٧７")


def fuzz_strings(n: int = 200, seed: int = 2026) -> list[str]:
    """n strings for property tests: the classics first, then random mixes (same seed, same list)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a whole number, 0 or more (got {n!r})")
    rng = random.Random(seed)
    out = list(CLASSICS[:n])
    while len(out) < n:
        length = rng.choice([1, 1, 2, 3, 4, 5, 8, 13, 40])
        text = "".join(rng.choice(_ALPHABET) for _ in range(length))
        if rng.random() < 0.2:
            text = " " + text + " "
        out.append(text)
    return out
