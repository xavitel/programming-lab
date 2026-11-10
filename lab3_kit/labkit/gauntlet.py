"""gauntlet(kind, pattern): run a regular expression against a set of adversarial cases (Lab 3).

    from labkit.gauntlet import gauntlet

    gauntlet("postcode", r"28\\d{3}")      # 20 cases: find a Madrid postcode (28001 to 28055) in messy text

For every case, re.search(pattern, text) must find exactly the expected text, or nothing at all when
nothing should be found. If the pattern has a group (...), the text of its first group counts too (what
match.group(1) gives). The cases are written to break the patterns people (and AI assistants) write first: a postcode inside a longer number, a postcode right after a letter, 28000 or 28099 (not Madrid
postcodes Garabato delivers to), and so on. Kinds: "postcode", "email", "phone". All the data is
fictional (example.org e-mails, phone numbers from reserved fictional ranges).

You never need to change this file; open it if you're curious how it works.
"""
from __future__ import annotations

import re

from .text import say, short

__all__ = ["gauntlet", "CASES"]

# kind -> (what the pattern must find, [(text, expected match or None, why it is a trap), ...])
CASES: dict[str, tuple[str, list[tuple[str, str | None, str]]]] = {
    "postcode": ("a Madrid postcode Garabato delivers to (28001 to 28055) inside messy text", [
        ("28013", "28013", "the plain case"),
        ("Madrid 28013", "28013", "a city name before it"),
        ("E-28013 Madrid", "28013", "a country prefix"),
        ("28013, Retiro", "28013", "a comma after it"),
        ("Retiro (28009)", "28009", "brackets around it"),
        ("28001", "28001", "the lowest postcode"),
        ("28055", "28055", "the highest postcode"),
        ("Madrid28013", "28013", "a letter right before it: \\b sees no word boundary there"),
        ("28013_retiro", "28013", "an underscore right after it: _ counts as a word character"),
        ("CP:28046.", "28046", "punctuation all around"),
        ("28000", None, "28000 is not a postcode Garabato delivers to"),
        ("28056", None, "one above the highest postcode"),
        ("28099", None, "28 + three digits, but not a Madrid postcode we deliver to"),
        ("08013 Barcelona", None, "a Barcelona postcode"),
        ("128013", None, "inside a longer number"),
        ("280130", None, "a longer number that starts with 28013"),
        ("2813", None, "a digit missing"),
        ("28O13", None, "a letter O instead of a zero"),
        ("order 4528013", None, "the end of a longer number"),
        ("scan PX-028013", None, "part of a scan id"),
    ]),
    "email": ("one e-mail address", [
        ("write to op.amber17@example.org", "op.amber17@example.org", "the plain case"),
        ("op.birch4@example.org, please", "op.birch4@example.org", "a comma after it"),
        ("(op.dune9@example.org)", "op.dune9@example.org", "brackets around it"),
        ("OP.EMBER31@EXAMPLE.ORG", "OP.EMBER31@EXAMPLE.ORG", "capital letters"),
        ("dispatch+returns@depot.example.org", "dispatch+returns@depot.example.org", "a + and a subdomain"),
        ("op_lumen6@example.org", "op_lumen6@example.org", "an underscore"),
        ("mail op.kelp27@example.org or call 555-0123", "op.kelp27@example.org", "a phone number nearby"),
        ("op.amber17@example.org.", "op.amber17@example.org", "a full stop after it"),
        ("no e-mail in this note", None, "nothing to find"),
        ("PX-001234", None, "a scan id"),
        ("@example.org", None, "no name before the @"),
        ("op.amber17@", None, "no domain after the @"),
        ("box 4@5 crushed", None, "an @ between two numbers"),
    ]),
    "phone": ("one phone number from the fictional ranges (UK 07700 900xxx, US 555-01xx)", [
        ("call 07700 900461", "07700 900461", "the plain case"),
        ("07700900461", "07700900461", "no space"),
        ("+44 7700 900461", "+44 7700 900461", "an international prefix"),
        ("555-0123", "555-0123", "the US fictional range"),
        ("if absent, call 555-0199.", "555-0199", "a full stop after it"),
        ("PX-001234", None, "a scan id"),
        ("28013", None, "a postcode"),
        ("2026-11-02T14:05", None, "a timestamp"),
        ("0.4627", None, "a pixel value"),
        ("02/11/2026 14:05", None, "a date"),
    ]),
}


def gauntlet(kind: str, pattern: object) -> dict:
    """Run `pattern` (a regular expression, as text) against the adversarial cases of `kind`.

    Prints one line per case (PASS or FAIL, what your pattern found, why the case is a trap) and the
    score. Returns {"passed": n, "total": n, "failed": [texts]}.
    """
    if kind not in CASES:
        say(f"gauntlet: I don't know the kind {kind!r}. Kinds: {', '.join(CASES)}.")
        return {}
    if isinstance(pattern, re.Pattern):
        compiled = pattern
    elif isinstance(pattern, str):
        if not pattern.strip():
            say("gauntlet: the pattern is empty. Paste the regular expression between the quotes, e.g. "
                "pattern = r\"28\\d{3}\".")
            return {}
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            say(f"gauntlet: Python can't read this pattern ({error}). Check the brackets and backslashes; "
                "a raw string r\"...\" keeps each backslash as it is.")
            return {}
    else:
        say(f"gauntlet: pass the pattern as text, e.g. r\"28\\d{{3}}\", not {type(pattern).__name__}.")
        return {}
    goal, cases = CASES[kind]
    text = compiled.pattern if isinstance(compiled.pattern, str) else repr(compiled.pattern)
    shown = f'r"{text}"' if '"' not in text else f"r'{text}'"
    say(f"Gauntlet \"{kind}\": your pattern {shown} must find {goal}.")
    if compiled.groups:
        say("  (Your pattern has a group (...): the text of the first group counts too.)")
    passed, failed = 0, []
    width = max(len(repr(text)) for text, _, _ in cases) + 2
    for text, expected, why in cases:
        try:
            match = compiled.search(text)
        except Exception as error:                       # e.g. a catastrophic pattern interrupted
            match, found = None, f"error ({type(error).__name__})"
        found = match.group(0) if match else None
        group = match.group(1) if match and compiled.groups else None
        used_group = found != expected and group is not None and group == expected
        if used_group:
            found = group
        ok = found == expected
        passed += ok
        if not ok:
            failed.append(text)
        if ok:
            what = (f"found {found!r}" + (" (group 1)" if used_group else "") if found is not None
                    else "found nothing, as it should")
        elif expected is None:
            what = f"found {found!r}, but should find nothing"
        elif found is None:
            what = f"found nothing, should find {expected!r}"
        else:
            what = f"found {found!r}, should find {expected!r}"
            if group is not None and group != found:
                what += f" (group 1: {group!r})"
        say(f"  {'PASS' if ok else 'FAIL'}  {repr(text).ljust(width)}{what}  ({why})")
    say(f"Score: {passed} of {len(cases)}." + ("" if passed == len(cases) else
        " Each FAIL is a case your pattern gets wrong: change the pattern and run the cell again."))
    return {"passed": passed, "total": len(cases), "failed": failed}


def _describe(kind: str) -> str:
    """The cases of `kind`, one per line (for the teacher guide)."""
    goal, cases = CASES[kind]
    return "\n".join(f"{short(text, 40)} -> {expected!r}" for text, expected, _ in cases)
