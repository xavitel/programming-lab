"""Six deliberately buggy versions of digits.py, for judging the tests you write in Lab 4 (ENG1).

    python -m labkit.mutate tests/test_digits.py

Each mutant is ONE small change to a correct digits.py (labkit/reference/lab4.py): (what it does wrong, the
text it replaces, the buggy text). A good test fails on the buggy version: then the mutant is "caught".
This file holds buggy code on purpose; none of it is a solution.
"""
from __future__ import annotations

MUTANTS = [
    ("a save_model that overwrites an existing model without --force",
     "    if path.exists() and not force:\n        raise FileExistsError",
     "    if path.exists() and not force and False:\n        raise FileExistsError"),
    ("a main() that prints its error messages on stdout instead of stderr",
     '        print(f"error: {err}", file=sys.stderr)',
     '        print(f"error: {err}")'),
    ("a main() that returns exit code 0 after an error",
     "        return 1\n    print(output)\n    return 0",
     "        return 0\n    print(output)\n    return 0"),
    ("a train command that ignores --limit (it always learns from every image)",
     "    images, labels = load_digits(args.data, limit=args.limit)\n    model = new_model(",
     "    images, labels = load_digits(args.data, limit=None)\n    model = new_model("),
    ("a predict command that reads a .txt image as if it were a .csv image",
     "        image = read_image_grid(args.image)",
     "        image = read_image_csv(args.image)"),
    ("a load_model that accepts a model of the wrong version",
     '    if model["version"] != MODEL_VERSION:',
     '    if model["version"] != MODEL_VERSION and False:'),
]
