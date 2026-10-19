"""Checks for Lab 2 · Read a Postcode, Pythonically (blueprint §5.2, as amended by docs/plan_labs2-5.md §1).

What each task is checked against. Expected values that come from the reference are stored in
cases/lab2_data.json by generators/gen_cases.py, so this file holds no solution code: only the inputs,
the toy answers and the misconception detectors ("Likely cause"). Real cases are golden checks: the
reference gives exactly the answers of the old legacy_reader.py (measured at build time by
labs/lab2_read_a_postcode/qa/measure_lab2.py), so their titles say "the same answer as legacy_reader".

Tasks, in notebook order:
    W                  ink_count (worked example, provided)
    C1 C2 C3 C4 C5     CORE: squared_distance, k_nearest, majority_vote, predict_knn, test_majority_vote
    AI                 "AI wrote this": knn_predict_ai (after the CORE; no rescue)
    C6 ENG1 SCI1 BLD1  OPTIONAL: normalise, fast_distance, downsample, per_digit_accuracy (+ report_card)
    ENG2 SCI2 BLD2 CH  ADVANCED: no rescue (test_knn, cross_validate, shift, read_postcode)
    BR1 BR2 BR3 BR4    bridge_2to3.ipynb (the 3-week gap; run inside the Lab 2 kit): save_predictions,
                       load_predictions, save_accuracy, parse_depot_line; rescue from reference/lab2.py
    HQ1 ... HQ5        holiday_quest.ipynb (the reject option of the Postcode Machine): code_score, best_two,
                       read_or_reject, cost_per_1000, margin_costs; no rescue (the solution is published at
                       Lab 3; instructor copy: teacher_only/solutions/holiday_quest.py, which self-checks)
The BR and HQ tasks are neither CORE nor OPTIONAL, so check_all() in the Lab 2 notebook never lists them.

Every case has a short title, printed as "TEST n: <title>" with its inputs labelled (Neighbours, Image, ...).
Two-function tasks take the first function and find the second by name (in the notebook first, then next
to the first one, so rescue() works):
    check("BLD1", per_digit_accuracy)     # also checks report_card
    check("BLD2", shift)                  # also checks to_grid and flatten
    check("CH", read_postcode)            # also checks read_postcode_constrained
Tests that students write (C5, ENG2) are judged on a correct version and on buggy versions from
labkit/mutants/lab2.py; "at least 3 inputs" counts the calls of the function under test while the test runs,
so a table-driven test (one assert in a loop) counts as much as three assert lines.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import tempfile

from ..check import is_stub, read_asserts as _read_asserts, student_line
from ..mutants import lab2 as mutants
from ..spec import Case, Detector, Digits, Fact, Speed, Task, approx, custom
from ..text import is_number

DATA_FILES = ("mnist_train_10k.csv.gz", "mnist_test_2k.csv.gz", "mnist_train_10k_14x14.csv.gz",
              "mnist_test_2k_14x14.csv.gz", "postcodes_test.csv", "valid_postcodes.txt")
FINALE = "Putting It All Together"     # the name of the CORE's final run (check_all, evaluate, start)

TRAIN = Digits("mnist_train_10k.csv.gz", limit=1000)     # = train_images, train_labels in the notebook
TEST = Digits("mnist_test_2k.csv.gz", limit=100)         # = test_images, test_labels in the notebook
TRAIN14 = Digits("mnist_train_10k_14x14.csv.gz", limit=1000)
N_GOLDEN = 300                                            # training images in the golden checks (fast)

NEXT_HINT_2 = "open Hint 2 below (the plan), fix your code, run the cell with def again, then this check."


# ============================================================================= small helpers
def _same_int(got: object, expected: object) -> bool:
    """An int label equal to the expected one (a float such as 3.0, a bool or a str doesn't count)."""
    return type(got) is int and got == expected


INT_LABEL = custom(_same_int, "(an int)")


def _partner(first: object, name: str):
    """The second function of a two-function task: in the notebook first, then next to `first`."""
    main = sys.modules.get("__main__")
    found = getattr(main, name, None) if main is not None else None
    if callable(found):
        return found
    found = getattr(first, "__globals__", {}).get(name)
    return found if callable(found) else None


def _partner_state(t, name: str) -> str | None:
    """'missing' / 'stub' / None for the partner function of a two-function task."""
    fn = _partner(t.args[0] if t.args else None, name)
    if fn is None:
        return "missing"
    if is_stub(fn):
        return "stub"
    return None


def _call_partner(first: object, name: str, *args, **kwargs):
    fn = _partner(first, name)
    return None if fn is None else fn(*args, **kwargs)


def _labels(t) -> list:
    """The labels of a list of (distance, label) neighbours (C3's argument)."""
    try:
        return [label for _distance, label in t.args[0]]
    except Exception:
        return []


def _counts(t) -> dict:
    counts: dict = {}
    for label in _labels(t):
        counts[label] = counts.get(label, 0) + 1
    return counts


def _tied(t) -> list:
    counts = _counts(t)
    if not counts:
        return []
    top = max(counts.values())
    return [label for label, count in counts.items() if count == top]


def _golden(i: int, name: str, call_text: str, **extra) -> Case:
    """A real case: test image #i against the first 300 training images (a golden check)."""
    return Case(lambda: (TEST.images[i], TRAIN.images[:N_GOLDEN], TRAIN.labels[:N_GOLDEN]), key=f"real-{i}",
                show=f"{name}(test_images[{i}], train_images[:{N_GOLDEN}], train_labels[:{N_GOLDEN}]{call_text})",
                title=f"The same answer as legacy_reader.{'predict' if name != 'k_nearest' else name}: test "
                      f"image #{i}, against a sample of {N_GOLDEN} training images (for speed)", **extra)


def _key_line(t) -> bool:
    """The error happened on a line of the student's own code that passes key= (to sorted, max or min)."""
    if t.error is None:
        return False
    found = student_line(t.error)
    return found is not None and "key=" in found[1].replace(" ", "")


def _key_name_error(t) -> bool:
    """A NameError for the name written right after key= (e.g. key=pair[0], with no lambda)."""
    if not t.raised(NameError):
        return False
    found = student_line(t.error)
    name = getattr(t.error, "name", None) or (re.findall(r"'(\w+)'", str(t.error)) or [""])[0]
    return bool(found and name and re.search(rf"key\s*=\s*{re.escape(name)}\b", found[1]))


# ============================================================================= W · ink_count (worked example)
# W and AI have no hint ladder, so their reports must not send students to "Hint 1" (the default Next step).
_W_NEXT = "fix it, run the cell with def again, then the golden check and this check."
_AI_NEXT = ("fix one bug in the copy (the cell with def knn_predict_ai), run that cell, then this check again. "
            "'What we found' has the answers once you've tried.")
W = Task(
    "W", "ink_count", title="Refactor ink_count",
    practise="a generator expression inside sum() (Theory S4)",
    input_labels=("Image",), expected_label="Expected count",
    cases=[
        Case(([0, 128, 255, 127],), 2, highlight=True,
             title="Four pixels: 128 and 255 are ink (128 is exactly the threshold)"),
        Case(([0, 50, 100, 200, 30, 120],), 4, kwargs={"threshold": 50}, title="A lower threshold: 50"),
        Case(([],), 0, title="An empty image has no ink"),
    ] + [Case(lambda i=i: (TEST.images[i],), key=f"real-{i}", show=f"ink_count(test_images[{i}])",
              highlight=i == 0, title=f"The same answer as legacy_reader.ink_count: test image #{i}")
         for i in (0, 1, 2)],
    detectors=[
        Detector(lambda t: t.args and isinstance(t.got, int) and t.got == sum(
                     1 for p in t.args[0] if p > t.kwargs.get("threshold", 128)),
                 "a pixel exactly equal to the threshold counts as ink: use >= threshold, not >.",
                 next_step="put >= back, run the cell with def again, then the golden check and this check."),
        Detector(lambda t: t.args and isinstance(t.got, int) and t.got == len(t.args[0]) - t.expected,
                 "you counted the paper pixels. Count the pixels that are >= threshold.",
                 next_step=_W_NEXT),
        Detector(lambda t: "threshold" in t.kwargs and t.got == sum(1 for p in t.args[0] if p >= 128),
                 "your code uses 128 where it should use the threshold parameter.",
                 next_step=_W_NEXT),
        Detector(lambda t: t.args and isinstance(t.got, int) and t.got == sum(t.args[0]) and t.got != t.expected,
                 "you added up the pixel values. Add a 1 for each ink pixel: sum(1 for pixel in image if ...).",
                 next_step=_W_NEXT),
    ],
    returns=int,
)


# ============================================================================= C1 · squared_distance
def _c1_wrong(t, value) -> bool:
    """C1: the student's number equals `value(a, b)` (a known mistake) and isn't the right answer."""
    return len(t.args) == 2 and is_number(t.got) and t.got != t.expected and t.got == value(t.args[0], t.args[1])


def _symmetric(fn) -> list[bool]:
    pairs = [(TEST.images[k][:], TRAIN.images[k][:]) for k in (1, 2, 3)]
    return [fn(a[:], b[:]) == fn(b[:], a[:]) for a, b in pairs]


C1 = Task(
    "C1", "squared_distance", title="How different are two images? In one line",
    practise="zip() and a generator expression inside sum() (Theory S4)",
    minimum_path=True,
    input_labels=("Image a", "Image b"), expected_label="Expected distance",
    cases=[
        Case(([0, 0], [3, 4]), 25, highlight=True,
             title="Two tiny images of 2 pixels: (0 - 3) ** 2 + (0 - 4) ** 2 = 9 + 16"),
        Case(([1, 2, 3], [1, 2, 3]), 0, highlight=True, title="Identical images are 0 apart"),
        Case(([2, 7, 1], [5, 3, 4]), 34, title="Three pixels: 9 + 16 + 9"),
        Case(call=_symmetric, expected=[True, True, True], key="symmetry", expected_label="Expected",
             facts={"symmetry": True},
             title="The same distance both ways: squared_distance(a, b) == squared_distance(b, a)",
             show="squared_distance(a, b) == squared_distance(b, a) for 3 pairs of real images"),
        Case(lambda: (TEST.images[0], TRAIN.images[0]), key="real-0", highlight=True,
             title="The same answer as legacy_reader.squared_distance: test image #0 and training image #0",
             show="squared_distance(test_images[0], train_images[0])"),
    ],
    detectors=[
        Detector(lambda t: t.raised(TypeError) and "generator" in str(t.error),
                 "a generator expression is a recipe, not a number: (x - y) ** 2 for x, y in zip(a, b) needs "
                 "sum(...) around it, and then you can't add or square it again outside."),
        Detector(lambda t: hasattr(t.got, "__next__") or type(t.got).__name__ == "generator",
                 "you returned the generator expression itself. Wrap it in sum(...): "
                 "return sum((x - y) ** 2 for x, y in zip(a, b))."),
        Detector(lambda t: t.facts.get("symmetry") and isinstance(t.got, list) and False in t.got,
                 "your answer changes when you swap a and b, so it can't be a distance yet. Square each "
                 "difference: (x - y) ** 2 is the same as (y - x) ** 2."),
        Detector(lambda t: not t.facts.get("symmetry") and isinstance(t.got, list),
                 "you returned the list of squared differences. Add them up: sum(...) around the comprehension "
                 "(the square brackets are then not needed)."),
        Detector(lambda t: isinstance(t.got, float) and is_number(t.expected) and t.expected > 0
                 and abs(t.got ** 2 - t.expected) < 1e-6 * max(1, t.expected),
                 "you returned the square root (math.dist does that). The spec asks for the squared distance."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum(x - y for x, y in zip(a, b)) ** 2),
                 "you squared the total instead of each difference: the ** 2 belongs inside sum(...), "
                 "on (x - y)."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum(abs(x - y) for x, y in zip(a, b))),
                 "you added abs(x - y). Square each difference instead: (x - y) ** 2."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum(x - y for x, y in zip(a, b))),
                 "you added the differences without squaring them, so positive and negative ones cancel out. "
                 "Square each one: (x - y) ** 2."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum(x - y ** 2 for x, y in zip(a, b))),
                 "** runs before -, so x - y ** 2 squares only y. Put brackets around the difference: "
                 "(x - y) ** 2."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum((x - y) ^ 2 for x, y in zip(a, b))),
                 "in Python, ^ is not a power (it compares bits: XOR). Square with ** 2."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum((x - y) ** 2 for x in a for y in b)),
                 "two for clauses (for x in a for y in b) pair every pixel of a with every pixel of b. zip(a, b) "
                 "pairs them position by position: for x, y in zip(a, b)."),
        Detector(lambda t: t.got is None and t.error is None,
                 "your function returned None. Write return in front of the expression: return sum(...)."),
    ],
    returns=int,
    speed=Speed(lambda: (TEST.images[0], TRAIN.images[0]), budget_s=150e-6,
                note="k_nearest calls this once per training image, for every digit it reads."),
)


# ============================================================================= C2 · k_nearest
def _c2_facts() -> dict:
    """The wrong answers the C2 detectors recognise, stored by gen_cases from reference/lab2.py."""
    return {"tuples": Fact("nearest_sorted_tuples"), "unsorted": Fact("nearest_unsorted"),
            "reversed": Fact("farthest_first"), "label_first": Fact("nearest_label_first"),
            "by_label": Fact("nearest_by_label"), "last_k": Fact("farthest_last_k")}


C2 = Task(
    "C2", "k_nearest", title="The k closest pictures, sorted with a key",
    practise="pairs as tuples, zip(), and sorted() with key= and a lambda (Theory S3–S4)",
    minimum_path=True,
    depends_on=("C1",),
    input_labels=("Image", "Training images", "Training labels"), expected_label="Expected neighbours",
    cases=[
        # the card's toy example: distances 16, 1, 1, 25; [4] (label 8) and [6] (label 3) tie, and the EARLIER
        # one has the bigger label, so sorted(pairs) without a key would swap them
        Case(([5], [[9], [4], [6], [0]], [7, 8, 3, 5]), [(1, 8), (1, 3)], kwargs={"k": 2}, highlight=True,
             title="The card's tiny world: distances 16, 1, 1, 25; [4] and [6] tie, so they keep the training "
                   "order", facts=_c2_facts()),
        Case(([5], [[7], [3], [9]], [6, 1, 2]), [(4, 6), (4, 1)], kwargs={"k": 2}, highlight=True,
             title="Another tie: [7] and [3] are both 4 away, so they keep the training order (label 6 first)",
             facts=_c2_facts()),
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), [(1, 1), (1, 9), (4, 9)],
             title="k = 3 by default: five training images, the three nearest", facts=_c2_facts()),
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), [(1, 1)], kwargs={"k": 1},
             title="k = 1: only the nearest one, still inside a list"),
    ] + [_golden(i, "k_nearest", ", k=3", kwargs={"k": 3}, highlight=i == 0, facts=_c2_facts())
         for i in (0, 1, 2, 3)],
    detectors=[
        Detector(_key_name_error,
                 "key= needs a function. pair[0] on its own means nothing to Python here (it doesn't know pair "
                 "yet): write key=lambda pair: pair[0], which reads 'for each pair, compare pair[0]'."),
        Detector(lambda t: t.raised(TypeError) and "NoneType" in str(t.error) and "subscriptable" in str(t.error),
                 "list.sort() sorts the list in place and returns None, so there is nothing to slice. Use "
                 "sorted(pairs, key=...), which returns a new sorted list, then [:k]."),
        Detector(lambda t: t.raised(TypeError) and _key_line(t),
                 "key= needs a function, which sorted() calls on each pair to get the value it compares: "
                 "key=lambda pair: pair[0]. A number, a list, or a call with brackets after it is not a function."),
        Detector(lambda t: t.raised(TypeError) and ("'int' and 'list'" in str(t.error)
                                                      or "'list' and 'int'" in str(t.error)),
                 "squared_distance got a whole list of images instead of one image. Pair each training image "
                 "with its label: for train, label in zip(train_images, train_labels)."),
        Detector(lambda t: t.got is None and t.error is None,
                 "your function returned None. Did you return pairs.sort(...)? list.sort() sorts in place and "
                 "returns None. Use sorted(pairs, key=...), which returns a new list."),
        Detector(lambda t: t.facts.get("tuples") is not None and t.got == t.facts["tuples"] and t.got != t.expected,
                 "you sorted the (distance, label) tuples without a key. On equal distances, Python then compares "
                 "the second items, the labels, so the smaller label jumps ahead. The spec keeps the training "
                 "order on a tie: sort by the distance only, key=lambda pair: pair[0].",
                 next_step="read the table about ties on the C2 card, then open Hint 3 below."),
        Detector(lambda t: t.facts.get("label_first") is not None and t.got == t.facts["label_first"],
                 "your pairs are (label, distance). The spec asks for (distance, label): the distance first."),
        Detector(lambda t: t.facts.get("unsorted") is not None and t.got == t.facts["unsorted"],
                 "these are the first k pairs in training order: they were never sorted. sorted(...) returns a "
                 "NEW sorted list; keep that one."),
        Detector(lambda t: t.facts.get("reversed") is not None and t.got == t.facts["reversed"],
                 "the farthest images come first. The closest come first: sort without reverse=True."),
        Detector(lambda t: t.facts.get("by_label") is not None and t.got == t.facts["by_label"]
                 and t.got != t.expected,
                 "your key looks at the label. In a (distance, label) pair the distance is the first item: "
                 "key=lambda pair: pair[0]."),
        Detector(lambda t: t.facts.get("last_k") is not None and t.got == t.facts["last_k"] and t.got != t.expected,
                 "[-k:] keeps the LAST k pairs of the sorted list: the farthest ones. [:k] keeps the first k, "
                 "the nearest."),
        Detector(lambda t: isinstance(t.got, list) and isinstance(t.expected, list)
                 and t.got == [list(pair) for pair in t.expected],
                 "your pairs are lists [distance, label]. Make them tuples: (distance, label)."),
        Detector(lambda t: isinstance(t.got, list) and isinstance(t.expected, list)
                 and len(t.got) != len(t.expected),
                 "you returned the wrong number of neighbours. Keep exactly k of them: slice the sorted list "
                 "with [:k] (and use the parameter k, not 3)."),
        Detector(lambda t: isinstance(t.got, list) and t.got and all(type(p) is int for p in t.got),
                 "you returned only labels (or only distances). The spec asks for (distance, label) pairs."),
    ],
    returns=list,
)


# ============================================================================= C3 · majority_vote
C3 = Task(
    "C3", "majority_vote", title="Let the neighbours vote",
    practise="counting with a dict, and max() with a key (Theory S3–S4)",
    minimum_path=True,
    input_labels=("Neighbours",), expected_label="Expected label",
    cases=[
        Case(([(3, 7), (5, 1), (9, 7)],), 7, highlight=True, title="Two 7s beat one 1"),
        Case(([(1, 6), (2, 4), (3, 9), (4, 4), (5, 1)],), 4,
             title="Two 4s beat a 6, a 9 and a 1: the winner is not the closest, the largest or the farthest label"),
        Case(([(1, 6), (2, 4), (3, 4), (4, 9)],), 4,
             title="Two 4s beat a 6 and a 9: the winner is neither the closest label nor the largest"),
        Case(([(2, 4), (5, 1)],), 4, highlight=True, title="A 1-1 tie: the closer neighbour (label 4) wins"),
        Case(([(1, 3), (2, 8), (4, 8), (6, 3)],), 3, highlight=True,
             title="A 2-2 tie: the closest of the tied neighbours wins (label 3, at distance 1)"),
        Case(([(3, 5)],), 5, title="A single neighbour decides alone"),
        Case(([(1, 2), (2, 6), (3, 6), (4, 2), (5, 6)],), 6, title="Five neighbours: three 6s beat two 2s"),
    ],
    detectors=[
        Detector(lambda t: t.raised(TypeError) and _key_line(t),
                 "key= needs a function, with no brackets after it: key=counts.get. max() calls it on each label to "
                 "get that label's count; counts.get(label), counts.get() or counts on its own is not that function."),
        Detector(lambda t: t.raised(KeyError),
                 "the first vote for a label finds no key yet. Use counts[label] = counts.get(label, 0) + 1."),
        Detector(lambda t: t.raised(ValueError) and "unpack" in str(t.error),
                 "each neighbour is a pair (distance, label): unpack it with for distance, label in neighbours."),
        Detector(lambda t: isinstance(t.got, tuple),
                 "you returned a whole neighbour (distance, label). Return just the winning label."),
        Detector(lambda t: isinstance(t.got, dict),
                 "you returned the counts. Return the winner: the key with the biggest count, "
                 "max(counts, key=counts.get)."),
        Detector(lambda t: isinstance(t.got, list),
                 "you returned a list. Return one label, an int such as 7."),
        Detector(lambda t: is_number(t.got) and t.got not in _labels(t)
                 and t.got in [d for d, _label in t.args[0]],
                 "you counted the distances. Each neighbour is (distance, label): the label is the second item, "
                 "so unpack it as for distance, label in neighbours."),
        Detector(lambda t: _labels(t) and t.got != t.expected and t.got != _labels(t)[0]
                 and t.got == min(_counts(t), key=_counts(t).get),
                 "min() returns the label with the FEWEST votes. The winner has the most: "
                 "max(counts, key=counts.get)."),
        Detector(lambda t: is_number(t.got) and t.got not in _labels(t) and t.got == max(_counts(t).values()),
                 "you returned how many votes the winner got. Return the label itself: max(counts, "
                 "key=counts.get) gives the key, not the count."),
        Detector(lambda t: len(_tied(t)) > 1 and t.got == min(_tied(t)) != _tied(t)[-1] and t.got != t.expected,
                 "a tie went to the smaller label. Count in neighbour order (closest first, as they come) with a "
                 "plain dict: it keeps the order in which labels first appear, and max() returns the FIRST key "
                 "with the top count. set() or sorted() lose that order."),
        Detector(lambda t: len(_tied(t)) > 1 and t.got == _tied(t)[-1] != min(_tied(t)) and t.got != t.expected,
                 "a tie went to the farther neighbour. Sorting the labels by count and taking the last one does "
                 "that. max(counts, key=counts.get) returns the FIRST key with the top count: on a dict filled "
                 "closest first, that is the closest tied neighbour."),
        Detector(lambda t: len(_tied(t)) > 1 and t.got in _tied(t) and t.got != t.expected,
                 "a tie went to the wrong label: the spec says the closest of the tied neighbours wins. Fill a "
                 "dict in neighbour order (closest first) and use max(counts, key=counts.get): it returns the "
                 "FIRST key with the top count. (set() and sorted() lose that order; sorting by count and taking "
                 "the last one picks the farthest.)"),
        Detector(lambda t: _labels(t) and t.got == _labels(t)[-1] != t.expected and t.got != max(_labels(t)),
                 "only the last neighbour's vote survived. Is counts = {} inside the for loop? Create the dict "
                 "once, before the loop."),
        Detector(lambda t: _labels(t) and t.got == max(_labels(t)) and t.got != t.expected,
                 "you returned the largest label: max(counts) (or max of the labels) compares the labels "
                 "themselves. You want the label with the most votes: max(counts, key=counts.get)."),
        Detector(lambda t: _labels(t) and t.got == _labels(t)[0] and t.got != t.expected,
                 "you returned the label of the closest neighbour. Let every neighbour vote, then count."),
    ],
    returns=int,
)


# ============================================================================= C4 · predict_knn
C4 = Task(
    "C4", "predict_knn", title="Read a digit: neighbours, then a vote",
    practise="composing your own functions (Theory S1)",
    minimum_path=True,
    depends_on=("C1", "C2", "C3"),
    input_labels=("Image", "Training images", "Training labels"), expected_label="Expected label",
    cases=[
        # distances from [5]: 25, 1, 1, 4, 625 -> nearest three: labels 1, 9, 9
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), 9, kwargs={"k": 3}, highlight=True,
             title="k = 3: the three nearest say 1, 9, 9, so 9 wins the vote"),
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), 1, kwargs={"k": 1},
             title="k = 1: plain nearest neighbour (two images tie at distance 1: the earlier one)"),
        Case(([5], [[4], [7], [0]], [2, 6, 3]), 2, kwargs={"k": 2}, highlight=True,
             title="A 1-1 vote: the closer neighbour wins"),
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), 9, title="k = 3 by default"),
    ] + [_golden(i, "predict_knn", "", highlight=i == 0) for i in (0, 1, 2, 3)],
    detectors=[
        Detector(lambda t: t.raised(TypeError) and "'function' object is not" in str(t.error),
                 "you passed the function k_nearest itself to majority_vote. Pass its result instead: "
                 "majority_vote(k_nearest(image, train_images, train_labels, k))."),
        Detector(lambda t: isinstance(t.got, list),
                 "you returned the neighbours. Pass them to majority_vote and return its answer: "
                 "majority_vote(k_nearest(...))."),
        Detector(lambda t: isinstance(t.got, tuple),
                 "you returned one (distance, label) pair. Return the winning label."),
        Detector(lambda t: t.kwargs.get("k") == 1 and t.got == 9,
                 "with k = 1 you still asked 3 neighbours: pass k on to k_nearest(image, train_images, "
                 "train_labels, k)."),
        Detector(lambda t: t.got is None and t.error is None,
                 "your function returned None: write return in front of majority_vote(...)."),
    ],
    returns=int,
)


# ============================================================================= C5 · test_majority_vote (judged)
def _judge(test_fn, ctx, *, param: str, global_name: str, correct, mutants_: dict, must_catch: int,
           min_inputs: int = 3, must_fail_on: dict | None = None, all_of: dict | None = None,
           score_on: dict | None = None) -> None:
    """Judge a test function written by a student (blueprint §4.4 "Rules for testing student tests").

    In order: no always-true assert (a, b); it calls `param`, not the global `global_name`; it passes on
    `correct`; it has an assert and checks at least `min_inputs` inputs (counted as calls of `param` while it
    runs on `correct`, so one assert in a loop over 4 cases counts 4, like 4 assert lines); it fails on every
    function in `must_fail_on`; it fails on every function in `all_of`; and it fails on at least `must_catch`
    of `mutants_` (survivors named in words). `score_on` adds a note with the score on more buggy versions.
    """
    name = getattr(test_fn, "__name__", "your test")
    asserts, tuple_line = _read_asserts(test_fn)
    if asserts is not None:
        if not ctx.require(tuple_line is None, case=f"{name}" + (f", line {tuple_line}" if tuple_line else ""),
                           title="Every assert can fail (no assert (a, b), which is always true)",
                           expected='assert a == b, "message"', got=("assert (...): a tuple, always true"
                                                                     if tuple_line else 'assert a == b, "message"'),
                           cause="Python reads assert (a, b) as 'assert this tuple', and a tuple that isn't empty "
                                 "is always true. Write assert a == b, \"message\" with no brackets around both."):
            return
    globals_ = getattr(test_fn, "__globals__", {})
    missing = object()
    saved = globals_.get(global_name, missing)
    calls: list = []
    inputs: list = []

    def recorder(*args, **kwargs):
        calls.append(args)
        return correct(*args, **kwargs)

    def counted(*args, **kwargs):              # the version the test receives: count what it checks
        inputs.append(args)
        return correct(*args, **kwargs)

    globals_[global_name] = recorder
    try:
        attempt = ctx.call(test_fn, counted, show=f"{name}({param}=a correct {global_name})", track=False)
    finally:
        if saved is missing:
            globals_.pop(global_name, None)
        else:
            globals_[global_name] = saved
    if attempt.interrupted:
        ctx.expect(attempt, None)
        return
    if not ctx.require(not calls, case=attempt.show, title=f"Your test calls {param}, not {global_name}",
                       expected=f"every assert calls {param}(...)",
                       got=f"your test calls {global_name}(...) directly" if calls else f"it calls {param}(...)",
                       cause=f"your test calls {global_name} directly, so it tests the same function whatever you "
                             f"pass in. Call {param}(...) instead: check() passes a different version each time."):
        return
    message = f": {attempt.error_text}" if attempt.error is not None else ""
    if not ctx.require(attempt.error is None, case=attempt.show,
                       title=f"Your test passes on a correct {global_name}", expected="no assert fails",
                       got=f"it failed{message}" if attempt.error is not None else "no assert fails",
                       cause="one of your asserts expects a wrong answer, or the test itself raises an error. Work "
                             "that input out by hand (the neighbours come closest first; a tie goes to the closest "
                             "tied neighbour)."):
        return
    n_inputs = len(inputs)
    no_assert = asserts == 0
    if no_assert:
        got = "no assert"
        cause = (f"your test has no assert, so it can never fail. Write assert {param}(...) == ..., \"message\" "
                 "for each input.")
    else:
        got = f"{n_inputs} call{'s' if n_inputs != 1 else ''} of {param}"
        cause = (f"your test checks only {n_inputs} input{'s' if n_inputs != 1 else ''}. Add more: one assert line "
                 f"per input, or one assert inside a loop over {min_inputs} or more cases (every call of {param} "
                 "counts).")
    if not ctx.require(not no_assert and n_inputs >= min_inputs, case=attempt.show,
                       title=f"Your test checks at least {min_inputs} inputs with asserts",
                       expected=f"at least {min_inputs} calls of {param}, checked by asserts",
                       got=got if (no_assert or n_inputs < min_inputs) else f"{n_inputs} calls of {param}",
                       cause=cause):
        return
    for label, buggy in (must_fail_on or {}).items():
        result = ctx.call(test_fn, buggy, show=f"{name}({param}={label})", track=False)
        if result.interrupted:
            return
        if not ctx.require(result.error is not None, case=result.show, title=f"Your test fails on {label}",
                           expected="at least one assert fails",
                           got="an assert failed" if result.error is not None else "every assert passed",
                           cause=f"your asserts never give {label} an input where its bugs show. Run it by hand on "
                                 "a tiny example and compare with the right answer."):
            return
    caught, survived = [], []
    for description, buggy in mutants_.items():
        result = ctx.call(test_fn, buggy, show=f"{name}(a buggy version)", track=False)
        if result.interrupted:
            return
        (caught if result.error is not None else survived).append(description)
    if all_of:
        missed = [d for d in all_of if d in survived]
        if not ctx.require(not missed, case=f"{name} on {len(all_of)} buggy versions, one per AI bug",
                           title=f"Your test catches each of the {len(all_of)} AI bugs on its own",
                           expected=f"fails on all {len(all_of)}", got=f"survived: {'; '.join(missed)}" if missed
                           else f"fails on all {len(all_of)}",
                           cause="write one assert per bug: an input where THAT bug alone gives a wrong answer."):
            return
    if len(caught) >= must_catch:            # on NOT YET, the failing test below names the survivors
        ctx.note(f"Buggy versions caught: {len(caught)} of {len(mutants_)}"
                 + (f" · survived: {'; '.join(survived)}" if survived else ""))
    if score_on and score_on is not mutants_:
        extra_caught = 0
        for buggy in score_on.values():
            result = ctx.call(test_fn, buggy, track=False)
            extra_caught += result.error is not None
        ctx.note(f"Mutation score: {extra_caught} of {len(score_on)} buggy readers caught.")
    ctx.require(len(caught) >= must_catch, case=f"{name} on {len(mutants_)} buggy versions of {global_name}",
                title=f"Your test catches at least {must_catch} of {len(mutants_)} buggy versions",
                expected=f"fails on at least {must_catch}",
                got=f"fails on {len(caught)}" + (f" (survived: {'; '.join(survived)})" if survived else ""),
                cause="add an assert for each survivor: which input makes that buggy version give a wrong answer? "
                      "(A buggy tie rule only shows on a tie.)")


def _c5_judge(fn, ctx) -> None:
    correct = ctx.reference("majority_vote")
    if correct is None:
        ctx.not_started("This check needs labkit/reference/lab2.py: re-extract the kit.")
        return
    _judge(fn, ctx, param="vote_fn", global_name="majority_vote", correct=correct,
           mutants_=mutants.VOTE_MUTANTS, must_catch=3, min_inputs=3)


C5 = Task(
    "C5", "test_majority_vote", kind="test-function", title="A test that catches bugs",
    practise="assert, and testing a function you pass in (a new tool)",
    custom=_c5_judge, quality=False, show_on_pass=5,
)


# ============================================================================= AI · knn_predict_ai
# The likely causes of the AI activity describe the symptom and ask a question: finding the line is the
# student's job (the exact lines and fixes are in "What we found" and in the solution notebook).
_BUG_MESSAGES = {
    1: "the nearest image lost to images that are farther away but have smaller labels. What does pairs.sort() "
       "compare first in each pair, and is that the distance?",
    2: "a training image identical to the test image (distance 0) never got a vote. Which line decides which "
       "neighbours vote, and does it start with the nearest one?",
    3: "two votes lost to one: the answer is the biggest label, not the most common one. What does max() compare "
       "when you give it a dict?",
}
_AI_TIE = ([5], [[7], [3], [40]], [6, 1, 2])          # [7] and [3] are both 4 away: the earlier image must win


def _ai_bugs_left(t) -> int | None:
    """Which AI bug is (certainly) still in this version: run the sample with every combination of its three
    bugs on the failing case and see which combinations give the same answer."""
    if t.error is not None or not t.args:
        return None
    matching = []
    for size in (1, 2, 3):
        for bugs in itertools.combinations((1, 2, 3), size):
            try:
                if mutants.ai_variant(set(bugs), *t.args, **t.kwargs) == t.got:
                    matching.append(set(bugs))
            except Exception:
                continue
    if not matching:
        return None
    certain = set.intersection(*matching)
    return min(certain) if certain else min(min(bugs) for bugs in matching)


AI = Task(
    "AI", "knn_predict_ai", title="AI wrote this: find the planted bugs",
    practise="reading code critically, and proving a fix with a check",
    reference="predict_knn", rescue=False, quality=False,
    input_labels=("Image", "Training images", "Training labels"), expected_label="Expected label",
    cases=[
        Case(([5], [[5], [9], [40]], [1, 2, 2]), 1, kwargs={"k": 1}, highlight=True,
             title="k = 1: a training image identical to the test image (distance 0) must win"),
        Case(([5], [[90], [4], [80]], [1, 7, 3]), 7, kwargs={"k": 1}, highlight=True,
             title="k = 1: the nearest image is [4] (distance 1), label 7"),
        Case(([5], [[4], [6], [7], [50]], [8, 2, 2, 9]), 2, kwargs={"k": 3}, highlight=True,
             title="k = 3: the three nearest say 8, 2, 2, so 2 wins the vote"),
        Case(_AI_TIE, 6, kwargs={"k": 1},
             title="k = 1: [7] and [3] are both 4 away: the earlier image (label 6) wins, as in C2"),
    ] + [_golden(i, "knn_predict_ai", "") for i in (0, 1)],
    detectors=[
        Detector(lambda t: tuple(t.args) == _AI_TIE and t.got == 1,
                 "two training images are equally far away, and the one with the smaller label won. When two "
                 "distances are equal, what does your sort compare next? (C2's tie rule: the earlier image wins.)",
                 next_step=_AI_NEXT),
        Detector(lambda t: _ai_bugs_left(t) == 2, _BUG_MESSAGES[2],
                 next_step=_AI_NEXT),
        Detector(lambda t: _ai_bugs_left(t) == 1, _BUG_MESSAGES[1],
                 next_step=_AI_NEXT),
        Detector(lambda t: _ai_bugs_left(t) == 3, _BUG_MESSAGES[3],
                 next_step=_AI_NEXT),
        Detector(lambda t: is_number(t.got) and len(t.args) > 2 and t.got not in t.args[2],
                 "your answer isn't one of the training labels: the votes count distances. If you changed the "
                 "pairs to (distance, label), unpack them the same way where the votes are counted: for distance, "
                 "label in nearest.",
                 next_step=_AI_NEXT),
        Detector(lambda t: t.raised(ValueError) and "empty" in str(t.error),
                 "max() got an empty dict of votes: no neighbour voted. Look at the slice that picks the "
                 "nearest neighbours.",
                 next_step=_AI_NEXT),
    ],
    returns=int,
)


# ============================================================================= C6 · normalise (OPTIONAL)
def _floats_like(t) -> bool:
    """The student's result and the expected one are lists of numbers of the same length, and they differ."""
    got, expected = t.got, t.expected
    return (isinstance(got, list) and isinstance(expected, list) and len(got) == len(expected) and got != expected
            and all(is_number(g) and is_number(e) for g, e in zip(got, expected)))


def _c6_rounded(t) -> bool:
    """Every value equals the expected one rounded to 0-6 decimals (round(pixel / 255, 3) and the like)."""
    return _floats_like(t) and any(all(g == round(e, d) for g, e in zip(t.got, t.expected)) for d in range(7))


def _c6_last_decimals(t) -> bool:
    """Every value is within 1e-9 of the expected one: the same maths, rounded differently (p * (1 / 255))."""
    return _floats_like(t) and all(abs(g - e) <= 1e-9 for g, e in zip(t.got, t.expected))


C6 = Task(
    "C6", "normalise", title="Scale the pixels: a list comprehension",
    practise="a list comprehension (Theory S4)",
    optional=True, not_needed_later=True,
    input_labels=("Image",), expected_label="Expected pixels",
    cases=[
        Case(([0, 51, 255],), [0.0, 0.2, 1.0], compare=approx(1e-12), highlight=True,
             title="Three pixels: paper 0 becomes 0.0, ink 255 becomes 1.0"),
        Case(([],), [], title="An empty image gives an empty list"),
        Case(([128, 64],), [128 / 255, 64 / 255], compare=approx(1e-12), title="Values in between: 128 / 255 and 64 / 255"),
    ] + [Case(lambda i=i: (TEST.images[i],), key=f"real-{i}", show=f"normalise(test_images[{i}])", highlight=i == 0,
              title=f"The same answer as legacy_reader.normalise: test image #{i}") for i in (0, 1)],
    detectors=[
        Detector(lambda t: hasattr(t.got, "__next__"),
                 "you returned a generator: round brackets make a recipe, square brackets make the list. Write "
                 "[pixel / 255 for pixel in image]."),
        Detector(lambda t: isinstance(t.got, list) and t.got and all(type(v) is int for v in t.got),
                 "every value is an int (0 or 1): // divides and rounds down. Use / to keep the fractions."),
        Detector(lambda t: isinstance(t.got, list) and t.args and len(t.got) == len(t.args[0]) and t.args[0]
                 and t.got == [p * 255 for p in t.args[0]],
                 "you multiplied by 255. Divide instead: pixel / 255."),
        Detector(lambda t: isinstance(t.got, tuple),
                 "you returned a tuple. Square brackets make a list: [pixel / 255 for pixel in image]."),
        Detector(lambda t: type(t.got).__name__ == "map",
                 "you returned a map object. Use a list comprehension, or wrap the map in list(...)."),
        Detector(_c6_rounded,
                 "your values are rounded. Don't round: keep every decimal of pixel / 255 (the legacy code doesn't "
                 "round, and a refactor must give exactly the same answers)."),
        Detector(_c6_last_decimals,
                 "your values differ from the legacy ones only in the last decimal places: p * (1 / 255) rounds "
                 "differently from p / 255. A refactor must give exactly the same answers: divide by 255."),
    ],
    returns=list,
)


# ============================================================================= ENG1 · fast_distance (OPTIONAL)
def _order_50(fn) -> list[int]:
    """Training images 0-49 sorted by their distance to test image #0 (equal distances: training order)."""
    image = TEST.images[0][:]
    return sorted(range(50), key=lambda i: fn(image, TRAIN.images[i][:]))


ENG1 = Task(
    "ENG1", "fast_distance", title="Measure, don't guess",
    practise="math.dist from the standard library, and profiling before optimising",
    input_labels=("Image a", "Image b"), expected_label="Expected distance",
    cases=[
        Case(([0, 0], [3, 4]), 5.0, compare=approx(1e-9), highlight=True,
             title="Two tiny images: the square root of 9 + 16 is 5.0"),
        Case(([1, 2, 3], [1, 2, 3]), 0.0, compare=approx(1e-12), title="Identical images are 0.0 apart"),
        Case(lambda: (TEST.images[0], TRAIN.images[0]), key="real-0", compare=approx(1e-6),
             title="Two real images: the square root of their squared distance",
             show="fast_distance(test_images[0], train_images[0])"),
        Case(call=_order_50, key="order", expected_label="Expected order", highlight=True,
             title="The same ORDER as squared_distance: training images 0-49 sorted by distance to test image #0",
             show="sorted(range(50), key=lambda i: fast_distance(test_images[0], train_images[i]))"),
    ],
    detectors=[
        Detector(lambda t: is_number(t.got) and is_number(t.expected) and t.expected > 0
                 and abs(t.got - t.expected ** 2) < 1e-6 * max(1, t.expected ** 2),
                 "that is the squared distance. math.dist(a, b) returns its square root: the straight-line "
                 "distance. The order of the neighbours stays the same, which is all k-NN needs."),
        Detector(lambda t: t.raised(TypeError) and "dist" in str(t.error),
                 "math.dist(a, b) takes the two images: two lists of the same length."),
        Detector(lambda t: t.raised(NameError) and "math" in str(t.error),
                 "Python doesn't know math yet: write import math at the top of the cell (math is in the "
                 "standard library: nothing to install)."),
    ],
    returns=float,
    speed=Speed(lambda: (TEST.images[0], TRAIN.images[0]), budget_s=25e-6,
                note="math.dist runs its loop in C; a loop in Python, or sum() over zip(), is several times slower."),
)


# ============================================================================= SCI1 · downsample (OPTIONAL)
def _toy_block(values: tuple[int, int, int, int]) -> list[int]:
    """A 28 x 28 image of paper whose top-left 2 x 2 block holds `values`."""
    image = [0] * 784
    image[0], image[1], image[28], image[29] = values
    return image


def _is_labkit_downsample(fn) -> bool:
    from ..data import downsample as provided
    return fn is provided


def _sci1_own(fn, ctx) -> None:
    ctx.require(not _is_labkit_downsample(fn), case="downsample", title="It is your own downsample",
                expected="a function you wrote", got="labkit's downsample, imported",
                cause="labkit already has a downsample (the provided one); the task is to write your own. Run the "
                      "cell with your def again, so that the name means your function.")


SCI1 = Task(
    "SCI1", "downsample", title="Shrink the image",
    practise="a list comprehension with two for clauses, and floor division (Theory S4)",
    input_labels=("Image",), expected_label="Expected 14 x 14 image",
    cases=[
        Case(([4] * 784,), [4] * 196, title="A flat grey image stays flat: 784 pixels of 4 become 196 of 4"),
        Case((_toy_block((1, 2, 3, 6)),), [3] + [0] * 195, highlight=True,
             title="One block of 1, 2, 3, 6 (top-left corner): (1 + 2 + 3 + 6) // 4 = 3",
             show="downsample(an image of paper whose top-left 2 x 2 block is 1, 2 / 3, 6)"),
        Case((_toy_block((1, 2, 3, 5)),), [2] + [0] * 195, highlight=True,
             title="The floor of 2.75 is 2: (1 + 2 + 3 + 5) // 4",
             show="downsample(an image of paper whose top-left 2 x 2 block is 1, 2 / 3, 5)"),
    ] + [Case(lambda i=i: (TEST.images[i],), key=f"real-{i}", show=f"downsample(test_images[{i}])",
              highlight=i == 0, title=f"The same as the kit's 14 x 14 file: test image #{i}") for i in (0, 1)],
    detectors=[
        Detector(lambda t: isinstance(t.got, list) and len(t.got) == 784,
                 "you returned 784 pixels: the image didn't shrink. Make one new pixel per 2 x 2 block: 14 x 14 "
                 "= 196 of them."),
        Detector(lambda t: isinstance(t.got, list) and len(t.got) == 196 and any(isinstance(v, float) for v in t.got),
                 "your pixels are floats: / divides exactly. The canonical rule rounds down with //: "
                 "(a + b + c + d) // 4."),
        Detector(lambda t: isinstance(t.got, list) and t.got[:1] == [3] and t.expected[:1] == [2],
                 "you rounded 2.75 up to 3. The canonical rule rounds down: (a + b + c + d) // 4."),
        Detector(lambda t: isinstance(t.got, list) and t.got[:1] == [1] and t.expected[:1] in ([3], [2]),
                 "you copied the top-left pixel of each block instead of averaging the four: add image[i], "
                 "image[i + 1], image[i + 28] and image[i + 29]."),
        Detector(lambda t: isinstance(t.got, list) and len(t.got) == 196 and t.got != t.expected
                 and t.got[:1] in ([(1 + 2 + 3 + 0) // 4], [(1 + 2 + 0 + 0) // 4], [(1 + 3) // 4]),
                 "the four pixels of a block are image[i], image[i + 1] (to the right), image[i + 28] (the row "
                 "below) and image[i + 29]: a row of the big image is 28 pixels long."),
        Detector(lambda t: isinstance(t.got, list) and len(t.got) not in (784, 196),
                 "a 14 x 14 image has 196 pixels: one per 2 x 2 block. Step through the rows and columns two at "
                 "a time: range(0, 28, 2)."),
    ],
    returns=list,
    custom=_sci1_own,
)


# ============================================================================= BLD1 · per_digit_accuracy + report_card
def _card(first, acc: dict):
    return _call_partner(first, "report_card", dict(acc))


def _card_case(t) -> bool:
    return t.show.startswith("report_card")


def _bld1_by_prediction(t) -> bool:
    """The accuracies the student would get by grouping the images by the predicted digit instead of the true one."""
    if _card_case(t) or not isinstance(t.got, dict) or len(t.args) != 2 or t.got == t.expected:
        return False
    predictions, truth = t.args
    wrong: dict = {}
    for digit in set(predictions):
        read_as = [(p, d) for p, d in zip(predictions, truth) if p == digit]
        wrong[digit] = sum(1 for p, d in read_as if p == d) / len(read_as)
    return set(t.got) == set(wrong) and all(is_number(t.got[k]) and abs(t.got[k] - wrong[k]) < 1e-9 for k in wrong)


_CARD_A = {7: 0.5, 1: 1.0}
_CARD_B = {3: 0.9, 8: 0.75, 5: 0.9}
_CARD_A_TEXT = "Rank  Digit  Accuracy\n   1      7     50.0%\n   2      1    100.0%"
_CARD_B_TEXT = "Rank  Digit  Accuracy\n   1      8     75.0%\n   2      3     90.0%\n   3      5     90.0%"


def _card_rows(text: object) -> list | None:
    """The (digit, accuracy) words of each table line, without the header and the rank."""
    if not isinstance(text, str):
        return None
    return [line.split()[1:] for line in text.split("\n")[1:]]


BLD1 = Task(
    "BLD1", "per_digit_accuracy", title="Report card",
    practise="a dict comprehension, and an aligned f-string table with enumerate(start=1) (Theory S4)",
    input_labels=("Predictions", "Ground truth"), expected_label="Expected accuracy per digit",
    cases=[
        Case(([7, 1, 1, 3], [7, 7, 1, 3]), {1: 1.0, 3: 1.0, 7: 0.5}, compare=approx(1e-9), highlight=True,
             title="per_digit_accuracy: the two 7s are read as 7 and 1, so digit 7 gets 0.5"),
        Case(([2, 2, 2], [2, 5, 5]), {2: 1.0, 5: 0.0}, compare=approx(1e-9),
             title="per_digit_accuracy: a digit never read correctly still gets a line (0.0)"),
        Case(([], []), {}, title="per_digit_accuracy with empty lists: an empty dict"),
        Case(call=lambda first: _card(first, _CARD_A), expected=_CARD_A_TEXT, key="card-a", highlight=True,
             expected_label="Expected table", title="report_card: worst digit first, ranks from 1",
             show="report_card({7: 0.5, 1: 1.0})"),
        Case(call=lambda first: _card(first, _CARD_B), expected=_CARD_B_TEXT, key="card-b",
             expected_label="Expected table", title="report_card: equal accuracies, the smaller digit first",
             show="report_card({3: 0.9, 8: 0.75, 5: 0.9})"),
    ],
    detectors=[
        Detector(lambda t: _card_case(t) and _partner_state(t, "report_card") == "missing",
                 "per_digit_accuracy works. Now write report_card (the cell below it), run that cell, then run "
                 "this check again."),
        Detector(lambda t: _card_case(t) and _partner_state(t, "report_card") == "stub",
                 "per_digit_accuracy works; report_card is still a stub. Write it, run its cell, then run this "
                 "check again."),
        Detector(lambda t: t.raised(ZeroDivisionError),
                 "a digit with no images: you divided by 0. Take the digits from the ground truth (each one has at "
                 "least one image), not from the predictions."),
        Detector(lambda t: not _card_case(t) and isinstance(t.got, dict) and len(t.args) == 2
                 and set(t.got) == set(t.args[0]) != set(t.args[1]),
                 "your keys come from the predictions. Judge each digit on the images whose TRUE label it is: "
                 "for digit in set(truth)."),
        Detector(_bld1_by_prediction,
                 "you grouped the images by the PREDICTED digit. Each digit is judged on the images whose TRUE label "
                 "it is: for digit in set(truth), count the images with t == digit and how many of them were read "
                 "right."),
        Detector(lambda t: not _card_case(t) and isinstance(t.got, dict) and t.got
                 and all(type(v) is int for v in t.got.values()) and t.got != t.expected,
                 "these are counts of right answers. Divide each one by how many images of that digit there are."),
        Detector(lambda t: not _card_case(t) and isinstance(t.got, dict) and isinstance(t.expected, dict)
                 and set(t.got) == set(t.expected)
                 and all(is_number(t.got[k]) and abs(t.got[k] - 100 * t.expected[k]) < 1e-6 for k in t.expected),
                 "these are percentages. The spec asks for fractions from 0.0 to 1.0; the table formats them."),
        Detector(lambda t: _card_case(t) and t.got != t.expected and _card_rows(t.got) is not None
                 and _card_rows(t.got) == (_card_rows(t.expected) or [])[::-1],
                 "the best digit comes first. The report card starts with the WORST digit: sort by accuracy, "
                 "smallest first (no reverse=True)."),
        Detector(lambda t: _card_case(t) and isinstance(t.got, str) and t.got.rstrip("\n") == t.expected
                 and t.got != t.expected,
                 "there is a newline at the end. Join the lines with \"\\n\".join(lines)."),
        Detector(lambda t: _card_case(t) and isinstance(t.got, str) and isinstance(t.expected, str)
                 and t.got.split("\n")[1:2] and t.got.split("\n")[1].strip().startswith("0"),
                 "the ranks start at 0. enumerate(..., start=1) counts from 1."),
        Detector(lambda t: _card_case(t) and isinstance(t.got, str) and "%" not in t.got,
                 "show the accuracy as a percentage with one decimal: f\"{value:>8.1%}\" turns 0.5 into "
                 "'   50.0%'."),
        Detector(lambda t: _card_case(t) and isinstance(t.got, str) and isinstance(t.expected, str)
                 and not t.got.startswith("Rank"),
                 "the first line is the header: f\"{'Rank':>4}  {'Digit':>5}  {'Accuracy':>8}\"."),
        Detector(lambda t: _card_case(t) and isinstance(t.got, str) and isinstance(t.expected, str)
                 and [line.split() for line in t.got.split("\n")] == [line.split() for line in t.expected.split("\n")],
                 "the right words in the right order, but the columns don't line up. Use the widths of the card: "
                 "{rank:>4}, two spaces, {digit:>5}, two spaces, {accuracy:>8.1%}."),
    ],
    returns=dict,
)


# ============================================================================= ENG2 · test_knn (ADVANCED, judged)
def _eng2_judge(fn, ctx) -> None:
    correct = ctx.reference("predict_knn")
    if correct is None:
        ctx.not_started("This check needs labkit/reference/lab2.py: re-extract the kit.")
        return
    from ..ai_samples.lab2_knn_ai import knn_predict_ai
    _judge(fn, ctx, param="predict_fn", global_name="predict_knn", correct=correct,
           mutants_=mutants.PREDICT_MUTANTS, must_catch=3, min_inputs=3,
           must_fail_on={"the AI's knn_predict_ai (all three bugs)": knn_predict_ai},
           all_of=mutants.AI_BUG_MUTANTS)


ENG2 = Task(
    "ENG2", "test_knn", kind="test-function", title="Bugs become tests",
    practise="regression tests: one assert per bug, and how many buggy readers they catch",
    rescue=False, custom=_eng2_judge, quality=False, show_on_pass=6,
)


# ============================================================================= SCI2 · cross_validate (ADVANCED)
# Literal expected values (there is no reference for ADVANCED tasks in the kit); the instructor file
# teacher_only/solutions/lab2_advanced.py checks that its solution PASSes these cases.
_CV_TOY = ([[7], [18], [17], [4], [11], [19]], [4, 7, 7, 1, 7, 1])


def _cv_leaked(t) -> bool:
    """With k = 1 every fold scores 1.0: the validation images were also in the part the reader learns from."""
    got, expected = t.got, t.expected
    return (isinstance(got, dict) and isinstance(expected, dict) and isinstance(got.get(1), list) and got.get(1)
            and all(v == 1.0 for v in got[1]) and got[1] != expected.get(1))


SCI2 = Task(
    "SCI2", "cross_validate", title="Choose k with cross-validation",
    practise="slicing, lists joined with +, and a dict of lists",
    rescue=False, quality=False,
    input_labels=("Images", "Labels"), expected_label="Expected scores",
    cases=[
        Case((_CV_TOY[0], _CV_TOY[1]), {1: [0.5, 0.5, 0.0], 3: [0.5, 0.5, 0.5]}, kwargs={"ks": (1, 3), "folds": 3},
             compare=approx(1e-9), highlight=True,
             title="A tiny world: 6 one-pixel images, 3 folds of 2 images, ks = (1, 3)"),
        Case(lambda: (TRAIN14.images[:60], TRAIN14.labels[:60]),
             {1: [0.55, 0.7, 0.75], 3: [0.55, 0.65, 0.65], 5: [0.6, 0.65, 0.55]},
             kwargs={"ks": (1, 3, 5), "folds": 3}, compare=approx(1e-9), key="real-60", highlight=True,
             title="Real 14 x 14 training images 0-59: 3 folds of 20 images, ks = (1, 3, 5)",
             show="cross_validate(the first 60 training images at 14 x 14, their labels, ks=(1, 3, 5), folds=3)"),
    ],
    detectors=[
        Detector(lambda t: isinstance(t.got, dict) and t.got and all(is_number(v) for v in t.got.values()),
                 "cross_validate returns a LIST of accuracies for each k, one per fold: {k: [fold 0, fold 1, ...]}. "
                 "One number per k is what a single validation split gives."),
        Detector(_cv_leaked,
                 "with k = 1 every fold scores 100%: the validation images are also in the part the reader learns "
                 "from, so each one finds itself at distance 0. Learn from images[:start] + images[stop:] only."),
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.expected, dict) and set(t.got) == set(t.expected)
                 and all(isinstance(v, list) and v and all(is_number(x) and x > 1 for x in v) for v in t.got.values()),
                 "your scores are counts. Divide each one by the number of images in the fold: right / size."),
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.expected, dict) and set(t.got) == set(t.expected)
                 and any(isinstance(v, list) and len(v) != len(t.expected[k]) for k, v in t.got.items()),
                 "one accuracy per fold: each list needs `folds` scores, one for every fold in range(folds)."),
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.expected, dict) and set(t.got) != set(t.expected),
                 "the keys of your dict are not the ks you were given: one key per k in ks, e.g. {1: [...], 3: [...]}."),
    ],
    returns=dict,
    # The speed trick (k_nearest once per image with max(ks)) takes about 0.05 s here on the build laptop with
    # Python 3.14; calling predict_knn for every k takes about 0.28 s (measured 2026-10-12): the note tells them.
    speed=Speed(lambda: (TRAIN14.images[:60], TRAIN14.labels[:60]), budget_s=0.13,
                note="On the 500 images of the cross-validation cell that means up to a minute. The speed trick on "
                     "the card (k_nearest once per image with max(ks), then neighbours[:k] for each k) is about 5 "
                     "times faster."),
)


# ============================================================================= BLD2 · shift (+ to_grid, flatten)
_TOY3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]       # a 3 x 3 image: rows 1 2 3 / 4 5 6 / 7 8 9


BLD2 = Task(
    "BLD2", "shift", title="Shift it",
    practise="nested lists, slicing and a nested comprehension",
    rescue=False, quality=False,
    input_labels=("Image",), expected_label="Expected image",
    cases=[
        Case((_TOY3[:], 1, 0), [0, 1, 2, 0, 4, 5, 0, 7, 8], kwargs={"width": 3}, highlight=True,
             title="shift: one pixel to the right (dx = 1); the empty column on the left is paper (0)"),
        Case((_TOY3[:], 0, -1), [4, 5, 6, 7, 8, 9, 0, 0, 0], kwargs={"width": 3}, highlight=True,
             title="shift: one pixel up (dy = -1); the empty row at the bottom is paper"),
        Case((_TOY3[:], -1, 1), [0, 0, 0, 2, 3, 0, 5, 6, 0], kwargs={"width": 3},
             title="shift: left and down at the same time"),
        Case((_TOY3[:], 0, 0), _TOY3[:], kwargs={"width": 3}, title="shift by (0, 0) changes nothing"),
        Case(call=lambda first: _call_partner(first, "to_grid", [1, 2, 3, 4, 5, 6], width=3), key="grid",
             expected=[[1, 2, 3], [4, 5, 6]], expected_label="Expected grid", highlight=True,
             title="to_grid: 6 pixels of width 3 make 2 rows", show="to_grid([1, 2, 3, 4, 5, 6], width=3)"),
        Case(call=lambda first: _call_partner(first, "flatten", [[1, 2], [3, 4]]), key="flatten",
             expected=[1, 2, 3, 4], expected_label="Expected pixels",
             title="flatten: the rows back into one list", show="flatten([[1, 2], [3, 4]])"),
    ],
    detectors=[
        Detector(lambda t: t.show.startswith(("to_grid", "flatten")) and t.got is None and t.error is None,
                 "to_grid or flatten isn't written yet (or returns None): write both, run their cells, then "
                 "run this check again."),
        Detector(lambda t: len(t.args) == 3 and t.args[1] and t.got == [0, 2, 3, 0, 5, 6, 0, 8, 9],
                 "the picture moved the wrong way: a positive dx moves the ink to the RIGHT."),
        Detector(lambda t: isinstance(t.got, list) and t.got and isinstance(t.got[0], list),
                 "shift returns a flat list of pixels, like every image: flatten the grid at the end."),
    ],
    returns=list,
)


# ============================================================================= CH · read_postcode (+ constrained)
def _toy_predict(image):
    """A toy 'reader' for 1-pixel images: the pixel value is the digit."""
    return image[0]


def _scores_for(best: str, runner_up: dict[int, int] | None = None) -> list[list[float]]:
    """Toy scores for 5 positions: the digit in `best` scores 1.0, every other digit 9.0; runner_up
    {position: digit} gives that digit 2.0 (a close second)."""
    scores = []
    for position, ch in enumerate(best):
        row = [9.0] * 10
        row[int(ch)] = 1.0
        if runner_up and position in runner_up:
            row[runner_up[position]] = 2.0
        scores.append(row)
    return scores


_VALID = {"28001", "28013", "28046", "28050"}


def _constrained(first, scores, valid):
    return _call_partner(first, "read_postcode_constrained", scores, set(valid))


CH = Task(
    "CH", "read_postcode", title="The Postcode Machine",
    practise="str.join, a set, and min() with a key over candidates",
    rescue=False, quality=False,
    input_labels=("Digit images", "Predict"), expected_label="Expected postcode",
    cases=[
        Case(([[2], [8], [0], [1], [3]], _toy_predict), "28013", highlight=True,
             title="read_postcode: five 1-pixel 'digits' read one at a time",
             show="read_postcode([[2], [8], [0], [1], [3]], predict=lambda image: image[0])"),
        Case(([[0], [0], [0], [4], [6]], _toy_predict), "00046",
             title="read_postcode: leading zeros stay (a postcode is text, not a number)",
             show="read_postcode([[0], [0], [0], [4], [6]], predict=lambda image: image[0])"),
        Case(call=lambda first: _constrained(first, _scores_for("28018", {4: 3}), _VALID), key="constrained",
             expected="28013", expected_label="Expected postcode", highlight=True,
             title="read_postcode_constrained: digit by digit says 28018 (not valid); the best VALID code is 28013",
             show="read_postcode_constrained(scores, {'28001', '28013', '28046', '28050'})"),
        Case(call=lambda first: _constrained(first, _scores_for("28046"), _VALID), key="valid-already",
             expected="28046", expected_label="Expected postcode",
             title="read_postcode_constrained: a valid code read digit by digit stays",
             show="read_postcode_constrained(scores, {'28001', '28013', '28046', '28050'})"),
    ],
    detectors=[
        Detector(lambda t: isinstance(t.got, list),
                 "you returned a list of digits. Join them into one string: \"\".join(str(d) for d in digits)."),
        Detector(lambda t: isinstance(t.got, int),
                 "a postcode is text: as an int, 00046 would lose its zeros. Return a string of 5 characters."),
        Detector(lambda t: t.show.startswith("read_postcode_constrained") and t.got is None and t.error is None,
                 "read_postcode_constrained isn't written yet (or returns None): write it, run its cell, then "
                 "run this check again."),
        Detector(lambda t: t.show.startswith("read_postcode_constrained") and t.got == "28018",
                 "you returned the digit-by-digit reading, which isn't a valid postcode. Pick the VALID code with "
                 "the smallest total score: min(valid, key=...)."),
    ],
    returns=str,
)


# ============================================================================= bridge 2 -> 3 (BR1-BR4)
# bridge_2to3.ipynb, the catch-up notebook for the three weeks between Lab 2 and Lab 3 (a preview of Theory
# S5-S6). It runs inside the Lab 2 kit. The file tasks call the student's function with a path in a scratch
# folder and read the file back, so each test shows what was really written. Expected values are literal;
# generators/gen_cases.py checks them against reference/lab2.py.
class _Words(str):
    """A text the TEST block prints as it is (no quotes): a description, not a value the function returns."""

    def __repr__(self) -> str:
        return str(self)


_NO_FILE = _Words("(no file at the path you were given)")
_HEADER_LINE = "image,prediction,truth"
_PATH_TEXT = "a file in a scratch folder (the check makes one)"
_CSV_TEXT = "image,prediction,truth\r\n0,7,7\r\n1,2,2\r\n2,1,7\r\n"
_CSV_REORDERED = "truth,image,prediction\r\n7,0,7\r\n2,1,2\r\n7,2,1\r\n"


def _scratch() -> tempfile.TemporaryDirectory:
    # ignore_cleanup_errors: on Windows a file the student left open can't be deleted; that is not their problem
    return tempfile.TemporaryDirectory(prefix="labkit_check_", ignore_cleanup_errors=True)


def _file_lines(fn, predictions: list, truth: list):
    """Call save_predictions(predictions, truth, path) in a scratch folder; return the lines of the file."""
    with _scratch() as folder:
        path = os.path.join(folder, "predictions.csv")
        fn(list(predictions), list(truth), path)
        if not os.path.isfile(path):
            return _NO_FILE
        with open(path, "rb") as f:
            data = f.read()
    return data.decode("utf-8", errors="replace").splitlines()


def _loaded(fn, text: str):
    """Write `text` to a scratch CSV file and return load_predictions(path)."""
    with _scratch() as folder:
        path = os.path.join(folder, "predictions.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return fn(path)


def _json_text(fn, per_digit: dict):
    """Call save_accuracy(per_digit, path) in a scratch folder; the text of the file (None: no file)."""
    with _scratch() as folder:
        path = os.path.join(folder, "accuracy.json")
        fn(dict(per_digit), path)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")


def _json_back(fn, per_digit: dict):
    """What json.load reads from the file save_accuracy wrote (or a text in brackets saying why it can't)."""
    text = _json_text(fn, per_digit)
    if text is None:
        return _NO_FILE
    try:
        return json.loads(text)
    except ValueError:
        return _Words(f"(not JSON: {text.strip()[:40]})")


def _json_layout(fn, per_digit: dict) -> str:
    text = _json_text(fn, per_digit)
    if text is None:
        return _NO_FILE
    try:
        json.loads(text)
    except ValueError:
        return _Words(f"(not JSON: {text.strip()[:40]})")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return _Words("all on one line")
    if any(line.startswith((" ", "\t")) for line in lines):
        return _Words("indented, one digit per line")
    return _Words("one digit per line, but not indented")


def _rows_of(t) -> list:
    return t.got if isinstance(t.got, list) else []


BR1 = Task(
    "BR1", "save_predictions", title="Save the predictions as CSV",
    practise="with open(path, \"w\", encoding=\"utf-8\", newline=\"\"), csv.writer and enumerate (Theory S5)",
    expected_label="Expected lines in the file",
    cases=[
        Case(call=lambda fn: _file_lines(fn, [7, 2, 1], [7, 2, 7]), key="three", highlight=True,
             expected=["image,prediction,truth", "0,7,7", "1,2,2", "2,1,7"],
             title="Three predictions: a header row, then one row per image (the first image is image 0)",
             show="save_predictions([7, 2, 1], [7, 2, 7], path)",
             show_inputs={"Predictions": "[7, 2, 1]", "Truth": "[7, 2, 7]", "Path": _PATH_TEXT}),
        Case(call=lambda fn: _file_lines(fn, [], []), key="empty", highlight=True,
             expected=["image,prediction,truth"],
             title="No predictions: the file still has its header row",
             show="save_predictions([], [], path)",
             show_inputs={"Predictions": "[]", "Truth": "[]", "Path": _PATH_TEXT}),
    ],
    detectors=[
        Detector(lambda t: t.got == _NO_FILE,
                 "nothing was written at `path`. Open the path you were given: open(path, \"w\", encoding=\"utf-8\", "
                 "newline=\"\"), not a file name of your own."),
        Detector(lambda t: t.raised(TypeError) and "bytes-like" in str(t.error),
                 "the file was opened in binary mode (\"wb\"). A CSV file is text: open(path, \"w\", "
                 "encoding=\"utf-8\", newline=\"\")."),
        Detector(lambda t: (t.raised(Exception) and "iterable expected" in str(t.error)),
                 "writer.writerow takes a LIST of values: writer.writerow([image, prediction, label])."),
        Detector(lambda t: t.got == [],
                 "the file is empty. Write every row inside the `with open(...) as f:` block: the file is complete "
                 "once that block closes it."),
        Detector(lambda t: "" in _rows_of(t),
                 "there is an empty line after every row: open the file with newline=\"\". The csv module writes "
                 "its own line ends, and without newline=\"\" Windows adds a second one."),
        Detector(lambda t: _rows_of(t) and _rows_of(t)[0] != _HEADER_LINE and _rows_of(t)[0][:1].isdigit(),
                 "the first line of the file is the header: writer.writerow([\"image\", \"prediction\", \"truth\"]) "
                 "once, before the rows."),
        Detector(lambda t: any(", " in line for line in _rows_of(t)),
                 "csv.writer puts the commas in for you: give writerow a list, writer.writerow([image, prediction, "
                 "label]), not a string you build yourself."),
        Detector(lambda t: _rows_of(t)[1:2] == ["1,7,7"],
                 "the first image is image 0 (as in test_images[0]): enumerate(...) counts from 0, so leave out "
                 "start=1."),
        Detector(lambda t: _rows_of(t)[1:2] == ["7,7"],
                 "each row starts with the position of the image: for image, (prediction, label) in "
                 "enumerate(zip(predictions, truth)) gives it."),
        Detector(lambda t: _rows_of(t)[1:] == ["0,7,7", "1,2,2", "2,7,1"],
                 "the columns are image, prediction, truth: the prediction comes before the true label."),
    ],
)


def _all_text(t) -> bool:
    got = t.got
    return (isinstance(got, tuple) and len(got) == 2 and all(isinstance(part, list) for part in got)
            and any(isinstance(value, str) for part in got for value in part))


BR2 = Task(
    "BR2", "load_predictions", title="Load them back",
    practise="csv.DictReader, and int() because everything in a CSV file is text (Theory S5)",
    expected_label="Expected (predictions, truth)",
    cases=[
        Case(call=lambda fn: _loaded(fn, _CSV_TEXT), key="three", highlight=True, expected=([7, 2, 1], [7, 2, 7]),
             title="A file written by save_predictions: two lists of ints, (predictions, truth)",
             show="load_predictions(path)",
             show_inputs={"File": "image,prediction,truth | 0,7,7 | 1,2,2 | 2,1,7   (| = a new line)"}),
        Case(call=lambda fn: _loaded(fn, "image,prediction,truth\r\n"), key="header-only", expected=([], []),
             title="Only the header row: two empty lists", show="load_predictions(path)",
             show_inputs={"File": "image,prediction,truth"}),
        Case(call=lambda fn: _loaded(fn, _CSV_REORDERED), key="reordered", highlight=True,
             expected=([7, 2, 1], [7, 2, 7]),
             title="The same data with the columns in another order: read them by name",
             show="load_predictions(path)",
             show_inputs={"File": "truth,image,prediction | 7,0,7 | 2,1,2 | 7,2,1   (| = a new line)"}),
    ],
    detectors=[
        Detector(_all_text,
                 "everything in a CSV file is text: you returned '7', and '7' == 7 is False (an accuracy of 0). "
                 "Turn each value back into a number: int(row[\"prediction\"])."),
        Detector(lambda t: t.raised(ValueError) and re.search(r"'(image|prediction|truth)'", str(t.error)),
                 "the header row was read as data (int(\"prediction\") fails). csv.DictReader reads the header for "
                 "you; with csv.reader, skip the first row."),
        Detector(lambda t: t.title.startswith("The same data with the columns"),
                 "you read the columns by position (row[1], row[2]). Read them by name with csv.DictReader: "
                 "row[\"prediction\"] and row[\"truth\"] work whatever the order of the columns."),
        Detector(lambda t: t.raised(KeyError),
                 "a column name doesn't match: the header of the file is image,prediction,truth, so the keys are "
                 "\"image\", \"prediction\" and \"truth\"."),
        Detector(lambda t: isinstance(t.got, list),
                 "return the two lists together, predictions first: return predictions, truth (a tuple)."),
        Detector(lambda t: isinstance(t.got, tuple) and len(t.got) == 3,
                 "return two lists, predictions and truth; the image column is only the position of each row."),
        Detector(lambda t: isinstance(t.got, tuple) and len(t.got) == 2 and t.got == tuple(reversed(t.expected))
                 and t.got != t.expected,
                 "predictions first, then truth: return predictions, truth."),
    ],
)


BR3 = Task(
    "BR3", "save_accuracy", title="Accuracy per digit, as JSON",
    practise="json.dump with indent=2, and what JSON does to dict keys (Theory S5)",
    cases=[
        Case(call=lambda fn: _json_back(fn, {7: 0.5, 1: 1.0}), key="back", highlight=True,
             expected={"7": 0.5, "1": 1.0}, expected_label="Expected data in the file",
             title="The accuracies, read back with json.load (JSON keys are always text: 7 comes back as \"7\")",
             show="save_accuracy({7: 0.5, 1: 1.0}, path)",
             show_inputs={"Accuracy per digit": "{7: 0.5, 1: 1.0}", "Path": _PATH_TEXT}),
        Case(call=lambda fn: _json_layout(fn, {7: 0.5, 1: 1.0}), key="layout", highlight=True,
             expected=_Words("indented, one digit per line"), expected_label="Expected layout",
             title="Indented, one digit per line (indent=2), so a person can read the file",
             show="save_accuracy({7: 0.5, 1: 1.0}, path)",
             show_inputs={"Accuracy per digit": "{7: 0.5, 1: 1.0}", "Path": _PATH_TEXT}),
        Case(call=lambda fn: _json_back(fn, {}), key="empty", expected={}, expected_label="Expected data in the file",
             title="No digits: an empty JSON object, {}", show="save_accuracy({}, path)",
             show_inputs={"Accuracy per digit": "{}", "Path": _PATH_TEXT}),
    ],
    detectors=[
        Detector(lambda t: t.got == _NO_FILE,
                 "nothing was written at `path`. Open the path you were given: open(path, \"w\", "
                 "encoding=\"utf-8\")."),
        Detector(lambda t: t.raised(AttributeError) and "has no attribute 'write'" in str(t.error),
                 "json.dump writes into an open FILE, not a path: with open(path, \"w\", encoding=\"utf-8\") as f: "
                 "then json.dump(per_digit, f, indent=2)."),
        Detector(lambda t: t.raised(TypeError) and "must be str, not dict" in str(t.error),
                 "f.write needs text, not a dict. json.dump(per_digit, f, indent=2) turns the dict into JSON text "
                 "and writes it."),
        Detector(lambda t: isinstance(t.got, str) and t.got.startswith("(not JSON") and "'" not in t.got
                 and ":" in t.got[10:],
                 "the file is not JSON: str(per_digit) writes {7: 0.5}, and JSON needs its keys in double quotes. "
                 "Write it with json.dump(per_digit, f, indent=2)."),
        Detector(lambda t: isinstance(t.got, str) and t.got.startswith("(not JSON"),
                 "the file is not JSON. Write it with json.dump(per_digit, f, indent=2), which writes valid JSON "
                 "for you."),
        Detector(lambda t: t.got in ("all on one line", "one digit per line, but not indented"),
                 "the data is right, but people read this file too: pass indent=2 to json.dump."),
    ],
)


def _unstripped(t) -> bool:
    got, expected = t.got, t.expected
    return (isinstance(got, tuple) and isinstance(expected, tuple) and len(got) == 2 and got != expected
            and all(isinstance(part, str) for part in got) and tuple(part.strip() for part in got) == expected)


BR4 = Task(
    "BR4", "parse_depot_line", title="One line of the depot sheet",
    practise="strip(), split() and a value or None (Theory S6)",
    input_labels=("Line",), expected_label="Expected (postcode, depot)",
    cases=[
        Case(("  28001 ; Retiro ",), ("28001", "Retiro"), highlight=True,
             title="The line from the depot sheet: spaces around everything"),
        Case(("28013;Chamartín",), ("28013", "Chamartín"), title="No spaces at all, and an accent"),
        Case(("28047 ; Vallecas\n",), ("28047", "Vallecas"),
             title="A line read from a file still ends with \\n: strip() removes it too"),
        Case(("28001 Retiro",), None, highlight=True, expected_label="Expected",
             title="No ';' in the line: not a depot line, so None"),
        Case(("",), None, expected_label="Expected", title="An empty line: None"),
        Case(("  ; Retiro",), None, expected_label="Expected", title="A depot without a postcode: None"),
        Case(("28001 ; Retiro ; Vallecas",), None, expected_label="Expected",
             title="Two ';' in one line: we can't trust it, so None"),
    ],
    detectors=[
        Detector(lambda t: isinstance(t.got, list),
                 "split() gives a list. Return a tuple of the two cleaned parts: return postcode, depot."),
        Detector(_unstripped,
                 "the parts still have spaces (or a \\n) around them: strip() each part after split(\";\")."),
        Detector(lambda t: isinstance(t.got, tuple) and t.got and isinstance(t.got[0], int),
                 "keep the postcode as text: int() would turn a postcode such as 00046 into 46."),
        Detector(lambda t: t.raised(ValueError) and "unpack" in str(t.error),
                 "this line doesn't split into exactly two parts. Split first, test len(parts) != 2, and return "
                 "None before you unpack."),
        Detector(lambda t: t.raised(IndexError),
                 "this line has no ';', so parts[1] doesn't exist. Test len(parts) != 2 first and return None."),
        Detector(lambda t: t.got is None and isinstance(t.expected, tuple),
                 "a good line gave None. Split on \";\" alone, then strip each part: the spaces around ';' are "
                 "not always there."),
        Detector(lambda t: t.expected is None and t.got is not None,
                 "a line that isn't \"postcode ; depot\" (no ';', two of them, or an empty part) gives None: a "
                 "value we can't trust is not guessed."),
    ],
    returns=(tuple, type(None)),
)


# ============================================================================= Holiday Quest (HQ1-HQ5)
# holiday_quest.ipynb: the reject option of the Postcode Machine, step by step, for everyone (the 3-week gap).
# No rescue: the solution is published at Lab 3. Literal expected values; teacher_only/solutions/holiday_quest.py
# checks that its solution PASSes every case. Toy scores come from _scores_for (the CH section above).
_ALL_ONES = [[1.0] * 10 for _ in range(5)]
_VALID_TEXT = "{'28001', '28013', '28046', '28050'}"
_S_28018 = "5 positions x 10 digits: the reading 28018 scores 1.0 per digit, a 3 at position 4 scores 2.0, " \
           "every other digit 9.0"
_S_28046 = "the reading 28046 scores 1.0 per digit, a 5 at position 3 and a 0 at position 4 score 2.0, " \
           "every other digit 9.0"
_S_ONES = "every digit scores 1.0 at every position (nothing looks like anything)"


def _hq(args: tuple, expected, title: str, inputs: dict, *, highlight: bool = False, kwargs: dict | None = None,
        compare=None, label: str | None = None) -> Case:
    return Case(args, expected, kwargs=kwargs or {}, title=title, show_inputs=inputs, highlight=highlight,
                compare=compare, expected_label=label)


def _swapped_score(t) -> bool:
    scores, code = t.args
    try:
        return abs(sum(scores[int(ch)][i] for i, ch in enumerate(code)) - t.got) < 1e-9
    except Exception:
        return False


HQ1 = Task(
    "HQ1", "code_score", title="How unlike is one candidate code?", rescue=False,
    practise="enumerate inside sum(), and int() on a character (Theory S4)",
    expected_label="Expected total",
    cases=[
        _hq((_scores_for("28018", {4: 3}), "28018"), 5.0, "The digit-by-digit reading: five digits at 1.0, total 5.0",
            {"Scores": _S_28018, "Code": "'28018'"}, highlight=True),
        _hq((_scores_for("28018", {4: 3}), "28013"), 6.0, "28013: four digits at 1.0 and a 3 at 2.0, total 6.0",
            {"Scores": _S_28018, "Code": "'28013'"}, highlight=True),
        _hq((_scores_for("28018", {4: 3}), "00046"), 37.0,
            "A code far from the reading: four unlike digits at 9.0 and one at 1.0, total 37.0",
            {"Scores": _S_28018, "Code": "'00046'"}),
    ],
    detectors=[
        Detector(lambda t: t.raised(TypeError) and "not str" in str(t.error),
                 "ch is a character such as '8', and a list index must be an int: scores[i][int(ch)]."),
        Detector(lambda t: t.raised(IndexError) or _swapped_score(t),
                 "scores[i] is the row of position i, and [int(ch)] picks the digit in that row: "
                 "scores[i][int(ch)], not scores[int(ch)][i]."),
        Detector(lambda t: isinstance(t.got, list),
                 "add the five scores up: sum(...) around the generator, not a list of them."),
    ],
    returns=(int, float),
)


def _pairs_like(t) -> bool:
    return isinstance(t.got, list) and all(isinstance(p, tuple) and len(p) == 2 for p in t.got)


HQ2 = Task(
    "HQ2", "best_two", title="The best two candidates", rescue=False,
    practise="(total, code) tuples, sorted() and a slice (Theory S3-S4)",
    depends_on=("HQ1",), expected_label="Expected best two",
    cases=[
        _hq((_scores_for("28018", {4: 3}), set(_VALID)), [(6.0, "28013"), (21.0, "28001")],
            "28013 is the best; three codes tie for second place at 21.0, so the smallest code comes second",
            {"Scores": _S_28018, "Valid codes": _VALID_TEXT}, highlight=True),
        _hq((_scores_for("28046", {3: 5, 4: 0}), set(_VALID)), [(5.0, "28046"), (7.0, "28050")],
            "A valid code read digit by digit wins; 28050 is a close second",
            {"Scores": _S_28046, "Valid codes": _VALID_TEXT}, highlight=True),
        _hq((_ALL_ONES, set(_VALID)), [(5.0, "28001"), (5.0, "28013")],
            "Every code scores the same: the two smallest codes, in order",
            {"Scores": _S_ONES, "Valid codes": _VALID_TEXT}),
    ],
    detectors=[
        Detector(lambda t: isinstance(t.got, list) and t.got and all(isinstance(p, str) for p in t.got),
                 "keep each code's total with it: return (total, code) tuples, e.g. [(6.0, '28013'), ...]; HQ3 "
                 "needs the totals."),
        Detector(lambda t: _pairs_like(t) and t.got and isinstance(t.got[0][0], str),
                 "put the total first, (total, code): tuples are sorted by their first item, so the smallest total "
                 "comes first."),
        Detector(lambda t: isinstance(t.got, list) and len(t.got) != 2,
                 "only the best two: slice the sorted list with [:2]."),
        Detector(lambda t: _pairs_like(t) and t.got and is_number(t.got[0][0]) and t.got[0][0] > t.expected[0][0],
                 "the smallest total comes first (a total is how UNlike the strip the code is): no reverse=True."),
        Detector(lambda t: _pairs_like(t) and [p[0] for p in t.got] == [p[0] for p in t.expected],
                 "the totals are right, but on equal totals the smaller code must come first. Sort the (total, code) "
                 "tuples themselves: a key that looks at the total only keeps the set's order, which can change "
                 "from run to run."),
        Detector(lambda t: isinstance(t.got, tuple),
                 "return a list of two tuples: sorted(...) gives a list, and [:2] keeps it a list."),
    ],
    returns=list,
)


HQ3 = Task(
    "HQ3", "read_or_reject", title="Say \"I'm not sure\"", rescue=False,
    practise="unpacking tuples, a comparison, and None as \"no answer\" (Theory S2-S3)",
    depends_on=("HQ2",), expected_label="Expected answer",
    cases=[
        _hq((_scores_for("28018", {4: 3}), set(_VALID), 5.0), "28013",
            "Sure enough: the best two are 15.0 apart, more than the margin 5.0, so the best code",
            {"Scores": _S_28018, "Valid codes": _VALID_TEXT, "Margin": "5.0"}, highlight=True),
        _hq((_scores_for("28018", {4: 3}), set(_VALID), 20.0), None,
            "Too close to call: 15.0 apart, less than the margin 20.0, so None (a person reads it)",
            {"Scores": _S_28018, "Valid codes": _VALID_TEXT, "Margin": "20.0"}, highlight=True),
        _hq((_scores_for("28018", {4: 3}), set(_VALID), 15.0), "28013",
            "Exactly the margin apart: not closer than the margin, so the best code",
            {"Scores": _S_28018, "Valid codes": _VALID_TEXT, "Margin": "15.0"}),
        _hq((_ALL_ONES, set(_VALID), 0), "28001", "Margin 0 never asks a person, even when the best two tie",
            {"Scores": _S_ONES, "Valid codes": _VALID_TEXT, "Margin": "0"}),
        _hq((_ALL_ONES, set(_VALID), 0.5), None, "The best two tie: any margin above 0 sends the parcel to a person",
            {"Scores": _S_ONES, "Valid codes": _VALID_TEXT, "Margin": "0.5"}),
    ],
    detectors=[
        Detector(lambda t: isinstance(t.got, str) and not t.got.isdigit(),
                 "return None itself (no quotes). None means \"no answer\"; any string would look like a postcode "
                 "to the code that uses it."),
        Detector(lambda t: isinstance(t.got, tuple),
                 "return the code only, e.g. '28013': best_two gives (total, code) tuples, so unpack them first."),
        Detector(lambda t: t.raised(TypeError) and "tuple" in str(t.error),
                 "best_two gives (total, code) tuples: unpack them first, (best, code), (second, _) = best_two(...), "
                 "then compare second - best with the margin."),
        Detector(lambda t: t.got is None and t.expected is not None and t.title.startswith("Sure enough"),
                 "you sent a sure answer to a person: return None only when the best two are CLOSE, "
                 "second - best < margin."),
        Detector(lambda t: t.got is None and t.expected is not None and t.title.startswith("Exactly"),
                 "a gap exactly equal to the margin is not closer than the margin: return None only when "
                 "second - best < margin (so >= keeps the code)."),
        Detector(lambda t: t.got is not None and t.expected is None and t.title.startswith("Too close"),
                 "the best two are closer than the margin here: return None, so a person reads the parcel."),
    ],
    returns=(str, type(None)),
)


def _cost_value(answers, truths, *, review=0.35, misroute=7.80) -> float | None:
    """The right cost per 1,000 parcels (None for no parcels)."""
    if not answers:
        return None
    reviews = sum(1 for a in answers if a is None)
    wrong = sum(1 for a, t in zip(answers, truths) if a is not None and a != t)
    return 1000 * (reviews * review + wrong * misroute) / len(answers)


def _none_as_misroute(t) -> bool:
    """The cost when None also counts as a wrong postcode (as a misroute, with or without the review)."""
    if not is_number(t.got) or not t.args[0]:
        return False
    answers, truths = t.args[0], t.args[1]
    review, misroute = t.kwargs.get("review_eur", 0.35), t.kwargs.get("misroute_eur", 7.80)
    reviews = sum(1 for a in answers if a is None)
    wrong = sum(1 for a, tr in zip(answers, truths) if a != tr)
    if reviews == 0:
        return False
    n = len(answers)
    return any(_close(t.got, 1000 * (r * review + wrong * misroute) / n) for r in (reviews, 0))


def _close(a, b) -> bool:
    return is_number(a) and b is not None and abs(a - b) < 1e-6


HQ4 = Task(
    "HQ4", "cost_per_1000", title="What does it cost?", rescue=False,
    practise="counting with conditions, `is None`, and default parameter values (Theory S2-S3)",
    input_labels=("Answers", "Truths"), expected_label="Expected cost per 1,000 parcels (€)",
    cases=[
        Case((["28013", None, "28001", "28046"], ["28013", "28050", "28002", "28046"]), 2037.5,
             compare=approx(1e-6), highlight=True,
             title="Four parcels: two right, one sent to a person (€0.35), one misrouted (€7.80)"),
        Case(([None, None], ["28001", "28002"]), 350.0, compare=approx(1e-6),
             title="Every parcel goes to a person: €0.35 each, so €350 per 1,000"),
        Case((["28001"], ["28001"]), 0.0, compare=approx(1e-6), title="Every parcel right: it costs nothing"),
        Case((["28013", None, "28001", "28046"], ["28013", "28050", "28002", "28046"]), 587.5,
             kwargs={"misroute_eur": 2.0}, compare=approx(1e-6), highlight=True,
             title="The same parcels with a cheaper misroute (misroute_eur=2.0)"),
        Case(([], []), 0.0, compare=approx(1e-6), title="No parcels: 0.0 (not an error)"),
    ],
    detectors=[
        Detector(lambda t: t.raised(ZeroDivisionError),
                 "with no parcels you divided by 0: return 0.0 first when the list is empty."),
        Detector(_none_as_misroute,
                 "None is not a wrong postcode: None != \"28050\" is True, so a review was counted as a misroute. "
                 "Test `answer is None` first."),
        Detector(lambda t: "misroute_eur" in t.kwargs and _close(t.got, _cost_value(t.args[0], t.args[1])),
                 "use the parameters review_eur and misroute_eur in the formula, not the numbers 0.35 and 7.80: "
                 "here misroute_eur is 2.0."),
        Detector(lambda t: is_number(t.got) and t.expected and abs(t.got * 1000 - t.expected) < 1e-6,
                 "this is the cost per parcel: multiply by 1,000 for the cost per 1,000 parcels."),
        Detector(lambda t: is_number(t.got) and t.expected and len(t.args[0]) > 1
                 and abs(t.got - t.expected * len(t.args[0])) < 1e-6,
                 "this is the total for these parcels: divide by the number of parcels, then multiply by 1,000."),
    ],
    returns=(int, float),
)


_STRIPS = [(_scores_for("28018", {4: 3}), "28013"), (_scores_for("28046", {3: 5, 4: 0}), "28050"),
           ([row[:] for row in _ALL_ONES], "28013")]
_STRIPS_TEXT = ("3 toy strips: 28013 read with a gap of 15.0 to the second best; 28050 read as 28046 (gap 2.0); "
                "28013 where every code ties (gap 0)")


HQ5 = Task(
    "HQ5", "margin_costs", title="The cost table", rescue=False,
    practise="a dict comprehension that calls your own functions (Theory S4)",
    depends_on=("HQ3", "HQ4"), expected_label="Expected {margin: cost per 1,000}",
    cases=[
        _hq((_STRIPS, set(_VALID), [0, 1, 5, 20]), {0: 5200.0, 1: 2716.666667, 5: 233.333333, 20: 350.0},
            "Four margins: 0 trusts every answer, 20 sends every parcel to a person",
            {"Strips": _STRIPS_TEXT, "Valid codes": _VALID_TEXT, "Margins": "[0, 1, 5, 20]"}, highlight=True,
            compare=approx(1e-4)),
        _hq((_STRIPS, set(_VALID), [0, 1, 5, 20]), {0: 1333.333333, 1: 783.333333, 5: 233.333333, 20: 350.0},
            "The same strips with a cheaper misroute (misroute_eur=2.0)",
            {"Strips": _STRIPS_TEXT, "Valid codes": _VALID_TEXT, "Margins": "[0, 1, 5, 20]",
             "misroute_eur": "2.0"}, highlight=True, kwargs={"misroute_eur": 2.0}, compare=approx(1e-4)),
        _hq((_STRIPS, set(_VALID), []), {}, "No margins: an empty table",
            {"Strips": _STRIPS_TEXT, "Valid codes": _VALID_TEXT, "Margins": "[]"}),
    ],
    detectors=[
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.expected, dict) and set(t.got) != set(t.expected),
                 "one key per margin you were given: {margin: cost_per_1000(...) for margin in margins}."),
        Detector(lambda t: isinstance(t.got, dict) and any(isinstance(v, list) for v in t.got.values()),
                 "each value is ONE number, the cost: pass the answers and the truths to cost_per_1000."),
        Detector(lambda t: "misroute_eur" in t.kwargs and isinstance(t.got, dict) and t.got
                 and all(_close(t.got.get(m), c) for m, c in {0: 5200.0, 1: 2716.666667, 5: 233.333333,
                                                               20: 350.0}.items()),
                 "pass misroute_eur on to cost_per_1000(..., misroute_eur=misroute_eur); here it is 2.0."),
        Detector(lambda t: isinstance(t.got, list),
                 "return a dict, {margin: cost}: a dict comprehension builds it in one line."),
    ],
    returns=dict,
)


# notebook order: W, CORE (C1-C5), the AI activity, OPTIONAL (C6, ENG1, SCI1, BLD1), ADVANCED; then the two
# notebooks of the 3-week gap (bridge_2to3: BR1-BR4; holiday_quest: HQ1-HQ5), which run inside this kit
TASKS = [W, C1, C2, C3, C4, C5, AI, C6, ENG1, SCI1, BLD1, ENG2, SCI2, BLD2, CH,
         BR1, BR2, BR3, BR4, HQ1, HQ2, HQ3, HQ4, HQ5]
