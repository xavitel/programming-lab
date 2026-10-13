"""Checks for Lab 1 · Read One Digit (blueprint §5.1).

What each task is checked against. Expected values that come from the reference are stored in
cases/lab1_data.json by generators/gen_cases.py, so this file holds no solution code: only the
inputs, the toy answers, and the misconception detectors ("Likely cause").

Tasks, in notebook order: W, C1, C2, C3, C5 (CORE), C4, ENG1, SCI1, BLD1 (OPTIONAL, checked like CORE),
ENG2, SCI2, BLD2 (ADVANCED: sanity checks only, no rescue).

Every case has a short title: check() prints it as "TEST n: <title>", with the inputs labelled by the
task's input_labels (e.g. Predictions / Ground truth), on PASS (the highlighted tests) and for the first
failing test.

SCI1 and BLD1 each ask for two functions. Their check takes the first one, and finds the second
one by name: in the notebook first, then next to the first function (so rescue("SCI1") works):
    check("SCI1", fit_centroids)      # also checks predict_centroid
    check("BLD1", confusion_pairs)    # also checks top_confusions
"""
from __future__ import annotations

import random
import sys

from ..check import is_stub
from ..drawing import centre_of_mass, prepare_drawing
from ..spec import Case, Detector, Digits, Fact, Raises, Speed, Task, approx, custom
from ..text import is_number, short

DATA_FILES = ("mnist_train_10k.csv.gz", "mnist_test_2k.csv.gz")
MOMENT_WORK = (100_000, 784)      # the final run: 100 test images x 1,000 training images, 784 pixels each
FINALE = "Putting It All Together"   # the name of that run in the notebook (check_all, evaluate, start)

TRAIN = Digits("mnist_train_10k.csv.gz", limit=1000)    # = train_images, train_labels in the notebook
TEST = Digits("mnist_test_2k.csv.gz", limit=100)        # = test_images, test_labels in the notebook

NEXT_HINT_2 = "open Hint 2 below (the plan), fix your code, run the cell with def again, then this check."


# ============================================================================= small helpers
def _number(value: object) -> bool:
    return is_number(value)


def _same_int(got: object, expected: object) -> bool:
    """An int label equal to the expected one (a float such as 3.0 or a bool doesn't count)."""
    return type(got) is int and got == expected


INT_LABEL = custom(_same_int, "(an int)")


def _random_pairs(seed: int = 2026, n: int = 3, size: int = 784) -> list[tuple[list[int], list[int]]]:
    """n pairs of random 784-pixel images (the same every time: the seed is fixed)."""
    rng = random.Random(seed)
    return [([rng.randrange(256) for _ in range(size)], [rng.randrange(256) for _ in range(size)])
            for _ in range(n)]


_PAIRS: list = []


def _pairs() -> list:
    if not _PAIRS:
        _PAIRS.extend(_random_pairs())
    return _PAIRS


def _symmetric(fn) -> list[bool]:
    return [fn(a[:], b[:]) == fn(b[:], a[:]) for a, b in _pairs()]


def _transpose(picture: object) -> str | None:
    """The same text picture with rows and columns swapped (square pictures only)."""
    if not isinstance(picture, str):
        return None
    rows = picture.split("\n")
    if len(set(map(len, rows))) != 1 or len(rows) != len(rows[0]):
        return None
    return "\n".join("".join(row[c] for row in rows) for c in range(len(rows)))


def _growing_rows(picture: str) -> list[str]:
    """The rows a line that is never reset gives: row 1 holds rows 0-1, row 2 holds rows 0-2, ..."""
    rows = picture.split("\n")
    return ["".join(rows[: k + 1]) for k in range(len(rows))]


def _pixel_by_pixel_rows(picture: str) -> list[str]:
    """The lines a renderer gives when it appends the line after every pixel: '.', '.#', '#', '#.'."""
    return [row[: c + 1] for row in picture.split("\n") for c in range(len(row))]


def _swap_ink(picture: object) -> str | None:
    if not isinstance(picture, str):
        return None
    return picture.translate(str.maketrans("#.", ".#"))


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
    """Call the partner function; None if it isn't defined yet (a detector then says so)."""
    fn = _partner(first, name)
    return None if fn is None else fn(*args, **kwargs)


def _labels_of(t) -> list:
    return t.args[2] if len(t.args) > 2 else []


def _c1_wrong(t, value) -> bool:
    """C1: the student's number equals `value(a, b)` (a known mistake) and isn't the right answer."""
    return len(t.args) == 2 and _number(t.got) and t.got != t.expected and t.got == value(t.args[0], t.args[1])


# ============================================================================= W · ink_count
W = Task(
    "W", "ink_count", title="Count the ink",
    practise="counting with a loop and an if (Theory S2)",
    input_labels=("Image",), expected_label="Expected count",
    cases=[
        Case(([0, 128, 255, 127],), 2, title="Four pixels: 128 and 255 are ink (128 is exactly the threshold)",
             highlight=True),
        Case(([0, 50, 100, 200, 30, 120],), 4, kwargs={"threshold": 50}, title="A lower threshold: 50",
             highlight=True),
        Case(([],), 0, title="An empty image has no ink"),
        Case(lambda: (TRAIN.images[0],), show="ink_count(train_images[0])", key="real-train-0",
             title="A real image: the first training image", highlight=True),
    ],
    detectors=[
        Detector(lambda t: t.args and isinstance(t.got, int) and t.got == sum(
                     1 for p in t.args[0] if p > t.kwargs.get("threshold", 128)),
                 "a pixel exactly equal to the threshold should count as ink: use >= threshold, not >."),
        Detector(lambda t: t.args and isinstance(t.got, int) and t.got == len(t.args[0]) - t.expected,
                 "you counted the paper pixels. Count the pixels that are >= threshold."),
        Detector(lambda t: "threshold" in t.kwargs and t.got == sum(1 for p in t.args[0] if p >= 128),
                 "your code uses 128 where it should use the threshold parameter."),
        Detector(lambda t: t.args and isinstance(t.got, int) and t.got == sum(t.args[0]) and t.got != t.expected,
                 "you added up the pixel values. Count them instead: add 1 for each ink pixel."),
    ],
    returns=int,
)


# ============================================================================= C1 · squared_distance
C1 = Task(
    "C1", "squared_distance", title="How different are two images?",
    practise="the running-total loop over two lists (Theory S2)",
    minimum_path=True,
    input_labels=("Image a", "Image b"), expected_label="Expected distance",
    cases=[
        Case(([0, 0], [3, 4]), 25, title="Two tiny images of 2 pixels: (0 - 3) ** 2 + (0 - 4) ** 2 = 9 + 16",
             highlight=True),
        Case(([1, 2, 3], [1, 2, 3]), 0, title="Identical images are 0 apart", highlight=True),
        Case(([2, 7, 1], [5, 3, 4]), 34, title="Three pixels: 9 + 16 + 9"),
        Case(call=_symmetric, expected=[True, True, True], key="symmetry", expected_label="Expected",
             title="The same distance both ways: squared_distance(a, b) == squared_distance(b, a)",
             show="squared_distance(a, b) == squared_distance(b, a) for 3 random pairs of 784-pixel images"),
        Case(lambda: (TEST.images[0], TRAIN.images[0]), key="real-0", highlight=True,
             title="Two real 784-pixel images: the first test image and the first training image",
             show="squared_distance(test_images[0], train_images[0])"),
    ],
    detectors=[
        Detector(lambda t: t.raised(TypeError) and "str" in str(t.error),
                 "some values are text (str), not numbers. Are the pixels strings (did you call int())? "
                 "Does a running total start at \"\" instead of 0?"),
        Detector(lambda t: isinstance(t.got, str),
                 "your function returned text (a str). Return the number itself, e.g. `return total`."),
        Detector(lambda t: isinstance(t.got, float) and _number(t.expected) and t.expected > 0
                 and abs(t.got ** 2 - t.expected) < 1e-6 * max(1, t.expected),
                 "you returned the square root. The spec asks for the squared distance, so skip math.sqrt."),
        Detector(lambda t: len(t.args) == 2 and _number(t.got) and t.got != t.expected
                 and t.got == sum(abs(x - y) for x, y in zip(t.args[0], t.args[1])),
                 "you added abs(a[i] - b[i]). Square each difference instead: (a[i] - b[i]) ** 2."),
        Detector(lambda t: len(t.args) == 2 and _number(t.got) and t.got != t.expected
                 and t.got == (sum(t.args[0]) - sum(t.args[1])) ** 2,
                 "you subtracted the totals of the two images. Compare them pixel by pixel: add up "
                 "(a[i] - b[i]) ** 2 for every position i."),
        Detector(lambda t: len(t.args) == 2 and _number(t.got) and t.got != t.expected
                 and t.got == sum(x - y for x, y in zip(t.args[0], t.args[1])),
                 "you added the differences without squaring them, so positive and negative ones cancel "
                 "out. Square each difference: (a[i] - b[i]) ** 2."),
        Detector(lambda t: len(t.args) == 2 and t.args[0] and _number(t.got) and t.got != t.expected
                 and t.got == (t.args[0][0] - t.args[1][0]) ** 2,
                 "only the first pixel was counted. Is `return` inside the loop? Move it one indent to the "
                 "left, so it runs after the loop has finished. (Or does the loop stop early? range(len(a)) "
                 "already reaches the last position.)"),
        Detector(lambda t: len(t.args) == 2 and len(t.args[0]) > 2 and _number(t.got) and t.got != t.expected
                 and t.got == sum((x - y) ** 2 for x, y in zip(t.args[0][:-1], t.args[1][:-1])),
                 "the last pixel was skipped: range(len(a)) already stops at the last position, so don't "
                 "subtract 1."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum((x - y) ^ 2 for x, y in zip(a, b))),
                 "in Python, ^ is not a power (it compares bits: XOR). Square with ** 2, or multiply the "
                 "difference by itself: (a[i] - b[i]) ** 2."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: (a[-1] - b[-1]) ** 2),
                 "only the last pixel counts: total = ... replaces the total at every step. Add to it "
                 "instead: total += (a[i] - b[i]) ** 2."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum((x - y) ** 2 for x in a for y in b)),
                 "two nested loops compare every pixel of a with every pixel of b. Use one loop over the "
                 "positions: for i in range(len(a)), then a[i] and b[i]."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum(x - y ** 2 for x, y in zip(a, b))),
                 "** runs before -, so a[i] - b[i] ** 2 squares only b[i]. Put brackets around the "
                 "difference: (a[i] - b[i]) ** 2."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum((a[v] - b[v]) ** 2 for v in a)),
                 "for i in a gives you the pixel values, not their positions. Loop over the positions: "
                 "for i in range(len(a))."),
        Detector(lambda t: _c1_wrong(t, lambda a, b: sum((x - y) * 2 for x, y in zip(a, b))),
                 "* 2 doubles a number; ** 2 squares it. Use (a[i] - b[i]) ** 2."),
        Detector(lambda t: isinstance(t.got, list) and t.got and all(v is False or v is True for v in t.got),
                 "your answer changes when you swap a and b, so it can't be a distance yet. Square each "
                 "difference: (a[i] - b[i]) ** 2 is the same as (b[i] - a[i]) ** 2."),
        Detector(lambda t: t.got is None and t.error is None,
                 "your function returned None. End it with `return total`, after the loop (not indented "
                 "inside it)."),
    ],
    returns=int,
    speed=Speed(lambda: (TEST.images[0], TRAIN.images[0]), budget_s=150e-6,
                note="C2 calls this 100,000 times. Are you converting values inside the loop?"),
)


# ============================================================================= C2 · predict_nearest
def _real_nearest_case(i: int) -> Case:
    return Case(lambda: (TEST.images[i], TRAIN.images[:200], TRAIN.labels[:200]), key=f"real-{i}",
                show=f"predict_nearest(test_images[{i}], train_images[:200], train_labels[:200])",
                facts={"index": Fact("nearest_index")}, highlight=i == 0,
                title=f"A real prediction: test image #{i} against the first 200 training images")


C2 = Task(
    "C2", "predict_nearest", title="The closest picture wins",
    practise="the best-so-far loop over parallel lists (Theory S2–S3)",
    minimum_path=True,
    depends_on=("C1",),
    input_labels=("Image", "Training images", "Training labels"), expected_label="Expected label",
    cases=[
        # image [6] vs [9], [4], [20], [0]: distances 9, 4, 196, 36 -> nearest at position 1, label 3
        Case(([6], [[9], [4], [20], [0]], [7, 3, 8, 5]), 3, highlight=True,
             title="A tiny world of 1-pixel images: distances 9, 4, 196, 36, so [4] is closest (label 3)",
             facts={"index": 1, "distance": 4, "farthest": 8}),
        # a tie: [3] and [7] are both 4 away from [5]; the earlier one (label 4) wins
        Case(([5], [[9], [3], [7]], [2, 4, 6]), 4, highlight=True,
             title="A tie: [3] and [7] are both 4 away from [5], so the earlier one wins",
             facts={"index": 1, "distance": 4, "later_tie": 6}),
        # a tie where the earlier image has the BIGGER label: min() over (distance, label) pairs picks 1
        Case(([5], [[7], [3], [9]], [6, 1, 2]), 6,
             title="Another tie, where the earlier image has the bigger label",
             facts={"index": 0, "distance": 4, "later_tie": 1}),
    ] + [_real_nearest_case(i) for i in range(5)],
    detectors=[
        Detector(lambda t: t.raised(TypeError) and "NoneType" in str(t.error)
                 and ("<" in str(t.error) or ">" in str(t.error)),
                 "best starts as None. Start it at float(\"inf\") so any real distance is smaller."),
        Detector(lambda t: t.raised(TypeError) and "'int' and 'list'" in str(t.error),
                 "squared_distance received the whole list of training images. Pass it one image at a time: "
                 "squared_distance(image, train_images[i])."),
        Detector(lambda t: t.raised(TypeError) and "indices must be integers" in str(t.error)
                 and "not list" in str(t.error),
                 "for i in train_images gives you the images themselves, not their positions. Loop over "
                 "positions with for i in range(len(train_images)), then use train_images[i] and train_labels[i]."),
        Detector(lambda t: isinstance(t.got, list),
                 "you returned a list (the nearest image itself?). Return its label: one int such as 7."),
        Detector(lambda t: t.facts.get("index") is not None and t.got == t.facts["index"] and t.got != t.expected,
                 "you returned the position of the nearest image. Return its label: train_labels[best_index]."),
        Detector(lambda t: "distance" in t.facts and t.got == t.facts["distance"] and t.got != t.expected,
                 "you returned the smallest distance. Return the label of that image instead."),
        Detector(lambda t: "later_tie" in t.facts and t.got == t.facts["later_tie"],
                 "two training images are equally close and you returned the wrong one: the spec says the "
                 "earlier one wins. With a loop, compare with <, not <=. With min() or sorted() over "
                 "(distance, label) pairs, a tie goes to the smaller label: pair each distance with its "
                 "position instead, (distance, i)."),
        Detector(lambda t: _labels_of(t) and t.got == _labels_of(t)[0] and t.got != t.expected,
                 "you returned the label of the first training image. Usual causes: (1) best starts at 0 (no "
                 "distance is ever smaller than 0, so it never changes): start it at float(\"inf\"); (2) "
                 "`return` sits inside the loop (so the loop stops after one image): return after the loop; "
                 "(3) the loop variable is also called image (for image in train_images), so every training "
                 "image is compared with itself (distance 0): call it train_image.",
                 next_step="open Hint 3 below and compare it with your code line by line."),
        Detector(lambda t: "farthest" in t.facts and t.got == t.facts["farthest"],
                 "you found the training image that is farthest away. The nearest one has the smallest "
                 "distance: check the comparison (<) and the starting value of best."),
        Detector(lambda t: _labels_of(t) and t.got == _labels_of(t)[-1] and t.got != t.expected,
                 "you returned the label of the last training image. Usual causes: (1) best_distance is never "
                 "updated inside the if (you need both best_distance = distance and best_index = i); (2) "
                 "best_distance = float(\"inf\") sits inside the loop, so it starts again for every image; "
                 "(3) the return uses i instead of best_index; (4) an update sits outside the if, so it runs "
                 "for every image.",
                 next_step="open Hint 3 below and compare it with your code line by line."),
        Detector(lambda t: t.got is None and t.error is None,
                 "your function returned None. Either the return line is missing, or the label is never set "
                 "because no distance is smaller than the starting value of best: start it at float(\"inf\")."),
        Detector(lambda t: _number(t.got) and not 0 <= t.got <= 9,
                 "that number can't be a label (labels are 0 to 9): did you return a distance or a position "
                 "instead of train_labels[...]?"),
    ],
    returns=int,
)


# ============================================================================= C3 · accuracy
def _n_correct(t) -> int | None:
    try:
        return sum(1 for p, q in zip(t.args[0], t.args[1]) if p == q)
    except Exception:
        return None


C3 = Task(
    "C3", "accuracy", title="How often is it right?",
    practise="counting, dividing, and guarding the empty case (Theory S2)",
    minimum_path=True,
    input_labels=("Predictions", "Ground truth"), expected_label="Expected accuracy",
    cases=[
        Case(([1, 2, 3, 4], [1, 2, 8, 4]), 0.75, compare=approx(1e-9), highlight=True,
             title="Accuracy calculation: 3 of 4 predictions match the ground truth"),
        Case(([1, 2, 3], [1, 2, 4]), 2 / 3, compare=approx(1e-3), title="2 of 3 right: a fraction, not 0 or 1"),
        Case(([], []), 0.0, highlight=True, title="Empty lists: nothing to count, so 0.0 (not an error)"),
        Case(([5, 5, 0], [5, 5, 0]), 1.0, highlight=True, title="Every prediction right: 1.0"),
        Case(([3, 1, 4, 1, 5], [3, 1, 4, 1, 6]), 0.8, compare=approx(1e-9), title="4 of 5 right"),
        Case(([7, 7, 7], [1, 2, 3]), 0.0, title="Every prediction wrong: 0.0"),
    ],
    detectors=[
        Detector(lambda t: t.raised(ZeroDivisionError),
                 "empty lists: there is nothing to divide by. Check for empty lists first and return 0.0."),
        Detector(lambda t: _number(t.got) and _number(t.expected) and t.expected > 0
                 and abs(t.got - 100 * t.expected) < 0.1,
                 "you returned a percentage. The spec asks for a fraction between 0 and 1: drop the * 100."),
        Detector(lambda t: type(t.got) is int and len(t.args[1]) == 0 and t.got == 0,
                 "for empty lists, return 0.0 (a float), not 0."),
        Detector(lambda t: type(t.got) is int and len(t.args[1]) > 1 and t.got == _n_correct(t) // len(t.args[1])
                 and t.got != t.expected,
                 "you used //, which rounds down to a whole number. Use / to get a fraction such as 0.667."),
        Detector(lambda t: _number(t.got) and len(t.args[1]) > 1 and t.got == _n_correct(t)
                 and t.got != t.expected,
                 "you returned how many predictions are right. The spec asks for the fraction: divide by the "
                 "number of predictions."),
        Detector(lambda t: _number(t.got) and _number(t.expected) and t.got != t.expected
                 and any(t.got == round(t.expected, k) for k in (1, 2, 3)),
                 "you rounded the result. Return the exact fraction, and round only when you print it, for "
                 "example f\"{acc:.1%}\"."),
        Detector(lambda t: _number(t.got) and len(t.args[1]) > 1 and t.got != t.expected
                 and t.got == (t.args[0][0] == t.args[1][0]) / len(t.args[1]),
                 "only the first prediction was counted: is `return` inside the loop? Move it one indent to the "
                 "left, so it runs after the loop."),
        Detector(lambda t: _number(t.got) and t.got == 1 and t.expected != 1 and len(t.args[1]) > 0,
                 "you returned 1.0, but some predictions differ from the ground truth. Compare one position at "
                 "a time (predictions[i] == truth[i]) and count only the matches."),
        Detector(lambda t: _number(t.got) and t.got == 0 and (_n_correct(t) or 0) > 0,
                 "you returned 0, but some predictions are right. Do you compare the whole lists "
                 "(predictions == truth) instead of one position at a time (predictions[i] == truth[i])? Or is "
                 "the count never increased?"),
    ],
    returns=float,
)


# ============================================================================= C4 · count_labels
def _off_by(t, step: int) -> bool:
    return (isinstance(t.got, dict) and isinstance(t.expected, dict) and t.expected
            and set(t.got) == set(t.expected) and all(t.got[k] == t.expected[k] + step for k in t.expected))


C4 = Task(
    "C4", "count_labels", title="Is our test fair?",
    practise="counting with a dict (Theory S2–S3)",
    not_needed_later=True, optional=True,
    input_labels=("Labels",), expected_label="Expected counts",
    cases=[
        Case(([3, 1, 3],), {3: 2, 1: 1}, title="Three labels: two 3s and one 1", highlight=True),
        Case(([],), {}, title="An empty list gives an empty dict", highlight=True),
        Case(([7, 7, 7, 7, 2],), {7: 4, 2: 1}, title="An unbalanced list: mostly 7s"),
        Case(lambda: (TEST.labels[:100],), key="real-test-labels", show="count_labels(test_labels)",
             title="The labels of our 100 test images", highlight=True),
    ],
    detectors=[
        Detector(lambda t: t.raised(KeyError),
                 "the first time you see a label, it isn't a key yet. Try counts[label] = counts.get(label, 0) + 1."),
        Detector(lambda t: isinstance(t.got, list),
                 "you returned a list. The spec asks for a dict that maps each label to how many times it "
                 "appears, like {3: 2, 1: 1}."),
        Detector(lambda t: isinstance(t.got, dict) and any(isinstance(k, str) for k in t.got),
                 "your keys are text such as '3'. Use the label itself as the key: counts[label], not "
                 "counts[str(label)]."),
        Detector(lambda t: _off_by(t, 1),
                 "each count is one too high: a new label should count 1 the first time you see it. Start it "
                 "at 0 and then add 1, or set it to 1 directly, not both."),
        Detector(lambda t: _off_by(t, -1),
                 "each count is one too low: the first time you see a label it should already count as 1."),
        Detector(lambda t: t.got == {} and t.args and len(t.args[0]) > 0,
                 "your dict stays empty: counts.get(label, 0) + 1 works out the new count but doesn't store it. "
                 "Write counts[label] = counts.get(label, 0) + 1."),
        Detector(lambda t: isinstance(t.got, dict) and t.args and len(set(t.args[0])) > 1
                 and t.got == {t.args[0][-1]: 1},
                 "only the last label is counted: counts = {} is inside the loop, so the dict starts again for "
                 "every label. Create it once, before the loop."),
        Detector(lambda t: isinstance(t.got, dict) and t.args and t.got and t.got != t.expected
                 and set(t.got) == set(range(len(t.args[0]))),
                 "your keys are positions (0, 1, 2, ...). Use the label itself as the key: for label in "
                 "labels: counts[label] = ..."),
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.expected, dict) and t.got != t.expected
                 and set(t.got) == set(t.expected) and all(v == 1 for v in t.got.values()),
                 "every count is 1: you set the count to 1 each time. Add 1 to the count you already have: "
                 "counts[label] = counts.get(label, 0) + 1."),
    ],
    returns=dict,
)


# ============================================================================= C5 · render_digit
C5 = Task(
    "C5", "render_digit", title="Draw it in text",
    practise="nested loops, r * width + c, and building strings (Theory S2–S3)",
    not_needed_later=True,
    input_labels=("Image",), expected_label="Expected picture",
    cases=[
        Case(([0, 200, 255, 10],), ".#\n#.", kwargs={"width": 2}, highlight=True,
             title="A 2 x 2 image: row 0 is [0, 200], row 1 is [255, 10]"),
        # rows: 255 255 255 / 0 0 128 / 0 127 0 (not symmetric, so a swap of rows and columns shows)
        Case(([255, 255, 255, 0, 0, 128, 0, 127, 0],), "###\n..#\n...", kwargs={"width": 3}, highlight=True,
             title="A 3 x 3 image: 128 is ink (>= threshold), 127 is paper",
             facts={"strict": "###\n...\n..."}),
        Case(([10, 60, 50, 90],), ".#\n##", kwargs={"width": 2, "threshold": 50},
             title="Another threshold: 50", facts={"threshold_128": "..\n.."}),
        # not square: 2 rows of 3 pixels (len(image) // width rows, not width rows)
        Case(([0, 255, 255, 0, 0, 255],), ".##\n..#", kwargs={"width": 3}, highlight=True,
             title="Not square: 2 rows of 3 pixels"),
        Case(lambda: (TEST.images[0],), key="real-0", show="render_digit(test_images[0])",
             title="A real 28 x 28 image: the first test image"),
    ],
    detectors=[
        Detector(lambda t: t.got is None and t.error is None and t.printed.strip(),
                 "your function printed the picture but returned None. Build the whole string and return it: "
                 "the check (and show_mistakes) need the string itself."),
        Detector(lambda t: isinstance(t.got, list),
                 "you returned a list of lines. Join them into one string: \"\\n\".join(lines)."),
        Detector(lambda t: isinstance(t.got, str) and isinstance(t.expected, str)
                 and t.got == str(t.expected.split("\n")),
                 "you turned the list into text with str(lines), which keeps the brackets and quotes. Join "
                 "the lines instead: \"\\n\".join(lines)."),
        Detector(lambda t: t.raised(IndexError) and t.args
                 and len(t.args[0]) // t.kwargs.get("width", 28) != t.kwargs.get("width", 28),
                 "this image isn't square: it has len(image) // width rows, not width rows. Loop over the rows "
                 "with for r in range(len(image) // width)."),
        Detector(lambda t: isinstance(t.got, str) and isinstance(t.expected, str) and "\n" in t.expected
                 and t.got != t.expected and t.got == "\n".join(_growing_rows(t.expected)),
                 "each line also holds the rows before it: line = \"\" runs only once. Start it inside the "
                 "row loop, so every row starts empty."),
        Detector(lambda t: isinstance(t.got, str) and isinstance(t.expected, str) and t.got != t.expected
                 and t.got == "\n".join(_pixel_by_pixel_rows(t.expected)),
                 "a line is added after every pixel: lines.append(line) is inside the column loop. Move it "
                 "one indent to the left, so it runs once per row."),
        Detector(lambda t: t.kwargs.get("width", 28) != 28 and (t.raised(IndexError) or t.got == ""),
                 "your code uses 28 where it should use the width parameter: this toy image is only "
                 "a few pixels wide."),
        Detector(lambda t: "threshold_128" in t.facts and t.got == t.facts["threshold_128"],
                 "your code uses 128 where it should use the threshold parameter."),
        Detector(lambda t: isinstance(t.got, str) and t.got == t.expected + "\n",
                 "there is an extra newline at the end. Join the lines with \"\\n\".join(lines) instead of "
                 "adding \"\\n\" after every line."),
        Detector(lambda t: isinstance(t.got, str) and t.got == "\n" + t.expected,
                 "there is an extra newline at the start. Join the lines with \"\\n\".join(lines)."),
        Detector(lambda t: isinstance(t.got, str) and "\n" in t.expected and t.got == t.expected.replace("\n", ""),
                 "everything is on one line. Put \"\\n\" between the rows: build one string per row, then "
                 "\"\\n\".join(rows)."),
        Detector(lambda t: isinstance(t.got, str) and t.got != t.expected and t.got == _transpose(t.expected),
                 "your picture is flipped along the diagonal (rows and columns swapped): pixel (r, c) is "
                 "image[r * width + c], not image[c * width + r]."),
        Detector(lambda t: "strict" in t.facts and t.got == t.facts["strict"],
                 "a pixel exactly equal to the threshold should be ink: use >= threshold, not >."),
        Detector(lambda t: isinstance(t.got, str) and t.got == _swap_ink(t.expected),
                 "ink and paper are swapped: use '#' for pixels >= threshold and '.' for the others."),
    ],
    returns=str,
)


# ============================================================================= ENG1 · safe_distance
def _expects_error(t) -> bool:
    return isinstance(t.expected, Raises)


ENG1 = Task(
    "ENG1", "safe_distance", title="Check Your Inputs: Are Both Images the Same Size?",
    practise="validating inputs and raising an error (raise, a new tool)",
    depends_on=("C1",),
    input_labels=("Image a", "Image b"),
    cases=[
        Case(([0, 0], [3, 4]), 25, title="Same size: the same answer as squared_distance", highlight=True),
        Case(([1, 2], [1]), Raises(ValueError, "images have different lengths: 2 vs 1"), highlight=True,
             title="Different sizes (2 pixels vs 1): raise a ValueError with a clear message"),
        Case(([5], [1, 2, 3]), Raises(ValueError, "images have different lengths: 1 vs 3"),
             title="Different sizes the other way round (1 vs 3)"),
        Case(([], []), 0, title="Two empty images are the same size: 0"),
        Case(lambda: (TEST.images[0], TRAIN.images[0]), key="real-0",
             show="safe_distance(test_images[0], train_images[0])", title="Two real images of 784 pixels"),
        Case(lambda: (TEST.images[0], TEST.images[1][:783]), key="real-short", highlight=True,
             expected=Raises(ValueError, "images have different lengths: 784 vs 783"),
             show="safe_distance(test_images[0], test_images[1][:783])",
             title="A real image with its last pixel missing (784 vs 783)"),
    ],
    detectors=[
        Detector(lambda t: _expects_error(t) and isinstance(t.got, str),
                 "you returned the message as text. Raise it instead: raise ValueError(\"...\")."),
        Detector(lambda t: _expects_error(t) and t.error is None and t.got is None and t.printed.strip(),
                 "you printed the message. Raise it instead, so the code that called safe_distance can't "
                 "miss it: raise ValueError(\"...\")."),
        Detector(lambda t: _expects_error(t) and t.raised(IndexError),
                 "Python raised IndexError inside your loop, because the images have different lengths. "
                 "Compare len(a) and len(b) first, and raise ValueError yourself before the loop."),
        Detector(lambda t: _expects_error(t) and t.error is None,
                 "no error was raised. Before the loop: if len(a) != len(b), raise ValueError(...)."),
        Detector(lambda t: _expects_error(t) and t.raised(ValueError)
                 and str(t.error) == f"images have different lengths: {len(t.args[1])} vs {len(t.args[0])}",
                 "the two lengths are in the wrong order: len(a) first, then len(b)."),
        Detector(lambda t: _expects_error(t) and t.raised(ValueError),
                 "right error type, but the message must read exactly as in the spec. Build it with an "
                 "f-string: f\"images have different lengths: {len(a)} vs {len(b)}\"."),
        Detector(lambda t: not _expects_error(t) and t.raised(ValueError),
                 "you raised ValueError for images of the same length. Raise it only when len(a) != len(b)."),
        Detector(lambda t: _expects_error(t) and t.error is not None and not t.raised(ValueError),
                 "the spec asks for a ValueError (the value of an argument is wrong): raise ValueError(...)."),
    ],
    returns=int,
)


# ============================================================================= SCI1 · fit_centroids + predict_centroid
def _copies(images: list) -> list:
    """Fresh copies of cached images: student code may change what it is given."""
    return [image[:] for image in images]


def _fit_100_seven(fit) -> object:
    return fit(_copies(TRAIN.images[:100]), TRAIN.labels[:100])[7]


def _predict_toy(fit) -> object:
    return _call_partner(fit, "predict_centroid", [1, 1], {0: [0.0, 0.0], 5: [3.0, 3.0], 9: [1.0, 1.5]})


def _predict_real(fit) -> list:
    centroids = fit(_copies(TRAIN.images), TRAIN.labels[:])
    predict = _partner(fit, "predict_centroid")
    if predict is None:
        return None
    return [predict(image[:], centroids) for image in TEST.images[:20]]


def _predict_case(t) -> bool:
    """True when the failing case belongs to predict_centroid (the second function)."""
    return t.show.startswith("predict_centroid") or t.show.startswith("[predict_centroid")


SCI1 = Task(
    "SCI1", "fit_centroids", title="The average digit",
    practise="averages with nested loops, and a dict of lists (Theory S2–S3)",
    input_labels=("Images", "Labels"), expected_label="Expected averages",
    cases=[
        Case(([[0, 2], [4, 7], [10, 10]], [1, 1, 7]), {1: [2.0, 4.5], 7: [10.0, 10.0]}, compare=approx(1e-9),
             title="fit_centroids: two images of a 1 and one image of a 7", highlight=True),
        Case(([], []), {}, title="fit_centroids with no images: an empty dict"),
        Case(call=_fit_100_seven, key="real-seven", compare=approx(1e-6), expected_label="Expected",
             title="fit_centroids on real images: the average 7",
             show="fit_centroids(train_images[:100], train_labels[:100])[7]"),
        Case(call=_predict_toy, expected=9, compare=INT_LABEL, expected_label="Expected label", highlight=True,
             title="predict_centroid: the closest of three averages",
             show="predict_centroid([1, 1], {0: [0.0, 0.0], 5: [3.0, 3.0], 9: [1.0, 1.5]})"),
        Case(call=_predict_real, key="real-predict", expected_label="Expected labels", highlight=True,
             title="predict_centroid on 20 real test images",
             show="[predict_centroid(image, fit_centroids(train_images, train_labels)) for image in test_images[:20]]"),
    ],
    detectors=[
        Detector(lambda t: _predict_case(t) and _partner_state(t, "predict_centroid") == "missing",
                 "fit_centroids gives the right averages. Now write predict_centroid (the cell below it), "
                 "run that cell, then run this check again."),
        Detector(lambda t: _predict_case(t) and _partner_state(t, "predict_centroid") == "stub",
                 "fit_centroids gives the right averages; predict_centroid is still a stub. Write it, run its "
                 "cell, then run this check again."),
        Detector(lambda t: len(t.args) == 2 and isinstance(t.got, list),
                 "you returned a list. The spec asks for a dict: {label: average image}."),
        Detector(lambda t: len(t.args) == 2 and isinstance(t.got, dict) and any(isinstance(k, str) for k in t.got),
                 "your keys are text. Use the label itself as the key: centroids[label]."),
        Detector(lambda t: isinstance(t.got, dict) and t.got.get(1) == [4, 9],
                 "these are totals, not averages: divide each total by how many images have that label."),
        Detector(lambda t: isinstance(t.got, dict) and t.got.get(1) == [2, 4],
                 "you used //, so each average is rounded down. Use / to keep the decimals (4.5, not 4)."),
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.got.get(1), list)
                 and t.got.get(1) and abs(t.got[1][1] - 9 / 3) < 1e-9,
                 "you divided by the number of ALL images. Divide each total by the number of images with "
                 "that label."),
        Detector(lambda t: _predict_case(t) and t.got == 2,
                 "you returned the position of the closest centroid in the dict. Return its label (the key)."),
        Detector(lambda t: _predict_case(t) and isinstance(t.got, float),
                 "predict_centroid returned a float: did you return the distance? Return the label (an int)."),
        Detector(lambda t: _predict_case(t) and isinstance(t.got, list) and t.got and isinstance(t.got[0], float),
                 "predict_centroid returned a list of numbers (a centroid?). Return its label."),
        Detector(lambda t: _predict_case(t) and t.got == 5,
                 "predict_centroid found the farthest centroid. Keep the smallest distance: compare with <."),
    ],
    returns=dict,
)


# ============================================================================= BLD1 · confusion_pairs + top_confusions
def _top(first, *args, **kwargs) -> object:
    return _call_partner(first, "top_confusions", *args, **kwargs)


_TOP_INPUT = {(2, 7): 2, (8, 3): 1, (4, 9): 5, (1, 7): 2}     # (2, 7) before (1, 7): a tie to order


def _top_case(t) -> bool:
    return t.show.startswith("top_confusions")


BLD1 = Task(
    "BLD1", "confusion_pairs", title="Where it fails",
    practise="tuples as dict keys, and sorting with a key (Theory S3, sorted() is a new tool)",
    input_labels=("Predictions", "Ground truth"), expected_label="Expected mistakes",
    cases=[
        Case(([7, 7, 1, 3, 7], [1, 1, 1, 8, 7]), {(1, 7): 2, (8, 3): 1}, highlight=True,
             title="confusion_pairs: a 1 read as 7 twice, an 8 read as 3 once"),
        Case(([4, 2], [4, 2]), {}, title="confusion_pairs with no mistakes: an empty dict"),
        Case(([], []), {}, title="confusion_pairs with empty lists"),
        Case(([9, 9, 4, 4, 9], [4, 4, 9, 4, 7]), {(4, 9): 2, (9, 4): 1, (7, 9): 1},
             title="confusion_pairs: three different mistakes"),
        Case(call=lambda first: _top(first, dict(_TOP_INPUT)), key="top-default", highlight=True,
             expected=[((4, 9), 5), ((1, 7), 2), ((2, 7), 2)], expected_label="Expected list",
             title="top_confusions: the biggest count first; equal counts, the smaller pair first",
             show="top_confusions({(2, 7): 2, (8, 3): 1, (4, 9): 5, (1, 7): 2})"),
        Case(call=lambda first: _top(first, {(3, 5): 1, (5, 3): 4}, n=10), key="top-short",
             expected=[((5, 3), 4), ((3, 5), 1)], expected_label="Expected list",
             title="top_confusions with fewer pairs than n",
             show="top_confusions({(3, 5): 1, (5, 3): 4}, n=10)"),
        Case(call=lambda first: _top(first, dict(_TOP_INPUT), n=1), key="top-one", expected_label="Expected list",
             title="top_confusions with n=1",
             expected=[((4, 9), 5)], show="top_confusions({(2, 7): 2, (8, 3): 1, (4, 9): 5, (1, 7): 2}, n=1)"),
    ],
    detectors=[
        Detector(lambda t: _top_case(t) and _partner_state(t, "top_confusions") == "missing",
                 "confusion_pairs counts the mistakes correctly. Now write top_confusions (the cell below "
                 "it), run that cell, then run this check again."),
        Detector(lambda t: _top_case(t) and _partner_state(t, "top_confusions") == "stub",
                 "confusion_pairs counts the mistakes correctly; top_confusions is still a stub. Write it, "
                 "run its cell, then run this check again."),
        Detector(lambda t: t.raised(TypeError) and "unhashable" in str(t.error),
                 "a list can't be a dict key. Use a tuple: pairs[(truth[i], predictions[i])]."),
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.expected, dict)
                 and any(isinstance(k, tuple) and len(k) == 2 and k[0] == k[1] for k in t.got),
                 "you also counted right answers such as (1, 1). Keep only the mistakes: skip the positions "
                 "where the prediction equals the truth."),
        Detector(lambda t: isinstance(t.got, dict) and isinstance(t.expected, dict) and t.expected
                 and t.got == {(k[1], k[0]): v for k, v in t.expected.items()},
                 "your pairs are (prediction, truth). The spec uses (truth, prediction): \"a 1 read as a 7\" "
                 "is (1, 7)."),
        Detector(lambda t: _top_case(t) and isinstance(t.got, dict),
                 "top_confusions returned a dict. Return a list of (pair, count) tuples: list(pairs.items()) "
                 "gives you one to sort."),
        Detector(lambda t: _top_case(t) and isinstance(t.got, list) and isinstance(t.expected, list)
                 and len(t.got) > len(t.expected),
                 "top_confusions returned too many items: keep only the first n (a slice: items[:n])."),
        Detector(lambda t: _top_case(t) and isinstance(t.got, list) and t.got and isinstance(t.expected, list)
                 and t.got == sorted(t.got, key=lambda item: item[1]) and t.got != t.expected,
                 "the smallest counts come first. Put the biggest count first: sort by -count, or use "
                 "reverse=True."),
        Detector(lambda t: _top_case(t) and isinstance(t.got, list) and isinstance(t.expected, list)
                 and sorted(t.got) == sorted(t.expected) and t.got != t.expected,
                 "the right items, in the wrong order. For equal counts the smaller pair comes first: "
                 "((1, 7), 2) before ((2, 7), 2). A key such as (-count, pair) does both at once."),
    ],
    returns=dict,
)


# ============================================================================= ENG2 · predict_nearest_fast (L2 sanity)
def _eng2_counts(fn, ctx) -> None:
    """The 4th parameter (a stats dict) must be there, and the function must count the pixels it skips."""
    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    if code is None or fn is ctx.reference("predict_nearest"):     # the expected answers' source: no stats
        return
    if not ctx.require(
            code.co_argcount >= 4, case="the def line of predict_nearest_fast",
            expected="4 parameters: image, train_images, train_labels, stats=None",
            got=f"{code.co_argcount} parameter{'s' if code.co_argcount != 1 else ''}",
            cause="keep the 4th parameter stats=None: the experiment below passes a dict, and your function "
                  "adds the pixels it skips to stats[\"skipped\"]."):
        return
    stats: dict = {}
    attempt = ctx.call(fn, TEST.images[0][:], _copies(TRAIN.images[:200]), TRAIN.labels[:200], stats, track=False,
                       show="predict_nearest_fast(test_images[0], train_images[:200], train_labels[:200], stats)")
    skipped = stats.get("skipped") if isinstance(stats, dict) else None
    ok = attempt.error is None and is_number(skipped) and skipped > 0
    ctx.require(ok, case=attempt.show, expected="stats[\"skipped\"] > 0 after the call",
                got=attempt.error_text or f"stats = {short(stats, 50)}",
                cause="add the number of pixels you did NOT look at to stats[\"skipped\"] each time you stop "
                      "early (784 - pixels looked at).")
    if ok:
        ctx.note(f"Skipped {skipped:,} of {200 * 784:,} pixel comparisons on that image "
                 f"({skipped / (200 * 784):.0%}).")


ENG2 = Task(
    "ENG2", "predict_nearest_fast", title="Make it faster",
    practise="early exit with break, and measuring speed",
    reference="predict_nearest",            # same answers as C2: the expected values come from it
    rescue=False,
    input_labels=("Image", "Training images", "Training labels"), expected_label="Expected label",
    cases=[
        Case(([6], [[9], [4], [20], [0]], [7, 3, 8, 5]), 3, title="The tiny world of C2: label 3", highlight=True),
        Case(([5], [[9], [3], [7]], [2, 4, 6]), 4, title="A tie: the earlier image wins, as in C2", highlight=True),
    ] + [Case(lambda i=i: (TEST.images[i], TRAIN.images[:200], TRAIN.labels[:200]), key=f"real-{i}",
              title=f"The same answer as predict_nearest on test image #{i}",
              show=f"predict_nearest_fast(test_images[{i}], train_images[:200], train_labels[:200])")
         for i in (5, 6, 7)],
    detectors=[
        Detector(lambda t: len(t.args) > 2 and t.args[1] and len(t.args[1]) == 3 and t.got == 6,
                 "on a tie you kept the later image. Stop only when the running total is >= the best so far, "
                 "and replace the best only when the full distance is strictly smaller (<)."),
    ],
    returns=int,
    custom=_eng2_counts,
    quality=False,
)


# ============================================================================= SCI2 · predict_knn (L2 sanity)
def _sci2_real(fn, ctx) -> None:
    """k = 1 must give the same label as plain nearest neighbour, on three real images."""
    nearest = ctx.reference("predict_nearest")
    if nearest is None:
        return
    for i in (10, 11, 12):
        args = (TEST.images[i], TRAIN.images[:200], TRAIN.labels[:200])
        attempt = ctx.call(fn, *args, k=1, show=f"predict_knn(test_images[{i}], train_images[:200], "
                                                 f"train_labels[:200], k=1)")
        expected = nearest(*[a[:] if isinstance(a, list) else a for a in args])
        if not ctx.expect(attempt, expected, compare=INT_LABEL):
            return


SCI2 = Task(
    "SCI2", "predict_knn", title="Let the Nearest Images Vote",
    practise="sorting (distance, position) pairs and a majority vote with a dict",
    rescue=False,
    input_labels=("Image", "Training images", "Training labels"), expected_label="Expected label",
    cases=[
        # distances from [5]: 25, 1, 1, 4, 625 -> nearest three: positions 1, 2, 3 -> labels 1, 9, 9
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), 9, kwargs={"k": 3}, highlight=True,
             title="k = 3: the three nearest say 1, 9, 9, so 9 wins the vote"),
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), 1, kwargs={"k": 1},
             title="k = 1: plain nearest neighbour (a tie: the earlier one)"),
        # a 1-1 vote: the closer neighbour ([4], label 2) wins
        Case(([5], [[4], [7], [0]], [2, 6, 3]), 2, kwargs={"k": 2}, highlight=True,
             title="A 1-1 vote: the closer neighbour wins"),
        Case(([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1]), 9, title="k = 3 by default"),
    ],
    detectors=[
        Detector(lambda t: t.kwargs.get("k") == 2 and t.got == 6,
                 "on a 1-1 vote you chose the bigger label. The spec says the closest of the tied "
                 "neighbours wins: walk the k nearest in order of distance and return the first whose label "
                 "has the top vote."),
        Detector(lambda t: isinstance(t.got, list),
                 "you returned a list (the k nearest?). Return the winning label: one int."),
    ],
    returns=int,
    custom=_sci2_real,
    quality=False,
)


# ============================================================================= BLD2 · center_digit (L2 sanity)
def _ink_box(image: list, width: int = 28) -> tuple[int, int, int, int] | None:
    rows = [i // width for i, v in enumerate(image) if v > 0]
    cols = [i % width for i, v in enumerate(image) if v > 0]
    if not rows:
        return None
    return min(rows), max(rows), min(cols), max(cols)


def _corner_toy() -> list[int]:
    """A 7 x 3 block of ink near the top-left corner."""
    image = [0] * 784
    for r in range(1, 8):
        for c in range(2, 5):
            image[r * 28 + c] = 255
    return image


def _shifted(image: list[int], dr: int, dc: int) -> list[int]:
    """`image` moved by (dr, dc) pixels; what falls off the edge is lost."""
    out = [0] * 784
    for r in range(28):
        for c in range(28):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 28 and 0 <= cc < 28:
                out[rr * 28 + cc] = image[r * 28 + c]
    return out


def _bld2_sanity(fn, ctx) -> None:
    box = _ink_box(TEST.images[3])
    moved = _shifted(TEST.images[3], -box[0], -box[2])           # pushed into the top-left corner
    inputs = [("center_digit(a 7 x 3 block of ink near the top-left corner)", _corner_toy()),
              ("center_digit(test_images[3] pushed into the top-left corner)", moved)]
    for show, image in inputs:
        attempt = ctx.call(fn, image, show=show)
        got = attempt.got
        if attempt.error is not None:
            ctx.expect(attempt, "a list of 784 ints")                 # records the error with "Your line"
            return
        ok = isinstance(got, list) and len(got) == 784
        if not ctx.require(ok, case=show, expected="a list of 784 pixels",
                           got=f"{short(got, 40)}" + (f" ({len(got)} values)" if isinstance(got, list) else ""),
                           cause="build a new list of 28 * 28 = 784 zeros, then paste the scaled digit into it."):
            return
        odd = [v for v in got if not (type(v) is int and 0 <= v <= 255)]
        if not ctx.require(not odd, case=show, expected="ints from 0 to 255",
                           got=f"{len(odd)} other values, such as {short(odd[:4], 40)}",
                           cause="copy pixel values from the input image (nearest neighbour), so they stay ints."):
            return
        out_box = _ink_box(got)
        if not ctx.require(out_box is not None, case=show, expected="some ink", got="a blank image",
                           cause="the digit got lost: check the offsets where you paste it."):
            return
        top, bottom, left, right = out_box
        longest = max(bottom - top + 1, right - left + 1)
        if not ctx.require(19 <= longest <= 21, case=show, expected="the longer side of the ink box is 20 px",
                           got=f"{longest} px (rows {top}-{bottom}, columns {left}-{right})",
                           cause="scale factor = 20 / the longer side of the crop; new size = round(old size * scale)."):
            return
        mass = centre_of_mass(got) or (0.0, 0.0)
        box_centre = ((top + bottom) / 2, (left + right) / 2)
        near = any(abs(r - 14) <= 2 and abs(c - 14) <= 2 for r, c in (mass, box_centre))
        if not ctx.require(near, case=show, expected="the ink centred within 2 px of (14, 14)",
                           got=f"ink box centre ({box_centre[0]:.1f}, {box_centre[1]:.1f}), "
                               f"centre of mass ({mass[0]:.1f}, {mass[1]:.1f})",
                           cause="paste the scaled digit at row (28 - height) // 2 and column (28 - width) // 2."):
            return
    blank = ctx.call(fn, [0] * 784, show="center_digit([0] * 784)  (a blank image)")
    if not ctx.expect(blank, [0] * 784):
        return
    # The task is your own version: a call to the provided prepare_drawing gives exactly its output.
    same = all(ctx.call(fn, image, track=False).got == prepare_drawing(image[:]) for _, image in inputs)
    ctx.require(not same, case=inputs[0][0], expected="your own crop, nearest-neighbour scale and paste",
                got="exactly what prepare_drawing returns",
                cause="center_digit hands the work to prepare_drawing. The task is to write those steps "
                      "yourself (the card lists them), then compare the two pictures.")


BLD2 = Task(
    "BLD2", "center_digit", title="Robust drawings",
    practise="cropping, scaling and pasting with 2-D indexing",
    rescue=False,
    custom=_bld2_sanity,
    quality=False,
)


# notebook order: CORE (W, C1, C2, C3, C5), OPTIONAL (C4, ENG1, SCI1, BLD1), ADVANCED (ENG2, SCI2, BLD2)
TASKS = [W, C1, C2, C3, C5, C4, ENG1, SCI1, BLD1, ENG2, SCI2, BLD2]
