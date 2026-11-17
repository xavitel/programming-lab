"""Reference code for Lab 2 · Read a Postcode, Pythonically.

Clear, commented versions of the worked example, every CORE task and every OPTIONAL task of Lab 2.
Lab 1 wrote these ideas with loops and counters; here they are written with the idioms of Theory S3-S4:
tuples, dicts, comprehensions, zip, enumerate, sorted / max with a key, and lambda. Each function
gives exactly the same answers as the old code in legacy_reader.py (the golden checks prove it).

Read them to catch up on the lab, to compare with your own code, or borrow one with rescue():

    k_nearest = rescue("C2")

The last section holds the four functions of the bridge notebook for the three weeks between Lab 2 and
Lab 3 (bridge_2to3.ipynb, tasks BR1-BR4): saving and loading predictions with csv, a JSON file of
accuracies, and one line of text taken apart with string methods (a preview of Theory S5-S6).

The data contract (the same in every lab): an image is a list of ints from 0 (paper) to 255 (ink),
row by row from the top-left corner, so pixel (r, c) of a 28 x 28 image is image[r * 28 + c].
A label is an int from 0 to 9. Images and labels travel as two parallel lists.
"""
from __future__ import annotations

import csv
import json
import math


# ============================================================================= W · worked example
def ink_count(image: list[int], threshold: int = 128) -> int:
    """Return how many pixels of `image` are ink: value >= threshold.

    The legacy version needs a while loop, a counter and the magic number 784. Here: add up a 1 for
    every pixel that is ink (a generator expression inside sum).

    >>> ink_count([0, 128, 255, 127])
    2
    """
    return sum(1 for pixel in image if pixel >= threshold)


# ============================================================================= C1
def squared_distance(a: list[int], b: list[int]) -> int:
    """Return the squared distance between two images: the sum of (x - y) ** 2 over matching pixels.

    zip(a, b) walks both images side by side, so there are no positions to manage.
    (Lab 1 wrote the same thing as a running total: for i in range(len(a)): total += (a[i] - b[i]) ** 2.)

    >>> squared_distance([0, 0], [3, 4])
    25
    """
    return sum((x - y) ** 2 for x, y in zip(a, b))


# ============================================================================= C2
def k_nearest(image: list[int], train_images: list[list[int]], train_labels: list[int],
              k: int = 3) -> list[tuple[int, int]]:
    """Return the k nearest training images as (distance, label) pairs, the closest first.

    Equal distances keep the order of the training data: sorted() is stable, and the key looks at the
    distance only. (sorted(pairs) without a key would also compare the labels on a tie.)

    >>> k_nearest([5], [[9], [4], [6], [0]], [7, 8, 3, 5], k=2)
    [(1, 8), (1, 3)]
    """
    pairs = [(squared_distance(image, train), label) for train, label in zip(train_images, train_labels)]
    return sorted(pairs, key=lambda pair: pair[0])[:k]


# ============================================================================= C3
def majority_vote(neighbours: list[tuple[int, int]]) -> int:
    """Return the most frequent label among the neighbours; a tie goes to the closest tied neighbour.

    The neighbours come closest first. A dict remembers the order in which its keys were added, and
    max() returns the FIRST key with the biggest count, so a tie goes to the label seen first: the
    closest of the tied neighbours.

    >>> majority_vote([(2, 7), (5, 1), (9, 7)])
    7
    >>> majority_vote([(2, 4), (5, 1)])         # a 1-1 tie: the closer neighbour wins
    4
    """
    counts = {}
    for _distance, label in neighbours:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


# ============================================================================= C4
def predict_knn(image: list[int], train_images: list[list[int]], train_labels: list[int], k: int = 3) -> int:
    """Read one digit: the k nearest training images vote (C2 + C3, composed).

    >>> predict_knn([5], [[0], [4], [6], [7], [30]], [2, 1, 9, 9, 1], k=3)
    9
    """
    return majority_vote(k_nearest(image, train_images, train_labels, k))


# ============================================================================= C5
def test_majority_vote(vote_fn) -> None:
    """Test a majority-vote function: every assert calls vote_fn (the function under test).

    It passes (runs without an error) on a correct vote, and fails on the usual bugs: the largest
    label instead of the most votes, ignoring a neighbour, or the wrong winner of a tie.
    """
    # a clear majority, where the winner is neither the closest nor the largest label
    assert vote_fn([(1, 9), (2, 4), (3, 4)]) == 4, "two 4s beat one 9"
    # a tie: the closest of the tied neighbours wins
    assert vote_fn([(1, 7), (2, 3)]) == 7, "a 1-1 tie goes to the closer neighbour"
    assert vote_fn([(1, 3), (2, 7)]) == 3, "the same tie the other way round"
    # one neighbour: its label
    assert vote_fn([(5, 6)]) == 6, "a single neighbour decides alone"


# ============================================================================= C6 (OPTIONAL)
def normalise(image: list[int]) -> list[float]:
    """Return every pixel divided by 255 (from 0.0 for paper to 1.0 for ink), as a list comprehension.

    >>> normalise([0, 51, 255])
    [0.0, 0.2, 1.0]
    """
    return [pixel / 255 for pixel in image]


# ============================================================================= ENG1 (OPTIONAL)
def fast_distance(a: list[int], b: list[int]) -> float:
    """Return the Euclidean distance between two images, computed in C by math.dist.

    It is the square root of squared_distance. The square root never changes which of two distances
    is smaller, so the nearest neighbours (and the predictions) stay exactly the same.

    >>> fast_distance([0, 0], [3, 4])
    5.0
    """
    return math.dist(a, b)


# ============================================================================= SCI1 (OPTIONAL)
def downsample(image: list[int]) -> list[int]:
    """Shrink a 28 x 28 image to 14 x 14: each new pixel is the floor average of a 2 x 2 block.

    (a + b + c + d) // 4, the canonical rule of every 14 x 14 file in the kit. The block whose top-left
    pixel is at position i also holds i + 1 (to its right), i + 28 (below) and i + 29 (below right).

    >>> downsample([4] * 784)[:3]
    [4, 4, 4]
    """
    corners = [r * 28 + c for r in range(0, 28, 2) for c in range(0, 28, 2)]   # top-left pixel of each block
    return [(image[i] + image[i + 1] + image[i + 28] + image[i + 29]) // 4 for i in corners]


# ============================================================================= BLD1 (OPTIONAL)
def per_digit_accuracy(predictions: list[int], truth: list[int]) -> dict[int, float]:
    """Return {digit: fraction read correctly} for every digit that appears in `truth`.

    Each digit is judged on the images whose TRUE label is that digit.

    >>> per_digit_accuracy([7, 1, 1, 3], [7, 7, 1, 3])
    {1: 1.0, 3: 1.0, 7: 0.5}
    """
    pairs = list(zip(predictions, truth))
    return {digit: sum(1 for p, t in pairs if t == digit and p == t) / sum(1 for _, t in pairs if t == digit)
            for digit in sorted(set(truth))}


def report_card(acc: dict[int, float]) -> str:
    """Return an aligned table, the worst digit first (equal accuracies: the smaller digit first).

    One header line, then one line per digit: rank (from 1), digit and accuracy as a percentage.

    >>> print(report_card({7: 0.5, 1: 1.0}))
    Rank  Digit  Accuracy
       1      7     50.0%
       2      1    100.0%
    """
    lines = [f"{'Rank':>4}  {'Digit':>5}  {'Accuracy':>8}"]
    worst_first = sorted(acc, key=lambda digit: (acc[digit], digit))
    for rank, digit in enumerate(worst_first, start=1):
        lines.append(f"{rank:>4}  {digit:>5}  {acc[digit]:>8.1%}")
    return "\n".join(lines)


# ============================================================================= bridge 2 -> 3 (BR1-BR4)
def save_predictions(predictions: list[int], truth: list[int], path: str) -> None:
    """Write a CSV file: a header row, then one row per image with its position, prediction and true label.

    save_predictions([7, 2], [7, 3], "output/predictions.csv") writes
        image,prediction,truth
        0,7,7
        1,2,3
    """
    # newline="": the csv module writes its own line ends (without it, Windows gets an empty row after each row)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction", "truth"])
        for image, (prediction, label) in enumerate(zip(predictions, truth)):
            writer.writerow([image, prediction, label])


def load_predictions(path: str) -> tuple[list[int], list[int]]:
    """Read a file written by save_predictions back into two lists of ints: (predictions, truth).

    csv.DictReader reads the header for us and gives each row as a dict, so the columns are found by
    their names. Everything in a CSV file is text: int() turns "7" back into 7.
    """
    predictions, truth = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            predictions.append(int(row["prediction"]))
            truth.append(int(row["truth"]))
    return predictions, truth


def save_accuracy(per_digit: dict[int, float], path: str) -> None:
    """Write {digit: accuracy} as an indented JSON file, one digit per line.

    JSON keys are always text, so the digit 7 is written as "7" (and json.load gives "7" back).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(per_digit, f, indent=2)


def parse_depot_line(line: str) -> tuple[str, str] | None:
    """Take one line of the depot sheet apart: "  28001 ; Retiro " -> ("28001", "Retiro").

    A line that is not "postcode ; depot" (no ";", more than one, or an empty part) gives None: a value
    we can't trust is not guessed. The postcode stays text (a postcode such as 00046 keeps its zeros).

    >>> parse_depot_line("  28001 ; Retiro ")
    ('28001', 'Retiro')
    >>> parse_depot_line("28001 Retiro") is None
    True
    """
    parts = line.split(";")
    if len(parts) != 2:
        return None
    postcode, depot = parts[0].strip(), parts[1].strip()
    if not postcode or not depot:
        return None
    return postcode, depot


# ============================================================================= helpers for the checks
# (Not tasks. generators/gen_cases.py runs them once to store "facts" that the misconception
# detectors in cases/lab2.py compare with, e.g. "you sorted the tuples, so ties went by label".)
def _pairs(image: list[int], train_images: list[list[int]], train_labels: list[int]) -> list[tuple[int, int]]:
    return [(squared_distance(image, train), label) for train, label in zip(train_images, train_labels)]


def nearest_sorted_tuples(image, train_images, train_labels, k: int = 3) -> list[tuple[int, int]]:
    """sorted(pairs)[:k]: ties broken by the smaller LABEL (the tuple's second item)."""
    return sorted(_pairs(image, train_images, train_labels))[:k]


def nearest_unsorted(image, train_images, train_labels, k: int = 3) -> list[tuple[int, int]]:
    """The first k pairs in training order (no sorting at all)."""
    return _pairs(image, train_images, train_labels)[:k]


def farthest_first(image, train_images, train_labels, k: int = 3) -> list[tuple[int, int]]:
    """The k FARTHEST pairs, farthest first (a sort in the wrong direction)."""
    return sorted(_pairs(image, train_images, train_labels), key=lambda pair: pair[0], reverse=True)[:k]


def nearest_label_first(image, train_images, train_labels, k: int = 3) -> list[tuple[int, int]]:
    """The right neighbours, written (label, distance) instead of (distance, label)."""
    return [(label, distance) for distance, label in k_nearest(image, train_images, train_labels, k)]


def nearest_by_label(image, train_images, train_labels, k: int = 3) -> list[tuple[int, int]]:
    """The first k pairs sorted by their LABEL (a key that looks at pair[1] instead of the distance)."""
    return sorted(_pairs(image, train_images, train_labels), key=lambda pair: pair[1])[:k]


def farthest_last_k(image, train_images, train_labels, k: int = 3) -> list[tuple[int, int]]:
    """The LAST k pairs of the sorted list (a slice [-k:] instead of [:k]): the farthest ones."""
    return sorted(_pairs(image, train_images, train_labels), key=lambda pair: pair[0])[-k:]
