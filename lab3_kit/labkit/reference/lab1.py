"""Reference code for Lab 1 · Read One Digit.

Clear, commented versions of the worked example, every CORE task and every OPTIONAL task of
Lab 1, written in plain loop style. Read them to catch up on the lab, to compare with your own
code, or borrow one with rescue() when you are stuck:

    predict_nearest = rescue("C2")

Later labs use these functions too (Labs 4 and 5 use fit_centroids, predict_centroid and
render_digit), so their names and signatures never change.

The data contract (the same in every lab): an image is a list of ints from 0 (paper) to 255 (ink),
row by row from the top-left corner, so pixel (r, c) of a 28 x 28 image is image[r * 28 + c].
A label is an int from 0 to 9. Images and labels travel as two parallel lists.
"""
from __future__ import annotations


# ============================================================================= W · worked example
def ink_count(image: list[int], threshold: int = 128) -> int:
    """Return how many pixels of `image` are ink: value >= threshold.

    >>> ink_count([0, 128, 255, 127])
    2
    """
    count = 0                          # the running count starts at zero
    for pixel in image:                # visit every pixel once
        if pixel >= threshold:         # >= : a pixel equal to the threshold counts as ink
            count += 1
    return count                       # after the loop, not inside it


# ============================================================================= C1
def squared_distance(a: list[int], b: list[int]) -> int:
    """Return the squared Euclidean distance between two images of equal length.

    It adds up (a[i] - b[i]) ** 2 over every position i: 0 for identical images, and the more
    two images differ, the bigger the number. No square root: to find the closest image we
    only compare distances, and the square root never changes which one is smallest.

    >>> squared_distance([0, 0], [3, 4])
    25
    """
    total = 0                                  # running total
    for i in range(len(a)):                    # positions 0, 1, ..., len(a) - 1
        difference = a[i] - b[i]
        total += difference * difference       # = (a[i] - b[i]) ** 2, and about 4 times faster on Python 3.10
    return total


# ============================================================================= C2
def predict_nearest(image: list[int], train_images: list[list[int]], train_labels: list[int]) -> int:
    """Return the label of the training image closest to `image` (nearest neighbour).

    Closest = the smallest squared_distance. If two training images are equally close, the
    earlier one wins.

    >>> predict_nearest([6], [[9], [4], [20], [0]], [7, 3, 8, 5])
    3
    """
    best_distance = float("inf")               # bigger than any real distance: the first image always wins
    best_index = 0                             # position of the best image so far
    for i in range(len(train_images)):
        distance = squared_distance(image, train_images[i])
        if distance < best_distance:           # strictly smaller: on a tie, the earlier image stays
            best_distance = distance
            best_index = i
    return train_labels[best_index]            # the LABEL of the nearest image, not its position


def nearest_index(image: list[int], train_images: list[list[int]], train_labels: list[int]) -> int:
    """Return the POSITION of the nearest training image (ties: the earliest).

    A helper for the checks: it lets check("C2", ...) recognise "you returned the index".
    """
    best_distance = float("inf")
    best_index = 0
    for i in range(len(train_images)):
        distance = squared_distance(image, train_images[i])
        if distance < best_distance:
            best_distance = distance
            best_index = i
    return best_index


# ============================================================================= C3
def accuracy(predictions: list[int], truth: list[int]) -> float:
    """Return the fraction of predictions that equal the truth, between 0.0 and 1.0.

    Empty lists give 0.0 (there is nothing to divide by).

    >>> accuracy([1, 2, 3], [1, 2, 4])
    0.6666666666666666
    """
    if len(truth) == 0:                        # guard the empty case first
        return 0.0
    correct = 0
    for i in range(len(truth)):
        if predictions[i] == truth[i]:
            correct += 1
    return correct / len(truth)                # / always gives a float in Python 3


# ============================================================================= C4
def count_labels(labels: list[int]) -> dict[int, int]:
    """Return how many times each label appears, as {label: count}.

    >>> count_labels([3, 1, 3])
    {3: 2, 1: 1}
    """
    counts: dict[int, int] = {}
    for label in labels:
        # .get(label, 0) gives 0 the first time a label is seen. The long way is:
        #     if label in counts: counts[label] += 1
        #     else:               counts[label] = 1
        counts[label] = counts.get(label, 0) + 1
    return counts


# ============================================================================= C5
def render_digit(image: list[int], width: int = 28, threshold: int = 128) -> str:
    """Return a text picture of `image`: one line per row, '#' for ink (>= threshold), '.' for paper.

    The lines are joined with newlines, with no newline after the last one:
    >>> print(render_digit([0, 200, 255, 10], width=2))
    .#
    #.
    """
    lines = []
    for r in range(len(image) // width):       # one line per row
        line = ""
        for c in range(width):                 # one character per pixel of that row
            if image[r * width + c] >= threshold:
                line += "#"
            else:
                line += "."
        lines.append(line)
    return "\n".join(lines)                    # "\n" between the lines, none after the last one


# ============================================================================= ENG1 (OPTIONAL)
def safe_distance(a: list[int], b: list[int]) -> int:
    """Like squared_distance, but refuse images of different lengths.

    Raises ValueError("images have different lengths: 784 vs 783") instead of silently giving a
    wrong answer (or an IndexError deep inside the loop).

    >>> safe_distance([0, 0], [3, 4])
    25
    """
    if len(a) != len(b):
        raise ValueError(f"images have different lengths: {len(a)} vs {len(b)}")
    return squared_distance(a, b)


# ============================================================================= SCI1 (OPTIONAL)
def fit_centroids(images: list[list[int]], labels: list[int]) -> dict[int, list[float]]:
    """Return the average image ("centroid") of each label, as {label: list of floats}.

    Pixel p of the centroid of digit d is the average of pixel p over all images labelled d.

    >>> fit_centroids([[0, 2], [4, 7], [10, 10]], [1, 1, 7])
    {1: [2.0, 4.5], 7: [10.0, 10.0]}
    """
    totals: dict[int, list[int]] = {}           # label -> running total of each pixel
    counts: dict[int, int] = {}                 # label -> how many images had that label
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        if label not in totals:
            totals[label] = [0] * len(image)    # a NEW list of zeros (never the image itself,
            counts[label] = 0                   # or the loop below would change your data)
        running = totals[label]
        for p in range(len(image)):
            running[p] += image[p]
        counts[label] += 1
    centroids: dict[int, list[float]] = {}
    for label in totals:
        average = []
        for total in totals[label]:
            average.append(total / counts[label])   # / keeps the decimals
        centroids[label] = average
    return centroids


def predict_centroid(image: list[int], centroids: dict[int, list[float]]) -> int:
    """Return the label whose centroid is closest to `image` (the smallest squared distance).

    Ten comparisons instead of a thousand: much faster than predict_nearest, a little less
    accurate. If two centroids are equally close, the one that comes first in the dict wins.

    >>> predict_centroid([1, 1], {0: [0.0, 0.0], 5: [3.0, 3.0], 9: [1.0, 1.5]})
    9
    """
    best_label = -1
    best_distance = float("inf")
    for label in centroids:
        distance = squared_distance(image, centroids[label])   # works with floats too
        if distance < best_distance:
            best_distance = distance
            best_label = label
    return best_label


# ============================================================================= BLD1 (OPTIONAL)
def confusion_pairs(predictions: list[int], truth: list[int]) -> dict[tuple[int, int], int]:
    """Count the mistakes, as {(true label, predicted label): how many times}.

    Only wrong answers are counted: a 4 read as a 9 adds 1 to the pair (4, 9).

    >>> confusion_pairs([7, 7, 1, 3, 7], [1, 1, 1, 8, 7])
    {(1, 7): 2, (8, 3): 1}
    """
    pairs: dict[tuple[int, int], int] = {}
    for i in range(len(truth)):
        if predictions[i] != truth[i]:              # skip the right answers
            pair = (truth[i], predictions[i])       # (what it was, what the machine said)
            pairs[pair] = pairs.get(pair, 0) + 1
    return pairs


def top_confusions(pairs: dict[tuple[int, int], int], n: int = 3) -> list[tuple[tuple[int, int], int]]:
    """Return the n most frequent mistakes as [((true, predicted), count), ...].

    Order: the biggest count first; for equal counts, the smaller pair first.

    >>> top_confusions({(2, 7): 2, (8, 3): 1, (4, 9): 5, (1, 7): 2})
    [((4, 9), 5), ((1, 7), 2), ((2, 7), 2)]
    """
    items = list(pairs.items())                       # [((true, predicted), count), ...]
    # Sort by (minus the count, the pair): -count puts big counts first, then pairs ascend.
    items.sort(key=lambda item: (-item[1], item[0]))
    return items[:n]
