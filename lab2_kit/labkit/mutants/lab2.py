"""Lab 2 mutants: buggy versions that a good test must catch (blueprint §4.4 "Rules for testing student tests").

    VOTE_MUTANTS     four buggy majority votes for check("C5", test_majority_vote)
    PREDICT_MUTANTS  six buggy k-NN readers for check("ENG2", test_knn); the first three each have ONE of
                     the three bugs of the "AI wrote this" sample, so a test that catches them has one
                     assert per AI bug
    closest_only     the buggy vote of the live demo before C5 (it is not one of the four)

Each dict maps a description in plain words (printed for every mutant that survives) to the function.
Everything here is wrong on purpose. Don't copy it into your notebook.
"""
from __future__ import annotations


# ============================================================================= buggy votes (C5)
def _count(neighbours):
    counts = {}
    for _distance, label in neighbours:
        counts[label] = counts.get(label, 0) + 1
    return counts


def largest_label(neighbours):
    """Bug: max() of the labels themselves, so the biggest label wins, not the most votes."""
    return max(label for _distance, label in neighbours)


def tie_to_smaller_label(neighbours):
    """Bug: a tie goes to the smaller label instead of the closest tied neighbour."""
    counts = _count(neighbours)
    top = max(counts.values())
    return min(label for label, count in counts.items() if count == top)


def tie_to_farther_neighbour(neighbours):
    """Bug: sorts the labels by count and takes the last one, so a tie goes to the farther neighbour."""
    counts = _count(neighbours)
    return sorted(counts, key=counts.get)[-1]


def ignores_last_neighbour(neighbours):
    """Bug: an off-by-one in the loop, so the last (farthest) neighbour never votes."""
    counts = _count(neighbours[:-1])
    return max(counts, key=counts.get)


def closest_only(neighbours):
    """Bug (the live demo): trusts the closest neighbour and ignores the vote."""
    return neighbours[0][1]


VOTE_MUTANTS = {
    "a vote that returns the largest label": largest_label,
    "a vote that breaks a tie in favour of the smaller label": tie_to_smaller_label,
    "a vote that breaks a tie in favour of the farther neighbour": tie_to_farther_neighbour,
    "a vote that ignores the last neighbour": ignores_last_neighbour,
}


# ============================================================================= buggy readers (ENG2)
def _distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _reader(image, train_images, train_labels, k=3, *, by_label=False, skip_first=False, largest=False,
            fixed_k=None, tuple_sort=False, tie_far=False):
    """One k-NN reader with the chosen bugs switched on."""
    pairs = [(_distance(image, train), label) for train, label in zip(train_images, train_labels)]
    if by_label:
        pairs = sorted(pairs, key=lambda pair: (pair[1], pair[0]))   # like sorting (label, distance) tuples
    elif tuple_sort:
        pairs = sorted(pairs)                                   # ties between distances: smaller label
    else:
        pairs = sorted(pairs, key=lambda pair: pair[0])
    k = fixed_k if fixed_k is not None else k
    nearest = pairs[1:k + 1] if skip_first else pairs[:k]
    if largest:
        return max(label for _distance, label in nearest)
    counts = _count(nearest)
    if tie_far:
        return sorted(counts, key=counts.get)[-1]
    return max(counts, key=counts.get)


def sorts_by_label(image, train_images, train_labels, k=3):
    return _reader(image, train_images, train_labels, k, by_label=True)


def skips_the_nearest(image, train_images, train_labels, k=3):
    return _reader(image, train_images, train_labels, k, skip_first=True)


def votes_for_largest_label(image, train_images, train_labels, k=3):
    return _reader(image, train_images, train_labels, k, largest=True)


def ignores_k(image, train_images, train_labels, k=3):
    return _reader(image, train_images, train_labels, k, fixed_k=3)


def ties_by_label(image, train_images, train_labels, k=3):
    return _reader(image, train_images, train_labels, k, tuple_sort=True)


def tied_vote_to_farther(image, train_images, train_labels, k=3):
    return _reader(image, train_images, train_labels, k, tie_far=True)


def ai_variant(bugs, image, train_images, train_labels, k=3):
    """The "AI wrote this" sample with only some of its bugs left in (bugs: a set of 1, 2 and 3).

    check("AI", ...) uses it to tell which bugs a fixed copy still has: bug 1 sorts (label, distance)
    pairs, bug 2 skips the nearest neighbour with [1:k + 1], bug 3 votes with max(votes).
    """
    return _reader(image, train_images, train_labels, k, by_label=1 in bugs, skip_first=2 in bugs,
                   largest=3 in bugs)


# The first three: one AI bug each (in the order of the "AI wrote this" activity).
AI_BUG_MUTANTS = {
    "a reader that sorts (label, distance) pairs, so it sorts by label (AI bug 1)": sorts_by_label,
    "a reader that skips the nearest neighbour with [1:k + 1] (AI bug 2)": skips_the_nearest,
    "a reader whose vote returns the largest label (AI bug 3)": votes_for_largest_label,
}

PREDICT_MUTANTS = dict(AI_BUG_MUTANTS, **{
    "a reader that ignores k and always asks 3 neighbours": ignores_k,
    "a reader that breaks equal distances by the smaller label (sorted(pairs) without a key)": ties_by_label,
    "a reader that gives a tied vote to the farther neighbour": tied_vote_to_farther,
})
