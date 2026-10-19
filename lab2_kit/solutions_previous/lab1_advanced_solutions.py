"""Lab 1 · Read One Digit: the ADVANCED solutions (ENG2, SCI2, BLD2 and the Challenge).

They were kept back until now so that the Challenge could stay open; here they are. Each function is one way to
solve its task, written to be read: compare it with yours. The full Lab 1 notebook, with these solutions in
place, is next to this file: lab1_solution_full.ipynb.

Contents
    ENG2  predict_nearest_fast  early exit + counting the pixel comparisons it skipped
    SCI2  predict_knn           majority vote, a tie goes to the closest tied neighbour
          choose_k              choosing k honestly, on a validation split of the TRAINING data
    BLD2  center_digit          crop to the ink, scale to 20 px (nearest neighbour), paste centred
    CH    my_fit / my_predict   League entry 1: an averaged multiclass perceptron on 14 x 14 images
          fit_subcentroids / predict_subcentroids   League entry 2: several average images per digit
          write_png             the PNG encoder alternative (zlib + struct)
          numpy_nearest         the numpy division alternative (1-NN over the full 60,000, validation only)

To try them, open a notebook in the kit folder (the folder with labkit/), then for example:
    import sys; sys.path.insert(0, "solutions_previous")
    from lab1_advanced_solutions import predict_knn
"""
from __future__ import annotations

import math
import random
import struct
import time
import zlib
from operator import mul
from pathlib import Path

from labkit import downsample                                     # noqa: F401
from labkit.reference.lab1 import accuracy, squared_distance      # noqa: F401


# ============================================================================= ENG2
def predict_nearest_fast(image: list[int], train_images: list[list[int]], train_labels: list[int],
                         stats: dict | None = None) -> int:
    """Like predict_nearest, but stop adding up a distance as soon as it can't win.

    Same answers as predict_nearest (ties: the earliest image). If `stats` is a dict, it counts
    stats["skipped"] (pixel comparisons skipped) and stats["compared"] (training images visited).
    """
    best_distance = float("inf")
    best_index = 0
    for i in range(len(train_images)):
        candidate = train_images[i]
        total = 0
        finished = True                           # did we add up every pixel?
        for p in range(len(image)):
            difference = image[p] - candidate[p]
            total += difference * difference
            if total >= best_distance:            # already as far as the best: it can't win any more
                finished = False
                if stats is not None:
                    stats["skipped"] = stats.get("skipped", 0) + len(image) - p - 1
                break
        if finished:                              # a complete distance that is strictly smaller
            best_distance = total
            best_index = i
        if stats is not None:
            stats["compared"] = stats.get("compared", 0) + 1
    return train_labels[best_index]


# ============================================================================= SCI2
def predict_knn(image: list[int], train_images: list[list[int]], train_labels: list[int], k: int = 3) -> int:
    """Majority vote of the k nearest training images; a tied vote goes to the closest tied neighbour."""
    neighbours = []
    for i in range(len(train_images)):
        neighbours.append((squared_distance(image, train_images[i]), i))
    neighbours.sort()                             # nearest first; equal distances: the earlier position first
    nearest = neighbours[:k]
    votes: dict[int, int] = {}
    for distance, i in nearest:
        label = train_labels[i]
        votes[label] = votes.get(label, 0) + 1
    top = max(votes.values())
    for distance, i in nearest:                   # walk from the closest: the first label with the top vote wins
        if votes[train_labels[i]] == top:
            return train_labels[i]
    return train_labels[nearest[0][1]]            # not reached


def choose_k(fit_images: list[list[int]], fit_labels: list[int], val_images: list[list[int]],
             val_labels: list[int], ks: tuple[int, ...] = (1, 3, 5, 7)) -> dict[int, float]:
    """Validation accuracy of predict_knn for each k (the notebook prints each line as it goes)."""
    scores: dict[int, float] = {}
    for k in ks:
        started = time.perf_counter()
        guesses = [predict_knn(image, fit_images, fit_labels, k=k) for image in val_images]
        scores[k] = accuracy(guesses, val_labels)
        print(f"k = {k}: {scores[k]:.1%} on the validation set ({time.perf_counter() - started:.1f} s)")
    return scores


# ============================================================================= BLD2
def center_digit(image: list[int]) -> list[int]:
    """Crop to the ink (pixels > 0), scale the longer side to 20 px (nearest neighbour), paste centred."""
    rows = [r for r in range(28) if any(image[r * 28 + c] > 0 for c in range(28))]
    cols = [c for c in range(28) if any(image[r * 28 + c] > 0 for r in range(28))]
    if not rows:                                  # a blank image stays blank
        return [0] * 784
    top, left = rows[0], cols[0]
    height, width = rows[-1] - top + 1, cols[-1] - left + 1
    scale = 20 / max(height, width)               # the longer side becomes 20 pixels
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    row_offset = (28 - new_height) // 2           # where the scaled digit starts, so it sits in the middle
    col_offset = (28 - new_width) // 2
    result = [0] * 784
    for r in range(new_height):
        source_r = top + min(height - 1, int(r / scale))      # nearest neighbour: the pixel it came from
        for c in range(new_width):
            source_c = left + min(width - 1, int(c / scale))
            result[(row_offset + r) * 28 + col_offset + c] = image[source_r * 28 + source_c]
    return result


# ============================================================================= CH entry 1: averaged perceptron
def my_fit(train_images: list[list[int]], train_labels: list[int]) -> list[list[float]]:
    """An averaged multiclass perceptron on 14 x 14 images: one weight list per digit.

    The Rosenblatt rule: guess the digit with the highest score (weights · pixels). When the
    guess is wrong, add the image to the right digit's weights and subtract it from the guessed
    digit's weights. Averaging the weights over all steps makes the result much steadier.
    """
    images = [downsample(image) + [255] for image in train_images]    # 196 pixels + a constant "bias" input
    size = len(images[0])
    weights = [[0] * size for _ in range(10)]
    history = [[0] * size for _ in range(10)]      # step-weighted updates, for the average
    order = list(range(len(images)))
    shuffler = random.Random(2026)
    step = 1
    for epoch in range(6):
        shuffler.shuffle(order)
        for i in order:
            x, truth = images[i], train_labels[i]
            scores = [sum(map(mul, w, x)) for w in weights]
            guess = scores.index(max(scores))
            if guess != truth:
                right, wrong = weights[truth], weights[guess]
                right_h, wrong_h = history[truth], history[guess]
                for p in range(size):
                    value = x[p]
                    if value:
                        right[p] += value
                        wrong[p] -= value
                        right_h[p] += step * value
                        wrong_h[p] -= step * value
            step += 1
    return [[w[p] - h[p] / step for p in range(size)] for w, h in zip(weights, history)]


def my_predict(image: list[int], model: list[list[float]]) -> int:
    """The digit whose weights give the highest score."""
    x = downsample(image) + [255]
    scores = [sum(map(mul, w, x)) for w in model]
    return scores.index(max(scores))


# ============================================================================= CH entry 2: several centroids per digit
def fit_subcentroids(train_images: list[list[int]], train_labels: list[int], per_digit: int = 12,
                     rounds: int = 6) -> list[tuple[list[float], int]]:
    """k-means inside each digit (14 x 14): several 'average images' per digit catch different styles."""
    shuffler = random.Random(2026)
    model = []
    for digit in range(10):
        group = [downsample(img) for img, label in zip(train_images, train_labels) if label == digit]
        centres = [list(map(float, img)) for img in shuffler.sample(group, per_digit)]
        for _ in range(rounds):
            sums = [[0.0] * 196 for _ in centres]
            counts = [0] * len(centres)
            for img in group:
                distances = [math.dist(img, c) for c in centres]
                j = distances.index(min(distances))
                counts[j] += 1
                s = sums[j]
                for p in range(196):
                    s[p] += img[p]
            centres = [[v / counts[j] for v in sums[j]] if counts[j] else centres[j] for j in range(len(centres))]
        model += [(c, digit) for c in centres]
    return model


def predict_subcentroids(image: list[int], model: list[tuple[list[float], int]]) -> int:
    x = downsample(image)
    best, best_label = float("inf"), -1
    for centre, label in model:
        d = math.dist(x, centre)
        if d < best:
            best, best_label = d, label
    return best_label


# ============================================================================= CH alternative: PNG encoder
def write_png(image: list[int], path: str | Path, width: int = 28) -> None:
    """Save `image` as an 8-bit greyscale PNG (ink dark on white paper), standard library only."""
    height = len(image) // width

    def chunk(kind: bytes, data: bytes) -> bytes:
        # length, type, data, then a CRC-32 checksum of type + data (PNG spec, section 5.3)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = bytearray()
    for r in range(height):
        raw.append(0)                                         # filter type 0 ("None") before each row
        raw.extend(255 - image[r * width + c] for c in range(width))   # 0 = paper -> white
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)   # 8-bit greyscale, no interlace
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


# ============================================================================= CH alternative: numpy division
def numpy_nearest(npz_path: str | Path = "data/full/mnist.npz", n_val: int = 1000) -> dict:
    """1-NN over the full MNIST training set with numpy, scored on a VALIDATION split (never x_test).

    The last 10,000 training images are held out for validation; we predict n_val of them.
    """
    import numpy as np

    data = np.load(npz_path, allow_pickle=False)
    x = data["x_train"].reshape(len(data["x_train"]), -1).astype(np.float32)    # 60,000 x 784
    y = data["y_train"]
    train_x, train_y = x[:-10_000], y[:-10_000]               # learn from 50,000
    val_x, val_y = x[-10_000:][:n_val], y[-10_000:][:n_val]   # validate on images it never saw
    started = time.perf_counter()
    # |a - b|^2 = |a|^2 - 2 a.b + |b|^2; |a|^2 is the same for every candidate, so we can drop it
    train_sq = (train_x ** 2).sum(axis=1)
    predictions = []
    for start in range(0, len(val_x), 250):                   # 250 at a time keeps memory small
        block = val_x[start:start + 250]
        distances = train_sq[None, :] - 2.0 * block @ train_x.T
        predictions.append(train_y[distances.argmin(axis=1)])
    predicted = np.concatenate(predictions)
    seconds = time.perf_counter() - started
    return {"accuracy": float((predicted == val_y).mean()), "seconds": seconds, "n_val": len(val_y),
            "n_train": len(train_y)}
