"""Small helpers for the provided cells of Lab 1's OPTIONAL and ADVANCED parts (standard library only).

    from labkit.lab1_tools import ready, split_train, check_png

    if ready(fit_centroids, predict_centroid):      # False (with a friendly line) while a stub is unfinished
        ...
    fit_images, fit_labels, val_images, val_labels = split_train(train_images, train_labels, n_val=200)
    check_png("output/test_digit_0.png", test_images[0])   # reads a PNG file back and compares every pixel

You never need to change this file; open it if you're curious how the helpers work.
"""
from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path

from . import cases
from .check import is_stub
from .env import current_lab
from .text import say

__all__ = ["ready", "split_train", "check_png"]


# The second function of the two-function tasks: its task, and how to borrow the reference.
PARTNERS = {"predict_centroid": "SCI1", "top_confusions": "BLD1"}


def _help_for(name: str) -> str:
    """'predict_knn (SCI2)' or 'predict_nearest (C2), or run: predict_nearest = rescue("C2")'."""
    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    if lab_cases is None:
        return name
    if name in PARTNERS:
        task = lab_cases.find(PARTNERS[name])
        if task is None:
            return name
        return f"{name} ({task.id}), or run: from labkit.reference.lab{lab} import {name}"
    task = lab_cases.by_name(name)
    if task is None:
        return name
    if task.rescue:
        return f"{name} ({task.id}), or run: {name} = rescue(\"{task.id}\")"
    return f"{name} ({task.id})"


def ready(*functions: object) -> bool:
    """True if every function is written (not a stub); otherwise print what's missing and return False.

    Provided cells use it so that they never crash on a task you haven't done yet:

        if ready(predict_knn):
            ...the experiment...
    """
    missing = []
    for fn in functions:
        if not callable(fn):
            missing.append(f"{fn!r} (not a function)")
        elif is_stub(fn):
            missing.append(_help_for(getattr(fn, "__name__", "a function")))
    if not missing:
        return True
    head = "a function that is still a stub" if len(missing) == 1 else "functions that are still stubs"
    say(f"This cell needs {head}:", *[f"   - {item}" for item in missing],
        "Write it, run its cell, then run this cell again.")
    return False


def split_train(images: list[list[int]], labels: list[int], n_val: int = 200,
                seed: int = 2026) -> tuple[list[list[int]], list[int], list[list[int]], list[int]]:
    """Split TRAINING data into a part to learn from and a validation part to choose settings with.

    Returns (fit_images, fit_labels, val_images, val_labels). The positions are shuffled with a
    fixed seed (the same split every time), then the last `n_val` become the validation set.
    Choose settings (such as k) on the validation set; look at the test set only once, at the end.

    >>> fit_x, fit_y, val_x, val_y = split_train([[0], [1], [2], [3]], [0, 1, 2, 3], n_val=1)
    >>> len(fit_x), len(val_x)
    (3, 1)
    """
    if len(images) != len(labels):
        raise ValueError(f"images and labels have different lengths: {len(images)} vs {len(labels)}")
    if isinstance(n_val, bool) or not isinstance(n_val, int) or not 0 < n_val < len(images):
        raise ValueError(f"n_val must be a whole number between 1 and {len(images) - 1} (got {n_val!r})")
    positions = list(range(len(images)))
    random.Random(seed).shuffle(positions)
    fit_part, val_part = positions[:-n_val], positions[-n_val:]
    return ([images[i] for i in fit_part], [labels[i] for i in fit_part],
            [images[i] for i in val_part], [labels[i] for i in val_part])


# ----------------------------------------------------------------------------- the PNG Challenge
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_problem(data: bytes, image: list[int], width: int) -> str | None:
    """What is wrong with a greyscale PNG of `image` (paper white: 255 - pixel), or None if it is right."""
    if not data.startswith(PNG_SIGNATURE):
        return f"the file doesn't start with the 8-byte PNG signature (it starts with {data[:8]!r})."
    chunks, at = [], len(PNG_SIGNATURE)
    while at < len(data):
        if at + 12 > len(data):
            return f"the file ends in the middle of a chunk header (at byte {at})."
        length, kind = struct.unpack(">I", data[at:at + 4])[0], data[at + 4:at + 8]
        body = data[at + 8:at + 8 + length]
        if len(body) != length or at + 12 + length > len(data):
            return f"chunk {kind!r} says it holds {length} bytes, but the file ends sooner."
        crc = struct.unpack(">I", data[at + 8 + length:at + 12 + length])[0]
        if crc != zlib.crc32(kind + body):
            return f"the CRC of chunk {kind!r} is wrong: it must be zlib.crc32(type + data), packed with \">I\"."
        chunks.append((kind, body))
        at += 12 + length
    kinds = [kind for kind, _ in chunks]
    if not kinds or kinds[0] != b"IHDR":
        return f"the first chunk must be IHDR (the chunks are {kinds})."
    if kinds[-1] != b"IEND":
        return f"the last chunk must be IEND (the chunks are {kinds})."
    if b"IDAT" not in kinds:
        return "there is no IDAT chunk (the pixels)."
    header = chunks[0][1]
    if len(header) != 13:
        return f"IHDR must hold 13 bytes (it holds {len(header)})."
    got_width, got_height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", header)
    height = len(image) // width
    if (got_width, got_height) != (width, height):
        return f"IHDR says {got_width} x {got_height} pixels; this image is {width} x {height}."
    if (depth, colour, interlace) != (8, 0, 0):
        return (f"IHDR says bit depth {depth}, colour type {colour}, interlace {interlace}; an 8-bit greyscale "
                "image without interlacing is 8, 0, 0.")
    try:
        raw = zlib.decompress(b"".join(body for kind, body in chunks if kind == b"IDAT"))
    except zlib.error as error:
        return f"the IDAT data can't be decompressed with zlib ({error})."
    if len(raw) != height * (width + 1):
        return (f"the decompressed pixels are {len(raw)} bytes; {height} rows of (1 filter byte + {width} "
                f"pixels) are {height * (width + 1)}.")
    pixels = []
    for r in range(height):
        row = raw[r * (width + 1):(r + 1) * (width + 1)]
        if row[0] != 0:
            return f"row {r} starts with filter byte {row[0]}; this check reads filter 0 (no filter) only."
        pixels.extend(row[1:])
    paper_white = [255 - value for value in image]
    if pixels == list(image) and pixels != paper_white:
        return "the pixels are the image as it is, so the paper is black. Write 255 - pixel (white paper)."
    for i, (got, want) in enumerate(zip(pixels, paper_white)):
        if got != want:
            return f"pixel (row {i // width}, column {i % width}) is {got}; expected 255 - {image[i]} = {want}."
    return None


def check_png(path: str | Path, image: list[int], width: int = 28) -> bool:
    """Read a greyscale PNG back and compare it with `image`; print PASS or the first problem found.

    It checks the signature, every chunk's CRC, IHDR (size, 8-bit greyscale), that the IDAT data
    decompresses, and every pixel (paper white: 255 - pixel). Returns True when everything matches.
    """
    file = Path(path)
    if not file.is_file():
        say(f"PNG check: {file} doesn't exist yet. Write the function, run its cell, then this cell again.")
        return False
    data = file.read_bytes()
    problem = _png_problem(data, list(image), width)
    if problem:
        say(f"PNG check: NOT YET · {problem}")
        return False
    say(f"PNG check: PASS · {file} ({len(data)} bytes) is a valid {width} x {len(image) // width} greyscale PNG, "
        "and every pixel matches. Open it in VS Code to see the digit.")
    return True
