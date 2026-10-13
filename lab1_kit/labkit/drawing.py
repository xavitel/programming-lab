"""Your own handwriting as an image the model can read.

Two routes to a 28 x 28 image (784 ints, 0 = paper, 255 = ink):

    image = sketch_to_image(sketch)     # keyboard route: a 14 x 14 grid of '#' (ink) and '.' (paper)
    image = prepare_drawing(drawing)    # canvas route: the 784 numbers copied from tools/draw_digit.html

prepare_drawing does what the MNIST authors did to every digit: crop to the ink, scale the longer
side to 20 pixels, and place it so its centre of mass sits at the centre of a 28 x 28 frame.
"""
from __future__ import annotations

import math
import re
import textwrap

from .text import say

SIZE = 28                 # output images are 28 x 28
BOX = 20                  # MNIST digits fit in a 20 x 20 box
CENTRE = 14.0             # MNIST's average centre of mass (row and column), measured on the kit data
PAPER_CHARS = set(". _-")


# ----------------------------------------------------------------------------- small image tools
def _rows(pixels: list, width: int) -> list[list[float]]:
    return [list(pixels[r * width:(r + 1) * width]) for r in range(len(pixels) // width)]


def _ink_box(grid: list[list[float]], threshold: float = 0) -> tuple[int, int, int, int] | None:
    """(top, bottom, left, right) of the pixels above `threshold`, or None for a blank image."""
    rows = [r for r, row in enumerate(grid) if any(v > threshold for v in row)]
    cols = [c for c in range(len(grid[0])) if any(row[c] > threshold for row in grid)]
    if not rows:
        return None
    return rows[0], rows[-1], cols[0], cols[-1]


def centre_of_mass(image: list[int], width: int = SIZE) -> tuple[float, float] | None:
    """The ink-weighted average (row, column) of an image, or None if it is blank."""
    total = sum(image)
    if total <= 0:
        return None
    row = sum((i // width) * v for i, v in enumerate(image)) / total
    col = sum((i % width) * v for i, v in enumerate(image)) / total
    return row, col


def _resize(grid: list[list[float]], new_h: int, new_w: int) -> list[list[float]]:
    """Bilinear resize of a small grid (each output pixel mixes its 4 nearest input pixels)."""
    h, w = len(grid), len(grid[0])
    out = []
    for y in range(new_h):
        sy = min(max((y + 0.5) * h / new_h - 0.5, 0.0), h - 1.0)
        y0 = int(sy)
        y1, fy = min(y0 + 1, h - 1), sy - y0
        row = []
        for x in range(new_w):
            sx = min(max((x + 0.5) * w / new_w - 0.5, 0.0), w - 1.0)
            x0 = int(sx)
            x1, fx = min(x0 + 1, w - 1), sx - x0
            top = grid[y0][x0] * (1 - fx) + grid[y0][x1] * fx
            bottom = grid[y1][x0] * (1 - fx) + grid[y1][x1] * fx
            row.append(top * (1 - fy) + bottom * fy)
        out.append(row)
    return out


def _place_centred(grid: list[list[float]]) -> list[int]:
    """Paste a grid (at most 28 x 28) into a 28 x 28 frame with its centre of mass at the centre."""
    h, w = len(grid), len(grid[0])
    top, left = (SIZE - h) // 2, (SIZE - w) // 2
    total = sum(sum(row) for row in grid)
    if total > 0:
        com_r = sum(r * v for r, row in enumerate(grid) for v in row) / total + top
        com_c = sum(c * v for row in grid for c, v in enumerate(row)) / total + left
        top += max(-top, min(SIZE - h - top, round(CENTRE - com_r)))      # move, but stay inside
        left += max(-left, min(SIZE - w - left, round(CENTRE - com_c)))
    image = [0] * (SIZE * SIZE)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            image[(top + r) * SIZE + left + c] = int(min(255, max(0, round(value))))
    return image


def _fit_box(grid: list[list[float]], box: int = BOX) -> list[list[float]]:
    """Crop to the ink and scale so the longer side is `box` pixels (keeping the shape)."""
    top, bottom, left, right = _ink_box(grid)
    crop = [row[left:right + 1] for row in grid[top:bottom + 1]]
    h, w = len(crop), len(crop[0])
    scale = box / max(h, w)
    return _resize(crop, max(1, round(h * scale)), max(1, round(w * scale)))


def _soften(image: list[int]) -> list[int]:
    """A light blur that keeps strokes dark and gives them soft edges, like a pen on paper."""
    weights = ((-1, -1, 1), (-1, 0, 2), (-1, 1, 1), (0, -1, 2), (0, 0, 4), (0, 1, 2), (1, -1, 1), (1, 0, 2), (1, 1, 1))
    out = []
    for r in range(SIZE):
        for c in range(SIZE):
            total = 0
            for dr, dc, weight in weights:
                rr, cc = r + dr, c + dc
                if 0 <= rr < SIZE and 0 <= cc < SIZE:
                    total += image[rr * SIZE + cc] * weight
            out.append(max(image[r * SIZE + c], round(total / 16)))
    return out


# ----------------------------------------------------------------------------- the keyboard route
def sketch_to_image(sketch: str) -> list[int]:
    """Turn a typed 14 x 14 sketch ('#' = ink, '.' = paper) into a 28 x 28 image.

    Each sketch cell becomes a 2 x 2 block, the digit is centred, and the edges are softened.
    Shorter or narrower sketches are padded with paper; a sketch bigger than 14 x 14 is refused.

    >>> len(sketch_to_image("..##..\\n.#..#.\\n....#.\\n...#..\\n..#...\\n.####."))
    784
    """
    if not isinstance(sketch, str):
        raise TypeError(f"sketch_to_image needs a text sketch (a str), not {type(sketch).__name__}")
    lines = [line.rstrip() for line in textwrap.dedent(sketch.expandtabs(4)).split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("the sketch is empty: draw your digit with # (ink) and . (paper)")
    if len(lines) > 14:
        raise ValueError(f"the sketch has {len(lines)} rows; the grid is at most 14 x 14")
    for number, line in enumerate(lines, start=1):
        if len(line) > 14:
            raise ValueError(f"row {number} of the sketch has {len(line)} characters; at most 14 fit")
    big = []
    for line in lines + [""] * (14 - len(lines)):
        row = [0 if ch in PAPER_CHARS else 255 for ch in line.ljust(14)]
        doubled = [value for value in row for _ in range(2)]
        big += [doubled, doubled[:]]
    if _ink_box(big) is None:
        say("sketch_to_image: the sketch has no ink (#), so the image is blank.")
        return [0] * (SIZE * SIZE)
    top, bottom, left, right = _ink_box(big)
    if max(bottom - top, right - left) + 1 > BOX + 2:     # a very big sketch: shrink it like MNIST
        big = _fit_box(big)
    else:
        big = [row[left:right + 1] for row in big[top:bottom + 1]]
    return _soften(_place_centred(big))


# ----------------------------------------------------------------------------- the canvas route
def parse_drawing(text: str) -> list[int]:
    """Read the 784 numbers copied from draw_digit.html (28 lines of 28 numbers, 0-255).

    Any mix of spaces, new lines, commas and brackets between the numbers is fine.
    """
    if isinstance(text, (list, tuple)):
        tokens = [v for item in text for v in (item if isinstance(item, (list, tuple)) else [item])]
    elif isinstance(text, str):
        tokens = [t for t in re.split(r"[\s,;\[\]()]+", text) if t]
    else:
        raise TypeError("parse_drawing needs the text you copied from draw_digit.html "
                        f"(a str), not {type(text).__name__}")
    pixels = []
    for token in tokens:
        try:
            value = float(token)
        except (TypeError, ValueError):
            raise ValueError(f"{token!r} is not a number; paste only the numbers from draw_digit.html") from None
        if not 0 <= value <= 255:
            raise ValueError(f"{token} is outside 0-255; paste only the numbers from draw_digit.html")
        pixels.append(int(round(value)))
    if len(pixels) != SIZE * SIZE:
        raise ValueError(f"I found {len(pixels)} numbers; a drawing has 784 (28 lines of 28). "
                         "Click \"Copy 784 numbers\" again and paste everything.")
    return pixels


def prepare_drawing(text_or_image) -> list[int]:
    """Make a raw drawing look like an MNIST digit, and return it as a 28 x 28 image.

    Accepts the text copied from draw_digit.html or a list of 784 numbers. Steps: crop to the
    bounding box of the ink (pixels > 0), scale the longer side to 20 pixels, then place it so the
    centre of mass is at the centre of the 28 x 28 frame.
    """
    pixels = parse_drawing(text_or_image)
    grid = _rows(pixels, SIZE)
    if _ink_box(grid) is None:
        say("prepare_drawing: the drawing is blank (no ink), so the image is blank too.")
        return [0] * (SIZE * SIZE)
    return _place_centred(_fit_box(grid))


def ink_centre_offset(image: list[int]) -> float:
    """How far (in pixels) the centre of mass is from the centre of the frame; 0.0 = perfectly centred."""
    com = centre_of_mass(image)
    return 0.0 if com is None else math.hypot(com[0] - CENTRE, com[1] - CENTRE)
