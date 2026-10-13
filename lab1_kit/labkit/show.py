"""Text pictures of digits, with a description line under every picture (for screen readers too).

    show_digit(image)                       # a 28 x 28 picture made of characters, then:
    Image: label 7 · 143 ink pixels (18%) · ink box rows 6–22, cols 8–19
    side_by_side([a, b], ["test #8", "nearest training image"])
    show_mistakes(predict_nearest, n=3)     # misread digits beside their nearest training image

The same image, three ways (Lab 1, "Getting to Know Our Data"):
    show_numbers(image, rows=range(4, 14), cols=range(8, 24))   # part of the grid of numbers, 0 to 255
    show_binary(image, threshold=128)                           # '#' if pixel >= threshold, else '.'
    show_greyscale(image)                                       # one character per shade, ' ' to '@'

Views that use YOUR functions, and say what to do if they aren't ready yet:
    show_top3(image, truth, train_images, train_labels, distance_fn=squared_distance, predict_fn=predict_nearest)
    show_rendered(render_digit, image, label=7)

A pixel counts as ink when its value is 128 or more (the same rule as ink_count in Lab 1).
"""
from __future__ import annotations

import itertools
import math

from .text import is_number, say, type_name
from .timing import plain_distance

RAMP = " .:-=+*#%@"           # paper (0) ... ink (255): 10 shades, darker = more ink
INK = 128                     # pixels >= 128 count as ink in describe()
MAX_LINE = 120                # wider pictures switch to one character per pixel
MAX_PIXELS = 100_000          # anything longer is not an image


# ----------------------------------------------------------------------------- reading images
def _pixels(image: object) -> tuple[list[int] | None, str]:
    """Turn anything image-like into ints 0-255. Returns (pixels, "") or (None, a friendly problem)."""
    if image is None:
        return None, "this is None, not an image. An image is a list of numbers (784 for 28 x 28)."
    if isinstance(image, (str, bytes)):
        return None, ("this is text, not a list of numbers. For pasted drawing values use "
                      "parse_drawing(text) or prepare_drawing(text) first.")
    if isinstance(image, dict):
        return None, "this is a dict, not an image. An image is a list of numbers (784 for 28 x 28)."
    try:
        values = list(itertools.islice(image, MAX_PIXELS + 1))     # islice: never loop forever
    except TypeError:
        return None, f"{type_name(image)} is not an image. An image is a list of numbers (784 for 28 x 28)."
    if len(values) > MAX_PIXELS:
        return None, f"this has more than {MAX_PIXELS:,} values, far too many for one image."
    if not values:
        return None, "this image is an empty list."
    if isinstance(values[0], (list, tuple)):          # a grid (list of rows): flatten it
        values = [v for row in values for v in (row if isinstance(row, (list, tuple)) else [row])]
    numbers = []
    for value in values:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None, f"the pixel {value!r} is not a number."
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                value = float(value)                  # e.g. numpy numbers
            except (TypeError, ValueError):
                return None, f"the pixel {value!r} is not a number."
        if value != value:                            # nan -> paper
            value = 0.0
        numbers.append(min(1e9, max(-1e9, value)))    # inf -> a big number, clamped below
    top = max(numbers)
    if 0 < top <= 1.0 and any(v != int(v) for v in numbers):   # normalised 0.0-1.0 pixels
        numbers = [v * 255 for v in numbers]
    return [int(min(255, max(0, round(v)))) for v in numbers], ""


def _width(n_pixels: int, width: int | None) -> tuple[int | None, str]:
    if width is not None:
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            return None, f"width must be a whole number such as 28, not {width!r}."
        if n_pixels % width:
            return None, f"{n_pixels} pixels don't make full rows of {width}."
        return width, ""
    side = math.isqrt(n_pixels)
    if side * side == n_pixels:
        return side, ""
    return None, f"I can't tell the width of an image with {n_pixels} values: pass width=..."


# ----------------------------------------------------------------------------- describe
def describe(image: object, *, label: int | None = None, name: str = "Image", width: int | None = None) -> str:
    """One sentence about an image: label, ink pixels and where the ink is.

    >>> describe([0, 255, 255, 0], label=1)
    'Image (2×2): label 1 · 2 ink pixels (50%) · ink box rows 0–1, cols 0–1'
    """
    pixels, problem = _pixels(image)
    if pixels is None:
        return f"{name}: not an image ({problem})"
    width, problem = _width(len(pixels), width)
    if width is None:
        return f"{name}: {len(pixels)} values ({problem})"
    height = len(pixels) // width
    size = "" if (width, height) == (28, 28) else f" ({width}×{height})"
    parts = [] if label is None else [f"label {label}"]
    ink = [i for i, value in enumerate(pixels) if value >= INK]
    if not ink:
        parts.append("no ink pixels (blank)")
    else:
        rows = [i // width for i in ink]
        cols = [i % width for i in ink]
        share = round(100 * len(ink) / len(pixels))
        parts.append(f"{len(ink)} ink pixel{'s' if len(ink) != 1 else ''} ({share}%)")
        parts.append(f"ink box rows {min(rows)}–{max(rows)}, cols {min(cols)}–{max(cols)}")
    return f"{name}{size}: " + " · ".join(parts)


# ----------------------------------------------------------------------------- pictures
def picture(pixels: list[int], width: int, *, double: bool = True) -> list[str]:
    """The framed character picture of an image, as a list of lines."""
    cell = 2 if double else 1
    lines = ["+" + "-" * (width * cell) + "+"]
    for start in range(0, len(pixels), width):
        row = pixels[start:start + width]
        lines.append("|" + "".join(RAMP[value * len(RAMP) // 256] * cell for value in row) + "|")
    lines.append(lines[0])
    return lines


def show_digit(image: object, *, width: int | None = None, label: int | None = None) -> None:
    """Print an image as a character picture, followed by its describe() line.

    Works for 28 x 28 (784 values) and 14 x 14 (196 values) images; use width=... for other shapes.
    """
    pixels, problem = _pixels(image)
    if pixels is None:
        say(f"show_digit: {problem}")
        return
    size, problem = _width(len(pixels), width)
    if size is None:
        say(f"show_digit: {problem}")
        return
    say(*picture(pixels, size, double=size * 2 + 2 <= MAX_LINE))
    say(describe(pixels, label=label, width=size))


def _join(blocks: list[list[str]], captions: list[str], gap: int = 3) -> list[str]:
    """Place text blocks next to each other, each with its caption on top."""
    widths = [max(len(caption), max((len(line) for line in block), default=0))
              for block, caption in zip(blocks, captions)]
    height = max(len(block) for block in blocks)
    spacer = " " * gap
    lines = [spacer.join(caption.ljust(w) for caption, w in zip(captions, widths)).rstrip()]
    for row in range(height):
        parts = [(block[row] if row < len(block) else "").ljust(w) for block, w in zip(blocks, widths)]
        lines.append(spacer.join(parts).rstrip())
    return lines


def side_by_side(images: list, captions: list[str] | None = None, *, width: int | None = None) -> None:
    """Print several images next to each other, each with a caption, then one describe() line each."""
    try:
        images = list(itertools.islice(images, 51))      # at most 50 pictures (islice: never endless)
    except TypeError:
        say("side_by_side: pass a list of images, e.g. side_by_side([image_a, image_b], [\"a\", \"b\"]).")
        return
    if not images:
        say("side_by_side: the list of images is empty.")
        return
    if is_number(images[0]):
        say("side_by_side needs a list of images, e.g. side_by_side([image_a, image_b], [\"a\", \"b\"]). "
            "For one image, use show_digit(image).")
        return
    if len(images) > 50:
        say("side_by_side: that's more than 50 images; pass a shorter list, e.g. images[:5].")
        return
    captions = [str(c) for c in (captions or [])]
    captions += [f"Image {i + 1}" for i in range(len(captions), len(images))]
    converted = []
    for image, caption in zip(images, captions):
        pixels, problem = _pixels(image)
        size, problem = _width(len(pixels), width) if pixels else (None, problem)
        if size is None:
            say(f"side_by_side: {caption}: {problem}")
            return
        converted.append((pixels, size))
    double = sum(size * 2 + 2 for _, size in converted) + 3 * (len(converted) - 1) <= MAX_LINE
    blocks = [picture(pixels, size, double=double) for pixels, size in converted]
    say(*_join(blocks, captions))
    for (pixels, size), caption in zip(converted, captions):
        say(describe(pixels, name=caption, width=size))


def text_bar(value: float, width: int = 30) -> str:
    """A text bar for a fraction between 0 and 1: text_bar(0.8, 10) -> '########.. 80.0%'.

    Numbers between 1 and 100 are read as percentages (text_bar(80) == text_bar(0.8)).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return "." * width + " n/a"
    fraction = value / 100 if 1 < value <= 100 else value
    fraction = min(1.0, max(0.0, fraction))
    filled = round(fraction * width)
    return "#" * filled + "." * (width - filled) + f" {fraction:.1%}"


# ----------------------------------------------------------------------------- mistakes
def _render_with(renderer, pixels: list[int], size: int) -> list[str] | None:
    """Lines drawn by the student's renderer (e.g. render_digit), or None if it isn't ready."""
    try:
        text = renderer(pixels) if size == 28 else renderer(pixels, size)
    except Exception:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return text.split("\n")


def show_mistakes(predict_fn, *, n: int = 3, renderer=None) -> None:
    """Show up to `n` test digits that predict_fn read wrongly, each beside its nearest training image.

    Re-uses the predictions of the last evaluate() run of the same function (no waiting); otherwise
    runs evaluate() first. renderer=render_digit draws the pictures with your own renderer.
    """
    from .evaluate import last_run, evaluate

    run = last_run(predict_fn)
    if run is None:
        say("No evaluate() run of this function yet, so I run it first.")
        evaluate(predict_fn)
        run = last_run(predict_fn)
        if run is None:
            return
    mistakes = run["mistakes"]
    if not mistakes:
        say(f"No mistakes: all {run['total']} test images were read correctly.")
        return
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        n = 3
    from .data import cached_digits, digit_files

    train_name, test_name = digit_files(run["size"])
    train_images, train_labels = cached_digits(train_name, run["n_train"])
    test_images, _ = cached_digits(test_name, run["n_test"])
    note_shown = False
    shown = mistakes[:n]
    say(f"{len(mistakes)} of {run['total']} test images were misread. Showing {len(shown)} of them:")
    for number, index in enumerate(shown, start=1):
        image = test_images[index]
        distances = [plain_distance(image, train) for train in train_images]
        nearest = distances.index(min(distances))
        truth, said = run["labels"][index], run["predictions"][index]
        captions = [f"Test image #{index} (truth {truth})",
                    f"Nearest training image #{nearest} (label {train_labels[nearest]})"]
        say("", f"Mistake {number} of {len(shown)}")
        blocks = None
        if renderer is not None:
            size = 28 if len(image) == 784 else 14
            blocks = [_render_with(renderer, image, size), _render_with(renderer, train_images[nearest], size)]
            if None in blocks:
                blocks = None
                if not note_shown:
                    say("(Your renderer isn't ready yet, so these use the built-in pictures.)")
                    note_shown = True
        if blocks is None:
            side = 28 if len(image) == 784 else 14
            blocks = [picture(image, side), picture(train_images[nearest], side)]
        say(*_join(blocks, captions))
        say(describe(image, label=truth, name=f"Test image #{index}"),
            describe(train_images[nearest], label=train_labels[nearest], name=f"Nearest training image #{nearest}"),
            f"Your machine said {said}; the truth is {truth}. Would you have read it correctly?")


# ----------------------------------------------------------------------------- the same image, three ways
def _positions(value: object, size: int, default: list[int], what: str) -> tuple[list[int] | None, str]:
    """Rows or columns to show: None (the default window), a range, or a list of whole numbers."""
    if value is None:
        return default, ""
    if isinstance(value, (bool, str, float)) or is_number(value):
        return None, f"{what} must be a range such as range(4, 14), not {value!r}."
    try:
        chosen = list(itertools.islice(value, size + 1))
    except TypeError:
        return None, f"{what} must be a range such as range(4, 14), not {type_name(value)}."
    if not chosen:
        return None, f"{what} is empty: try range(4, 14)."
    for item in chosen:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < size:
            return None, f"{what} must be whole numbers from 0 to {size - 1} (got {item!r})."
    return chosen, ""


def _ink_window(pixels: list[int], width: int, rows: int = 10, cols: int = 14) -> tuple[list[int], list[int]]:
    """A window of at most rows x cols: from the top of the ink down, centred on the ink left to right
    (the whole image if it is small)."""
    height = len(pixels) // width
    ink = [i for i, value in enumerate(pixels) if value > 0]
    if ink:
        first_r = min(i // width for i in ink)
        mid_c = (min(i % width for i in ink) + max(i % width for i in ink)) // 2
    else:
        first_r, mid_c = (height - rows) // 2, width // 2
    top = max(0, min(height - rows, first_r)) if height > rows else 0
    left = max(0, min(width - cols, mid_c - cols // 2)) if width > cols else 0
    return list(range(top, min(height, top + rows))), list(range(left, min(width, left + cols)))


def _span(values: list[int]) -> str:
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}–{values[-1]}" if len(values) > 1 else str(values[0])
    return ", ".join(map(str, values))


def show_numbers(image: object, rows: object = None, cols: object = None, *, width: int | None = None) -> None:
    """Print part of an image as the numbers it really is (0 = paper, 255 = ink).

    rows and cols are ranges, e.g. show_numbers(image, rows=range(4, 14), cols=range(8, 24)); without them
    it shows 10 rows from the top of the ink, and 14 columns around its middle.
    """
    pixels, problem = _pixels(image)
    if pixels is None:
        say(f"show_numbers: {problem}")
        return
    size, problem = _width(len(pixels), width)
    if size is None:
        say(f"show_numbers: {problem}")
        return
    height = len(pixels) // size
    default_rows, default_cols = _ink_window(pixels, size)
    row_list, problem = _positions(rows, height, default_rows, "rows")
    if row_list is None:
        say(f"show_numbers: {problem}")
        return
    col_list, problem = _positions(cols, size, default_cols, "cols")
    if col_list is None:
        say(f"show_numbers: {problem}")
        return
    if len(col_list) > 28:
        col_list = col_list[:28]
    lines = [f"Pixel values of rows {_span(row_list)} and columns {_span(col_list)} "
             f"(0 = paper, 255 = ink; pixel (r, c) is image[r * {size} + c])"]
    lines.append(" " * 5 + "".join(f"{'c' + str(c):>4}" for c in col_list))
    for r in row_list:
        lines.append(f"{'r' + str(r):<5}" + "".join(f"{pixels[r * size + c]:>4}" for c in col_list))
    say(*lines)


def _threshold_problem(threshold: object) -> str | None:
    if isinstance(threshold, bool) or not is_number(threshold) or threshold != threshold:
        return f"threshold must be a number such as 128, not {threshold!r}."
    return None


def show_binary(image: object, threshold: float = INK, *, width: int | None = None) -> None:
    """Print an image in two characters only: '#' if a pixel is >= threshold (ink), '.' otherwise (paper).

    The same rule as ink_count and render_digit. Change the threshold to see how the picture changes:
    show_binary(image, threshold=50) keeps the faint grey edges, threshold=200 only the darkest ink.
    """
    problem = _threshold_problem(threshold)
    pixels, why = _pixels(image)
    if problem or pixels is None:
        say(f"show_binary: {problem or why}")
        return
    size, why = _width(len(pixels), width)
    if size is None:
        say(f"show_binary: {why}")
        return
    cell = 2 if size * 2 + 2 <= MAX_LINE else 1
    lines = ["+" + "-" * (size * cell) + "+"]
    for start in range(0, len(pixels), size):
        row = pixels[start:start + size]
        lines.append("|" + "".join(("#" if value >= threshold else ".") * cell for value in row) + "|")
    lines.append(lines[0])
    ink = sum(1 for value in pixels if value >= threshold)
    shown = int(threshold) if float(threshold).is_integer() else threshold
    say(*lines, f"Threshold {shown}: {ink} of {len(pixels)} pixels are ink (#), {len(pixels) - ink} are paper (.).")


def ramp_legend() -> str:
    """Which character stands for which pixel values in show_greyscale and show_digit."""
    parts = []
    for k, char in enumerate(RAMP):
        low = -(-k * 256 // len(RAMP))                       # the smallest value v with v * 10 // 256 == k
        high = -(-(k + 1) * 256 // len(RAMP)) - 1
        parts.append(f"'{char}' {low}–{min(high, 255)}")
    return "Shades from paper to ink: " + ", ".join(parts)


def show_greyscale(image: object, *, width: int | None = None) -> None:
    """Print an image with one character per shade of grey, from ' ' (paper, 0) to '@' (ink, 255).

    Unlike show_binary, nothing is thrown away: faint pixels get faint characters.
    """
    pixels, problem = _pixels(image)
    if pixels is None:
        say(f"show_greyscale: {problem}")
        return
    size, problem = _width(len(pixels), width)
    if size is None:
        say(f"show_greyscale: {problem}")
        return
    say(*picture(pixels, size, double=size * 2 + 2 <= MAX_LINE), ramp_legend())


# ----------------------------------------------------------------------------- views of the student's functions
def _task_for(fn: object, fallback: str):
    """(task, its name) for a student function, found by its name (or the fallback name), or (None, name)."""
    from . import cases
    from .env import current_lab

    lab = current_lab()
    lab_cases = cases.load(lab) if lab else None
    name = getattr(fn, "__name__", None) or fallback
    if lab_cases is None:
        return None, name, None
    task = lab_cases.by_name(name) or lab_cases.by_name(fallback)
    return task, name, lab_cases


def _rescue_hint(task, name: str) -> str:
    if task is not None and task.rescue:
        return f"run {task.name} = rescue(\"{task.id}\") in a new cell, then run this cell again"
    return "finish it, then run this cell again"


def _not_ready(fn: object, fallback: str) -> str | None:
    """A friendly 'This cell needs C1 ...' line if fn is missing or still a stub; None if it is written."""
    from .check import is_stub

    task, name, _ = _task_for(fn, fallback)
    if callable(fn) and not is_stub(fn):
        return None
    label = f"{task.name} ({task.id})" if task is not None else name
    hint = _rescue_hint(task, name)
    return f"This cell needs {label}. Not there yet? {hint[0].upper()}{hint[1:]}."


def _passing(fn: object, fallback: str) -> tuple[bool, str]:
    """(True, "") if fn passes its task's check, else (False, a short note naming the check)."""
    from .check import PASS, quiet_status

    task, name, lab_cases = _task_for(fn, fallback)
    if task is None or getattr(fn, "__name__", None) != task.name:
        return True, ""
    try:
        status = quiet_status(task, lab_cases)
    except Exception:
        return True, ""
    if status == PASS:
        return True, ""
    return False, (f"Note: {task.name} doesn't pass check(\"{task.id}\", {task.name}) yet, so what you see "
                   f"below may be wrong. Run that check to see why, or {_rescue_hint(task, name)}.")


_ORDINALS = ["1st", "2nd", "3rd"] + [f"{n}th" for n in range(4, 21)]


def show_top3(image: object, truth: object, train_images: list, train_labels: list, *, distance_fn=None,
              predict_fn=None, k: int = 3) -> None:
    """Show a test image, its true label, and the k training images closest to it according to YOUR
    distance function, each with its label and distance, then what YOUR predict function says.

        show_top3(test_images[0], test_labels[0], train_images, train_labels,
                  distance_fn=squared_distance, predict_fn=predict_nearest)

    If a function isn't written yet (or raises an error), it says what to do instead of crashing.
    """
    from .check import run_call, your_line

    pixels, problem = _pixels(image)
    if pixels is None:
        say(f"show_top3: the test image: {problem}")
        return
    size, problem = _width(len(pixels), None)
    if size is None:
        say(f"show_top3: {problem}")
        return
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        k = 3
    try:
        train_images, train_labels = list(train_images), list(train_labels)
    except TypeError:
        say("show_top3: pass the training images and their labels as two lists.")
        return
    if not train_images or len(train_images) != len(train_labels):
        say("show_top3: train_images and train_labels must be two non-empty lists of the same length.")
        return
    message = _not_ready(distance_fn, "squared_distance")
    if message:
        say(message)
        return
    task, name, _ = _task_for(distance_fn, "squared_distance")
    attempt = run_call(lambda: [distance_fn(list(image), list(train)) for train in train_images], (), track=False)
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return
    distances = attempt.got
    if attempt.error is not None or not isinstance(distances, list) or not all(is_number(d) for d in distances):
        if attempt.error is not None:
            lines = [f"{name} raised {attempt.error_text} while comparing the test image with the training images."]
            place = your_line(attempt.error, distance_fn, always_name=True)
            if place:
                lines.append(f"{place[0]} {place[1]}")
        else:
            bad = next((d for d in distances if not is_number(d)), None) if isinstance(distances, list) else None
            lines = [f"{name} returned {bad!r} ({type_name(bad)}); a distance must be a number."]
        check_text = f"check(\"{task.id}\", {task.name}) shows why, or " if task is not None else ""
        lines.append(f"Next step: {check_text}{_rescue_hint(task, name)}.")
        say(*lines)
        return
    ok, note = _passing(distance_fn, "squared_distance")
    if not ok:
        say(note)
    order = sorted(range(len(distances)), key=lambda i: (distances[i], i))[:k]
    truth_text = f"true label {truth}" if truth is not None else "true label unknown"
    images = [pixels] + [train_images[i] for i in order]
    captions = [f"Test image ({truth_text})"] + [
        f"{_ORDINALS[n]} closest: label {train_labels[i]} (training image #{i})" for n, i in enumerate(order)]
    converted = []
    for picture_pixels, caption in zip(images, captions):
        values, _ = _pixels(picture_pixels)
        converted.append((values or [0] * len(pixels), caption))
    say(f"The test image and the {k} training images closest to it, by your {name}:")
    for start in range(0, len(converted), 2):
        pair = converted[start:start + 2]
        say(*_join([picture(values, size) for values, _ in pair], [caption for _, caption in pair]))
    width = max(len(str(len(train_images))), 3)
    say(f"Distances from the test image ({truth_text}), computed by your {name}:")
    for n, i in enumerate(order):
        say(f"  {_ORDINALS[n]} closest   training image #{str(i).ljust(width)}   label {train_labels[i]}   "
            f"squared distance {distances[i]:,}")
    if predict_fn is None:
        return
    message = _not_ready(predict_fn, "predict_nearest")
    if message:
        say(message)
        return
    from .evaluate import try_predict

    guess = try_predict(predict_fn, list(image), train_images, train_labels)
    if guess is None:
        return
    p_name = getattr(predict_fn, "__name__", "your function")
    line = f"Your {p_name} says {guess}."
    if truth is not None:
        line += f" The true label is {truth}: " + ("correct." if guess == truth else "not this time.")
    say(line)
    if order and guess != train_labels[order[0]]:
        say(f"(That isn't the label of the 1st closest image, {train_labels[order[0]]}: check C1 and C2.)")


def show_rendered(renderer: object, image: object, *, label: object = None) -> None:
    """Print an image drawn by YOUR renderer (e.g. render_digit), or say what to do if it isn't ready.

        show_rendered(render_digit, test_images[0], label=test_labels[0])
    """
    from .check import run_call, your_line

    message = _not_ready(renderer, "render_digit")
    if message:
        say(message)
        return
    task, name, _ = _task_for(renderer, "render_digit")
    pixels, problem = _pixels(image)
    if pixels is None:
        say(f"show_rendered: {problem}")
        return
    attempt = run_call(renderer, (list(image),), track=False)
    check_text = f"check(\"{task.id}\", {task.name}) shows why, or " if task is not None else ""
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return
    if attempt.error is not None:
        lines = [f"{name} raised {attempt.error_text} on this image."]
        place = your_line(attempt.error, renderer, always_name=True)
        if place:
            lines.append(f"{place[0]} {place[1]}")
        say(*lines, f"Next step: {check_text}{_rescue_hint(task, name)}.")
        return
    text = attempt.got
    if not isinstance(text, str) or not text.strip():
        what = "printed the picture but returned None" if text is None and attempt.printed.strip() else (
            f"returned {type_name(text)}")
        say(f"{name} {what}; it should return the picture as one string.",
            f"Next step: {check_text}{_rescue_hint(task, name)}.")
        return
    ok, note = _passing(renderer, "render_digit")
    if not ok:
        say(note)
    rows = text.split("\n")
    say(f"Drawn by your {name}" + (f" (true label {label})" if label is not None else "") + ":", *rows,
        f"{len(rows)} lines of {max(len(r) for r in rows)} characters, '#' for ink and '.' for paper.")
