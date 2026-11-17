"""Loading the kit's data files, and the canonical 28 x 28 -> 14 x 14 downsample.

The data contract (the same in every lab): two parallel lists, `images` and `labels`.
An image is a list of ints from 0 (paper) to 255 (ink), row by row from the top-left corner:
pixel (r, c) of a 28 x 28 image is image[r * 28 + c]. A 14 x 14 image has 196 values.

    train_images, train_labels = load_digits("data/mnist_train_10k.csv.gz", limit=1000)
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import json
import os
import sys
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent.parent    # the kit folder: it contains labkit/ and data/
DATA_DIR = KIT_DIR / "data"
IMAGE_SIZES = {784: 28, 196: 14}                   # number of pixels -> side length

_cache: dict[Path, tuple[list[list[int]], list[int], bool]] = {}
_watching: list[list[str]] = []          # one list per active watch_reads() block
_audit_hook_on = False


# ----------------------------------------------------------------------------- which files were read
def _note_read(path: object) -> None:
    for seen in _watching:
        seen.append(str(path))


def _audit(event: str, args: tuple) -> None:
    """Python's audit hook: records the files opened while a watch_reads() block runs. Never raises."""
    if event == "open" and _watching:
        try:
            target = args[0] if args else None
            if isinstance(target, (str, bytes, os.PathLike)):
                _note_read(os.fsdecode(target))
        except Exception:
            pass


@contextlib.contextmanager
def watch_reads():
    """Record every file opened inside the block (labkit's loaders, even from the cache, and plain open()).

    The League uses it to notice an entry that reads the test file. Nothing is recorded outside the block,
    and nothing leaves your laptop.

        with watch_reads() as seen:
            ...
        print(seen)          # ['C:/.../data/mnist_train_10k.csv.gz', ...]
    """
    global _audit_hook_on
    if not _audit_hook_on:
        try:
            sys.addaudithook(_audit)          # Python 3.8+; it stays installed, but is idle outside a block
        except Exception:
            pass
        _audit_hook_on = True
    seen: list[str] = []
    _watching.append(seen)
    try:
        yield seen
    finally:
        _watching.remove(seen)


# ----------------------------------------------------------------------------- finding files
def resolve(path: str | Path) -> Path:
    """Find a data file: as given, then inside the kit folder, then inside data/.

    Raises FileNotFoundError with a message that names the file and the folders searched.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"a file path must be text such as \"data/mnist_test_2k.csv.gz\", not {type(path).__name__}")
    given = Path(path).expanduser()
    candidates = [given]
    if not given.is_absolute():
        candidates += [KIT_DIR / given, DATA_DIR / given, DATA_DIR / given.name]
    for candidate in candidates:
        if candidate.is_file():
            if _watching:
                _note_read(candidate)
            return candidate
    raise FileNotFoundError(f"{path} not found (looked in {Path.cwd()} and in the kit folder {KIT_DIR})")


def open_text(path: str | Path):
    """Open a text file for reading as UTF-8, unzipping it on the fly if it is gzipped."""
    file = resolve(path)
    with open(file, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"          # every gzip file starts with these two bytes
    if gzipped:
        return gzip.open(file, "rt", encoding="utf-8", newline="")
    return open(file, encoding="utf-8-sig", newline="")


def _check_limit(limit: object) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(f"limit must be a whole number such as 1000, or None for every row (got {limit!r})")
    if limit < 0:
        raise ValueError(f"limit must be 0 or more (got {limit})")
    return limit


# ----------------------------------------------------------------------------- digits
def load_digits(path: str | Path, limit: int | None = None) -> tuple[list[list[int]], list[int]]:
    """Read a digit file into two parallel lists: (images, labels).

    Works for every digit file in data/: 28 x 28 (784 pixels) and 14 x 14 (196 pixels),
    gzipped (.csv.gz) or plain (.csv). `limit` keeps only the first rows; every block of 100
    rows has 10 images of each digit, so limit=1000 gives 100 of each.

    >>> images, labels = load_digits("data/mnist_test_2k.csv.gz", limit=100)
    >>> len(images), len(images[0]), len(labels)
    (100, 784, 100)
    """
    limit = _check_limit(limit)
    file = resolve(path)
    images: list[list[int]] = []
    labels: list[int] = []
    n_pixels = None
    with open_text(file) as f:
        for line_number, row in enumerate(csv.reader(f), start=1):
            if limit is not None and len(labels) >= limit:
                break
            if not row:
                continue                                      # an empty line
            if line_number == 1 and row[0].strip().lower() == "label":
                continue                                      # the header: label,p0,p1,...
            if n_pixels is None:
                n_pixels = len(row) - 1
                if n_pixels not in IMAGE_SIZES:
                    raise ValueError(f"{file.name}, line {line_number}: {n_pixels} pixels per image; "
                                     "a digit file has 784 (28 x 28) or 196 (14 x 14)")
            if len(row) - 1 != n_pixels:
                raise ValueError(f"{file.name}, line {line_number}: {len(row) - 1} pixels, expected {n_pixels}")
            try:
                labels.append(int(row[0]))
                images.append(list(map(int, row[1:])))       # every pixel: text -> int
            except ValueError:
                raise ValueError(f"{file.name}, line {line_number}: a value is not a whole number") from None
    return images, labels


def cached_digits(name: str, limit: int | None = None) -> tuple[list[list[int]], list[int]]:
    """Like load_digits, but keeps the result in memory for labkit's own tools.

    The lists are shared: callers that hand them to student code must copy them first.
    """
    limit = _check_limit(limit)
    file = resolve(name)
    if file in _cache:
        images, labels, complete = _cache[file]
        if complete or (limit is not None and limit <= len(labels)):
            return images[:limit], labels[:limit]
    images, labels = load_digits(file, limit)
    _cache[file] = (images, labels, limit is None or len(labels) < limit)
    return images[:limit], labels[:limit]


def digit_files(size: int = 28) -> tuple[str, str]:
    """The (train, test) file names for 28 x 28 or 14 x 14 images."""
    if size == 28:
        return "mnist_train_10k.csv.gz", "mnist_test_2k.csv.gz"
    if size == 14:
        return "mnist_train_10k_14x14.csv.gz", "mnist_test_2k_14x14.csv.gz"
    raise ValueError(f"size must be 28 or 14 (got {size!r})")


def downsample(image: list[int]) -> list[int]:
    """Shrink a 28 x 28 image to 14 x 14: each new pixel is the floor average of a 2 x 2 block.

    This is the canonical rule used by every 14 x 14 file in the kit: (a + b + c + d) // 4.

    >>> downsample([4] * 784)[:3]
    [4, 4, 4]
    """
    if len(image) != 784:
        raise ValueError(f"downsample needs a 28 x 28 image (784 values), got {len(image)} values")
    small = []
    for r in range(0, 28, 2):            # top row of each 2 x 2 block
        for c in range(0, 28, 2):        # left column of each 2 x 2 block
            i = r * 28 + c
            small.append((image[i] + image[i + 1] + image[i + 28] + image[i + 29]) // 4)
    return small


# ----------------------------------------------------------------------------- single images
def _pixels_from_text(values: list[str], where: str) -> list[int]:
    pixels = []
    for value in values:
        try:
            pixel = int(value)
        except ValueError:
            raise ValueError(f"{where}: {value!r} is not a whole number") from None
        if not 0 <= pixel <= 255:
            raise ValueError(f"{where}: {pixel} is outside 0-255")
        pixels.append(pixel)
    return pixels


def read_image_csv(path: str | Path) -> list[int]:
    """Read one image saved as a single line of comma-separated ints (784 or 196 values)."""
    file = resolve(path)
    with open_text(file) as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{file.name}: expected one line of pixels, found {len(lines)} lines")
    values = [v.strip() for v in lines[0].split(",") if v.strip()]
    if len(values) not in IMAGE_SIZES:
        raise ValueError(f"{file.name}: expected 784 numbers, found {len(values)}")
    return _pixels_from_text(values, file.name)


def read_image_grid(path: str | Path) -> list[int]:
    """Read one image saved as lines of space-separated ints (e.g. 28 lines of 28 numbers).

    Returns the pixels row by row as one flat list. Every line must have the same length,
    so a 28 x 140 postcode strip (5 digits side by side) also works.
    """
    file = resolve(path)
    with open_text(file) as f:
        rows = [line.split() for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{file.name}: the file is empty")
    width = len(rows[0])
    pixels: list[int] = []
    for number, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(f"{file.name}, line {number}: {len(row)} numbers, expected {width} like line 1")
        pixels += _pixels_from_text(row, f"{file.name}, line {number}")
    return pixels


# ----------------------------------------------------------------------------- postcodes
def load_postcodes(path: str | Path = "data/postcodes_test.csv") -> list[dict]:
    """Read the postcode strips: one dict per strip.

    Each dict is {"strip_id": "S001", "postcode": "28046", "indices": [750, 1923, 1544, 961, 868]}:
    the indices are positions in the test lists returned by load_digits, and their labels
    spell the postcode.
    """
    strips = []
    with open_text(path) as f:
        for row in csv.DictReader(f):
            strips.append({
                "strip_id": row["strip_id"],
                "postcode": row["postcode"],
                "indices": [int(row[f"i{k}"]) for k in range(5)],
            })
    return strips


def load_valid_postcodes(path: str | Path = "data/valid_postcodes.txt") -> set[str]:
    """Read the simplified list of valid postcodes (28001 ... 28055) as a set of strings."""
    with open_text(path) as f:
        return {line.strip() for line in f if line.strip()}


# ----------------------------------------------------------------------------- integrity
def sha256_of(path: str | Path) -> str:
    """The SHA-256 checksum of a file, as 64 hex characters."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(folder: Path | None = None) -> dict:
    """data/MANIFEST.json as a dict {file name: {"sha256", "bytes", "rows", "columns"}}."""
    with open((folder or DATA_DIR) / "MANIFEST.json", encoding="utf-8") as f:
        return json.load(f)
