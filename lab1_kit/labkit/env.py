"""The environment report printed by labkit.start(N) in the setup cell.

It checks, in this order: versions and paths, the data files (sizes and SHA-256 checksums against
data/MANIFEST.json), whether the kit folder is writable, whether numpy is installed (optional),
Anaconda base, and how fast this laptop computes one distance. If a data file is missing or
damaged it prints ONE concrete fix and stops the setup cell.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from . import cases
from .data import DATA_DIR, KIT_DIR, read_manifest, sha256_of
from .text import say
from .timing import distance_time, format_duration, format_eta

# Data files each lab uses (relative to data/). A cases/labN.py may extend this with DATA_FILES.
LAB_DATA = {
    1: ("mnist_train_10k.csv.gz", "mnist_test_2k.csv.gz"),
    2: ("mnist_train_10k.csv.gz", "mnist_test_2k.csv.gz", "mnist_train_10k_14x14.csv.gz",
        "mnist_test_2k_14x14.csv.gz", "postcodes_test.csv", "valid_postcodes.txt"),
    3: ("mnist_train_10k.csv.gz", "mnist_test_2k.csv.gz", "mnist_train_10k_14x14.csv.gz",
        "mnist_test_2k_14x14.csv.gz", "valid_postcodes.txt"),
    4: ("mnist_train_10k.csv.gz", "mnist_test_2k.csv.gz", "mnist_train_10k_14x14.csv.gz",
        "mnist_test_2k_14x14.csv.gz"),
    5: ("mnist_train_10k_14x14.csv.gz", "mnist_test_2k_14x14.csv.gz", "mnist_train_10k.csv.gz",
        "mnist_test_2k.csv.gz", "postcodes_test.csv", "valid_postcodes.txt", "depots.json",
        "holdout/league_holdout.csv.gz"),
}
# The final run of each lab (Lab 1: Putting It All Together), in distances: (how many, pixels per image).
MOMENT_WORK = {1: (100_000, 784)}

REEXTRACT = ("Fix: delete the kit folder, extract the kit zip again (right-click > Extract All), "
             "then open the new folder in VS Code (File > Open Folder) and run this cell again.")

_state = {"lab": None}


def current_lab() -> int | None:
    """The lab number given to start(); if start() was not run, guess it when only one lab is in the kit."""
    if _state["lab"] is not None:
        return _state["lab"]
    labs = cases.available()
    return labs[0] if len(labs) == 1 else None


def set_lab(lab: int) -> None:
    _state["lab"] = lab


# ----------------------------------------------------------------------------- the pieces
def os_name() -> str:
    """'Windows 11', 'macOS 15.1', 'Google Colab (Linux)', 'Linux'."""
    import platform

    if "google.colab" in sys.modules:
        return "Google Colab (Linux)"
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        return "Windows 11" if build >= 22000 else f"Windows {platform.release()}"
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return f"macOS {version}" if version else "macOS"
    return platform.system() or sys.platform


def lab_files(lab: int, lab_cases=None) -> list[str]:
    """The data files `lab` needs, in report order."""
    names = list(LAB_DATA.get(lab, ()))
    extra = getattr(getattr(lab_cases, "module", None), "DATA_FILES", ())
    for name in extra:
        if name not in names:
            names.append(name)
    return names


def check_data(lab: int, lab_cases=None, data_dir: Path | None = None) -> tuple[list[str], str | None, list[str]]:
    """Verify the data folder against MANIFEST.json.

    Returns (names of the lab's files that are OK, a STOP message or None, warnings).
    Missing or damaged files this lab needs are a STOP; a changed optional file is only a warning.
    """
    folder = data_dir or DATA_DIR
    if not (folder / "MANIFEST.json").is_file():
        return [], f"STOP: data/MANIFEST.json is missing, so I can't check the data files.\n{REEXTRACT}", []
    try:
        manifest = read_manifest(folder)
    except (OSError, ValueError):
        return [], f"STOP: data/MANIFEST.json is damaged.\n{REEXTRACT}", []
    needed = lab_files(lab, lab_cases)
    ok, warnings = [], []
    for name in needed:
        if not (folder / name).is_file():
            return ok, f"STOP: data/{name} is missing.\n{REEXTRACT}", warnings
    for name, facts in manifest.items():
        path = folder / name
        if not path.is_file():
            continue                                  # a file another lab uses; not in this kit
        damaged = path.stat().st_size != facts.get("bytes") or sha256_of(path) != facts.get("sha256")
        if damaged and name in needed:
            return ok, (f"STOP: data/{name} is damaged (its checksum doesn't match the kit's list).\n"
                        f"{REEXTRACT}"), warnings
        if damaged:
            warnings.append(f"Note: data/{name} has changed since the kit was built. "
                            "Fine if you edited it on purpose; re-extract the kit to get the original.")
    for name in needed:
        if name in manifest:
            ok.append(name)
        else:
            warnings.append(f"Note: data/{name} is not listed in MANIFEST.json, so it was not checked.")
    return ok, None, warnings


def folder_writable(folder: Path) -> bool:
    """Can we create (and delete) a file in `folder`?"""
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".labkit_write_test_", delete=True):
            return True
    except OSError:
        return False


def numpy_status() -> str:
    """'not installed (only some Challenges use it)' or '2.3.3 (optional; only some Challenges use it)'."""
    if importlib.util.find_spec("numpy") is None:
        return "not installed (only some Challenges use it)"
    try:
        from importlib.metadata import version
        return f"{version('numpy')} (optional; only some Challenges use it)"
    except Exception:
        return "installed (optional; only some Challenges use it)"


def is_conda_base() -> bool:
    """True if this Python is the 'base' environment of Anaconda or Miniconda."""
    prefix = Path(sys.prefix)
    if not (prefix / "conda-meta").is_dir():
        return False
    return os.environ.get("CONDA_DEFAULT_ENV") == "base" or prefix.parent.name != "envs"


def speed_line(lab: int, lab_cases=None) -> str:
    """'Speed: 1 distance takes 55 µs here, so the full test in Putting It All Together will take about 6 s.'

    The name of the run comes from the lab's cases file (FINALE); without one it says 'the full test'.

    Timed with the distance written the way the hints show it, (a[i] - b[i]) ** 2 (timing.plain_distance):
    that is what most students write, and on Python 3.10 it is much slower than difference * difference.
    """
    if lab_cases is None:
        try:
            lab_cases = cases.load(lab)
        except Exception:                 # a broken cases file: the line just has no name for the run
            lab_cases = None
    work = getattr(getattr(lab_cases, "module", None), "MOMENT_WORK", None) or MOMENT_WORK.get(lab)
    pixels = work[1] if work else 784
    seconds = distance_time(None, pixels)
    size = "" if pixels == 784 else f" ({pixels} pixels)"
    line = f"Speed: 1 distance{size} takes {format_duration(seconds)} here"
    if not work:
        return f"{line}."
    finale = getattr(getattr(lab_cases, "module", None), "FINALE", None)
    run = f"the full test in {finale}" if finale else "the full test"
    line = f"{line}, so {run} will take {format_eta(work[0] * seconds)}."
    if work[0] * seconds > 15 and sys.version_info < (3, 11):
        line += ("\nNote: that's fine, just slow. Python 3.11 and newer run this kind of loop about 3 times "
                 "faster; if you have one (e.g. the course .venv), you can pick it as the kernel.")
    return line


# ----------------------------------------------------------------------------- the report
def start(lab: int, *, kit_dir: Path | None = None) -> int:
    """Print the environment report for lab `lab` and remember the lab number.

    Raises SystemExit (which stops the setup cell) with one concrete fix when the data is
    missing or damaged.
    """
    import platform

    from . import __version__

    if isinstance(lab, bool) or not isinstance(lab, int) or lab < 1:
        raise SystemExit(f"STOP: start() needs the lab number, e.g. labkit.start(1) (got {lab!r}).")
    set_lab(lab)
    kit = kit_dir or KIT_DIR
    say(f"Lab {lab} kit {__version__} · Python {platform.python_version()} · {os_name()} · kernel {sys.executable}")
    try:
        lab_cases = cases.load(lab)
    except Exception as error:        # a broken cases file: report it, but keep the kit usable
        lab_cases = None
        say(f"Note: the checks for Lab {lab} could not be loaded ({type(error).__name__}: {error}). "
            "Tell Javier; everything else works.")

    ok, stop, warnings = check_data(lab, lab_cases, kit / "data")
    if stop:
        raise SystemExit(stop)            # Jupyter shows the message and stops the setup cell
    if ok:
        say("Data: " + " · ".join(f"{name.split('/')[-1]} OK" for name in ok) + " (checksums match)")
    for warning in warnings:
        say(warning)

    writable = "yes" if folder_writable(kit) else (
        "no (some tasks save files: extract the kit into your Documents folder and open it from there)")
    say(f"Folder writable: {writable} · numpy: {numpy_status()}")
    if is_conda_base():
        say("Note: Anaconda 'base' detected. Fine for this lab.")
    if lab_cases is None and cases.available():
        labs = ", ".join(map(str, cases.available()))
        say(f"Note: this kit has no checks for Lab {lab}; it has checks for Lab {labs}. "
            "Is LAB right at the top of this cell?")
    say(speed_line(lab, lab_cases))
    say('READY. Scroll down to "How this lab works".')
    return lab
