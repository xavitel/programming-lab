"""labkit: the helper package of the Programming Lab kits (standard library only, Python 3.10+).

You only need three things from it:
    check("C1", squared_distance)      # PASS / NOT YET / NOT STARTED, with what to fix next
    rescue("C2")                       # borrow the reference so the rest of the lab keeps working
    the hints in the notebook (Hint 1, Hint 2, ...)

Everything else (load_digits, show_digit, evaluate, ...) is used by provided cells. The code is
meant to be readable: open any module to see how it works. Start with labkit/reference/.
"""
from __future__ import annotations

__version__ = "2026.10.1"
LAB: int | None = None            # set by start(N) in the setup cell

from .check import check, check_all                                    # noqa: E402
from .data import DATA_DIR, downsample, load_digits, load_postcodes, load_valid_postcodes  # noqa: E402
from .data import read_image_csv, read_image_grid                       # noqa: E402
from .drawing import parse_drawing, prepare_drawing, sketch_to_image    # noqa: E402
from .evaluate import compare, count_code_lines, evaluate, stopwatch, try_predict    # noqa: E402
from .league import league_final, league_run, machine_factor            # noqa: E402
from .rescue import rescue                                              # noqa: E402
from .show import describe, show_digit, show_mistakes, side_by_side, text_bar  # noqa: E402
from .show import show_binary, show_greyscale, show_numbers, show_rendered, show_top3  # noqa: E402
from . import env as _env                                               # noqa: E402


def start(lab: int) -> None:
    """Print the environment report for lab `lab` (the setup cell calls this) and remember the lab."""
    global LAB
    LAB = _env.start(lab)


__all__ = [
    "__version__", "LAB", "start", "check", "check_all", "rescue", "DATA_DIR", "load_digits",
    "downsample", "load_postcodes", "load_valid_postcodes", "read_image_csv", "read_image_grid",
    "show_digit", "describe", "side_by_side", "show_mistakes", "text_bar", "show_numbers", "show_binary",
    "show_greyscale", "show_top3", "show_rendered", "evaluate", "compare",
    "stopwatch", "count_code_lines", "try_predict", "sketch_to_image", "parse_drawing", "prepare_drawing",
    "league_run", "league_final", "machine_factor",
]
