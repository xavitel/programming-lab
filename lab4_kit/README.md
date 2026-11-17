# digits: Garabato's digit reader, as a command-line tool

*Programming Lab · Lab 4 · Ship It to the Depot. Garabato Parcels is a fictional company; the digits come from
MNIST (LeCun, Cortes, Burges), CC BY-SA 3.0, adapted.*

`digits.py` reads handwritten digits from image files. The night shift at the depot runs it from a terminal:
no Jupyter needed. You build it in Lab 4, in `digits.py`, next to the notebook `lab4_ship_it.ipynb`.

## What is in this folder

| Path | What it is |
|---|---|
| `digits.py` | the tool you complete in Lab 4 (open it in VS Code next to the notebook) |
| `lab4_ship_it.ipynb` | the lab notebook: the tasks, the checks, and the notebook route `run_cli(digits.main, "...")` |
| `web_app.py` | the web API of the drawing page, with FastAPI (ADVANCED BLD2; `uv sync --group web` first) |
| `samples/` | 25 parcel images: `parcel_001.csv` ... `parcel_020.csv` (one line of 784 numbers) and `parcel_021.txt` ... `parcel_025.txt` (28 lines of 28 numbers) |
| `inbox/` | 30 images that arrived overnight, for the batch command (OPTIONAL BLD1) |
| `strips/` | 10 postcode strips of 5 digits, and `strips_truth.csv` (the Challenge) |
| `tests/test_example.py` | an example test file for `digits.py` (OPTIONAL ENG1) |
| `data/` | the digit files (training and test images) |
| `labkit/` | the course helpers: `check()`, `rescue()`, `run_cli()`, the test runner, ... |
| `tools/draw_digit.html` | draw a digit with the mouse (served by `web_app.py`, it also asks the API for a prediction) |

## Usage

TODO (ADVANCED ENG2): write how to run `digits.py`, for someone who has never seen this course: what to
install, which folder to be in, and one example command for each subcommand, with what it prints.

## Running the tests

```text
python -m labkit.minitest tests        # standard library only: runs tests/test_*.py
uv run --with pytest pytest -q         # pytest (needs internet the first time)
python -m labkit.mutate tests/test_digits.py   # how many buggy versions of digits.py do your tests catch?
```

On Windows, type `py` instead of `python` if `python` is not found; on macOS, `python3`.

## AI-use declaration

This course asks you to use AI critically and to say how you used it. Nobody collects or grades this file.
Please don't write your name.

- **Tools I used** (or "none"):
- **For which tasks** (e.g. C2, ENG1):
- **One prompt I used** (copy it, or summarise it in one line):
- **How I checked the answer** (check(), my own test, reading the code, a classmate…):
- **Something the AI got wrong, or that I changed:**
- **One line of AI-written code I can now explain in my own words:**
