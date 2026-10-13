# The data in this kit

Everything in this folder is ready to use: you never need to download anything for Core.
Each kit contains only the files its lab needs, so some files listed below may not be in your kit.

**Garabato Parcels is a fictional company.** Its depots, scanners, operators and the postcode-to-depot
assignments are invented for teaching. No file here contains anyone's personal data.

## How a digit is stored

Every digit file uses the same layout, the data contract you meet in every lab:

- one header line, then one image per line;
- column `label` is the digit that was written (0 to 9);
- the other columns `p0`, `p1`, … are the pixels, row by row from the top-left corner:
  **0 = paper, 255 = ink**;
- a 28 × 28 image has 784 pixels (`p0` … `p783`), and pixel (r, c) is in column `p{r * 28 + c}`;
- a 14 × 14 image has 196 pixels (`p0` … `p195`), and pixel (r, c) is in column `p{r * 14 + c}`.

In Python you always work with two parallel lists: `images` (each image is a `list[int]`) and
`labels` (a `list[int]`). The kit's loader does the reading for you:

```python
train_images, train_labels = load_digits("data/mnist_train_10k.csv.gz", limit=1000)
```

A `.csv.gz` file is CSV text compressed with gzip. To read one with the standard library only:

```python
import csv, gzip

with gzip.open("data/mnist_test_2k.csv.gz", "rt", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)      # ['label', 'p0', 'p1', ...]
    first = next(reader)       # a list of strings: convert each one with int()
```

## Row order

The rows are shuffled with a fixed seed (2026), and **every block of 100 rows holds exactly 10 images of
each digit**. So the first 100, 1,000 or 2,000 rows always form a balanced set, and `limit=` gives you a
fair mix of digits.

## Data dictionary

| File | Labs | Rows × columns | Size | What it is |
|---|---|---|---|---|
| `mnist_train_10k.csv.gz` | 1–5 | 10,000 × 785 | 2.1 MB | Training images: 1,000 of each digit, from the MNIST training set. |
| `mnist_test_2k.csv.gz` | 1–5 | 2,000 × 785 | 0.43 MB | Test images: 200 of each digit, from the MNIST test set. Never train on them: they exist to measure honest accuracy. |
| `mnist_train_10k_14x14.csv.gz` | 2–5 | 10,000 × 197 | 0.88 MB | The same 10,000 images, in the same order, shrunk to 14 × 14. |
| `mnist_test_2k_14x14.csv.gz` | 2–5 | 2,000 × 197 | 0.18 MB | The same 2,000 test images, in the same order, shrunk to 14 × 14. |
| `mnist_sample_20.csv` | 1 | 20 × 785 | 40 KB | The first 20 rows of the training file as plain text, so you can open it and look. |
| `valid_postcodes.txt` | 2–5 | 55 lines | 0.3 KB | The postcodes Garabato delivers to: `28001` to `28055`, one per line. |
| `postcodes_test.csv` | 2, 5 | 300 × 7 | 10 KB | 300 handwritten postcode strips, built from the test images. |
| `depots.json` | 5 | 55 entries | 1.3 KB | Which fictional depot serves each postcode. |
| `holdout/league_holdout.csv.gz` | 5 | 1,000 × 785 | 0.22 MB | The holdout for the League final. |
| `MANIFEST.json` | all | – | – | Size and SHA-256 checksum of every data file. The setup cell uses it to tell you if a file is damaged. |

Other files in your kit's `data/` folder (for example the Lab 3 scanner export) are described in that lab's notebook.

## File details

### `mnist_train_10k.csv.gz` and `mnist_test_2k.csv.gz`

Balanced random subsets of MNIST: 1,000 images of each digit from the 60,000 training images, and 200 of each
digit from the 10,000 test images. Header `label,p0,…,p783`; every value is a whole number from 0 to 255.

### The 14 × 14 files

Each small pixel is the average of a 2 × 2 block of the big image, rounded down with floor division:
`(a + b + c + d) // 4`, where `a`, `b`, `c` and `d` are the four big pixels of the block. Small pixel
(row, col) comes from big rows `2 * row` and `2 * row + 1` and big columns `2 * col` and `2 * col + 1`.
Row *k* of a 14 × 14 file is exactly the shrunk version of
row *k* of the matching 28 × 28 file, with the same label. A 14 × 14 image has a quarter of the pixels, so each
distance is about four times faster, and accuracy stays almost the same.

### `mnist_sample_20.csv`

The header and the first 20 rows of `mnist_train_10k.csv.gz`, uncompressed. Open it in VS Code to see what one
image looks like as text.

### `valid_postcodes.txt`

The 55 codes `28001` … `28055`, one per line, nothing else. **This is a simplified list made for the lab, not an
official postcode list.** For the lab, a postcode is valid if, and only if, it appears in this file.

### `postcodes_test.csv`

| Column | Meaning |
|---|---|
| `strip_id` | `S001` … `S300` |
| `postcode` | the five-digit code the strip spells; always in `valid_postcodes.txt` |
| `i0` … `i4` | row numbers in `mnist_test_2k` (0 = the first image after the header). They are also indexes into the `images` and `labels` lists that `load_digits` returns, and they work the same for `mnist_test_2k_14x14`. The label of row `i0` is the first digit of the postcode, and so on. |

Example: `S003,28013,415,26,635,282,62` means that test images 415, 26, 635, 282 and 62, placed side by side,
spell 2-8-0-1-3. Every code appears 5 or 6 times. Every code starts with 2 and 8 and there are only 200 test images
of each digit, so images are reused across strips, but never twice in the same strip.

### `depots.json`

A JSON object such as `{"28001": "Retiro", "28002": "Retiro", "28003": "Chamartín", …}`: one entry for each of the
55 valid postcodes. The keys are strings (JSON keys always are). The values are the three fictional depots
**Chamartín** (19 codes), **Retiro** (18) and **Vallecas** (18). The assignment is invented and does not follow
real Madrid geography. The file is UTF-8, so open it with `encoding="utf-8"` to keep the accent in Chamartín.

### `holdout/league_holdout.csv.gz`

1,000 MNIST test images, 100 of each digit, none of which is in `mnist_test_2k`. Same layout as the 28 × 28
files. It is used once, by `league_final()` in Lab 5. A holdout only tells the truth the first time you use it,
so please don't open it, train on it or tune on it.

### `MANIFEST.json`

For every data file: `{"sha256": …, "bytes": …, "rows": …, "columns": …}`. `rows` counts data rows (header
excluded); `columns` is `null` for JSON.

## Where the images come from

MNIST is a classic public dataset of 70,000 handwritten digits (60,000 for training, 10,000 for testing), each
28 × 28 pixels. The files here are small, balanced samples of it. If you want all 70,000 images for an optional
numpy route, run `tools/get_full_mnist.py`.

## Licence

> The digit images are from the MNIST database by Yann LeCun, Corinna Cortes and Christopher J. C. Burges, licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). The subsets, downsampled versions, postcode strips, samples and defect-injected files in this kit are adaptations of MNIST: they were sampled, downsampled, recombined and deliberately corrupted for teaching. They are shared under the same licence. Garabato Parcels, its depots, scanners, operators and all metadata are entirely fictional; no real person's data is included.

More details are in `LICENSE-DATA.md`, in the kit's main folder.
