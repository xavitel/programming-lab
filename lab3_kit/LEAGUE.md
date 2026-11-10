# The Pure-Python League

How accurate can a digit reader get, within a fixed time budget? The League is the Challenge that runs through the
whole course. It's for fun and for learning to measure honestly.

- **Opt-in.** Take part if you enjoy it; nothing changes if you don't.
- **Pseudonyms only.** The whiteboard shows only pseudonyms, never names, and there is no shared sheet.
- **Not graded, nothing collected.** Results stay on your laptop until you choose to write your pseudonym and score
  on the whiteboard.

## The season

| When | What | Data you predict |
|---|---|---|
| Labs 1–5 | **tuning**, as often as you like: `league_run(..., on="validation")` | the last 2,000 training images (it learns from the first 8,000) |
| Labs 1–4 | **practice rounds**, once per idea: `league_run(...)` | the 2,000 public test images |
| Lab 5 | **the final, one run**: `league_final(...)` | a holdout of 1,000 MNIST test images kept apart for the final: none of them is in the public 2,000 |

An entry is two functions:

```python
def my_fit(train_images, train_labels):
    ...                       # learn something once, for example one average image per digit
    return model              # anything you like: a dict, a list, ...

def my_predict(image, model):
    ...                       # one image in, one label (0-9) out
    return label

result = league_run(my_predict, fit_fn=my_fit, on="validation")    # tune: as often as you like
result = league_run(my_predict, fit_fn=my_fit)                     # practice (Labs 1-4): once per idea
result = league_final(my_predict, fit_fn=my_fit, size=14)          # the final (Lab 5): once
```

Without `fit_fn`, your function gets the training data at every call: `my_predict(image, train_images, train_labels)`.
From Lab 2 on, `size=14` uses the 14 × 14 files. `help(league_run)` and `help(league_final)` show every option.

## Divisions

| | **stdlib** | **library** (the Lab 5 final) |
|---|---|---|
| Code | Python's standard library only (`math.dist`, `statistics`, ...) | numpy, scikit-learn and PyTorch are allowed |
| Training data | the 10,000-image kit subset (or its 14 × 14 version) | the kit subset, or the full 60,000 MNIST training images (`tools/get_full_mnist.py`), loaded inside `my_fit` |
| Time budget | 60 s to predict all the images, normalised by `machine_factor()` (a slower laptop gets more time); `my_fit` runs before the clock starts | the same |

`league_final` works out the division from what your code uses; `division="library"` puts a stdlib entry there too.
During Labs 1–4, a practice round that loads numpy counts as a **numpy** division: practice for the library division.

In the library division you also: explain every line of your code; report **validation** accuracy (from a split
of the training data); and never train or tune on `x_test` of `mnist.npz`, because the public test images and the
holdout both come from it.

## The rules (the League works on trust)

1. **Your predict function sees one image at a time, and only its pixels.** It never reads test labels, the test
   file or the holdout file.
2. **No memorised answers.** No test labels in your code (typed, pasted or saved to a file), and no looking up test
   images in the full MNIST file.
3. **Tune on validation, never on the test set.** Run the public test once per idea.
4. **The holdout is kept apart for the final: please don't open it.** `data/holdout/league_holdout.csv.gz` ships
   with the Lab 5 kit, and only `league_final` reads it. A holdout only tells the truth the first time, so the
   final is **one run**: your first run is saved in `output/league_final.json`.
5. **AI is a pair programmer here** (the Challenge setting of the AI dial): verify what it suggests, and say how you
   used it in your Engineering log.

Nothing is collected and nobody checks your code: this is an honour system. The League notices two things and says
so instead of printing a board line: your code reading the test or holdout file during the run, and a score far
above what such entries reach. If something is unclear, ask before the final.

## Ideas to get started

Downsampling to 14 × 14, several centroids per digit, pruning the training set, stopping a distance early once it
can't win, a multiclass perceptron, or (library division) a small neural network. Measure every idea: accuracy *and*
time.

## Scoreboard format (whiteboard)

`pseudonym · division · accuracy · normalised seconds`, for example `NightOwl · stdlib · 91.3% · 48.0 s`.
