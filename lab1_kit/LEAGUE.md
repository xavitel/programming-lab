# The Pure-Python League

How accurate can a digit reader get in plain Python, within a fixed time budget? The League is the Challenge that
runs through the whole course.

- **Opt-in.** Take part if you enjoy it; nothing changes if you don't.
- **Pseudonyms only.** Choose a pseudonym. The whiteboard shows only pseudonyms, never names, and there is no
  shared sheet.
- **Not graded, nothing collected.** Results stay on your laptop until you choose to write your pseudonym and score
  on the whiteboard.

## The season

| When | What counts | Data you predict |
|---|---|---|
| Labs 1–4 | tuning rounds, **as often as you like** | a validation split of the training data: `league_run(..., on="validation")` learns from the first 8,000 training images and scores the last 2,000 |
| Labs 1–4 | practice rounds, **once per idea**: this is the number for the board | the 2,000 public test images (`data/mnist_test_2k.csv.gz`): `league_run(...)` |
| Lab 5 | **the final: one run**, the official ranking | a **hidden holdout** of 1,000 MNIST test images that are not in the 2,000 public ones; it ships only in the Lab 5 kit |

Tuning and practice rounds use `league_run(...)`; the final uses `league_final(...)` (Lab 5 kit). An entry looks like
this:

```python
def my_fit(train_images, train_labels):
    ...                       # learn something once, for example one average image per digit
    return model              # anything you like: a dict, a list, ...

def my_predict(image, model):
    ...                       # one image in, one label (0-9) out
    return label

result = league_run(my_predict, fit_fn=my_fit, on="validation")    # tune: as often as you like
result = league_run(my_predict, fit_fn=my_fit)                     # the public test: once per idea
```

Without `fit_fn`, your function gets the training data at every call: `my_predict(image, train_images, train_labels)`.
From Lab 2 on, `league_run(..., size=14)` uses the 14 × 14 files. `help(league_run)` shows every option.

## Divisions

### Stdlib division (Labs 1–5)

- **Python's standard library only.** `math.dist` and `statistics` are allowed; numpy, pandas and every other
  third-party package are not.
- **Training data: the 10,000-image kit subset only** (`data/mnist_train_10k.csv.gz`, or its 14 × 14 version from
  Lab 2 on: they are the same images).
- **Time budget: 60 seconds** to predict all 2,000 test images, **normalised** by `machine_factor()`, so a slower
  laptop gets proportionally more time. The budget covers prediction; `fit_fn` runs before the clock starts (its time
  is reported too), but keep it to a few minutes so your laptop stays usable.

### Numpy division (final at Lab 5; you can practise from Lab 1)

- numpy is allowed, and you may train on the **full 60,000 MNIST training images**: `tools/get_full_mnist.py`
  downloads them.
- No machine-learning frameworks (no scikit-learn, PyTorch, TensorFlow, JAX, Keras or similar).
- You can explain every line of your code.
- You report **validation** accuracy (from a split of the training data), not test accuracy.
- Never train or tune on `x_test` from `mnist.npz`: the public test images and the holdout both come from it.

## Rules for both divisions

1. **Your `predict` function sees one image at a time, and only its pixels.** It must not read test labels, the
   test file or the holdout file.
2. **No memorised answers.** Your code must not contain test labels (typed, pasted or saved to a file), and must not
   look up test images in the full MNIST file to find their labels.
3. **Tune honestly.** Choose settings on a validation split of the training data (`on="validation"` does it for
   you), never on the test set. Run the public test once per idea.
4. **The final is one run.** A holdout only tells the truth the first time you use it, so please don't open, train
   on or tune on `data/holdout/league_holdout.csv.gz`.
5. **AI is a pair programmer here** (the Challenge setting of the AI dial): verify what it suggests, and say how you
   used it in your Engineering log.

These rules work on trust: nothing is collected, and `league_run` can't see everything your code does. It does
notice two things and says so instead of printing a board line: your code reading the test or holdout file during
the run, and a score of 99% or more (far beyond plain-Python readers on these images). If something is unclear,
ask Javier before the final.

## Ideas to get started

Downsampling to 14 × 14, several centroids per digit, pruning the training set, stopping a distance early once it
can't win, or a multiclass perceptron. Measure every idea: accuracy *and* time.

## Scoreboard format (whiteboard)

`pseudonym · division · accuracy · normalised seconds`, for example `NightOwl · stdlib · 91.3% · 48 s`.
