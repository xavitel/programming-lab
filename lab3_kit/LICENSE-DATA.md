# Data licence

> The digit images are from the MNIST database by Yann LeCun, Corinna Cortes and Christopher J. C. Burges, licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). The subsets, downsampled versions, postcode strips, samples and defect-injected files in this kit are adaptations of MNIST: they were sampled, downsampled, recombined and deliberately corrupted for teaching. They are shared under the same licence. Garabato Parcels, its depots, scanners, operators and all metadata are entirely fictional; no real person's data is included.

## Source

- **Work:** the MNIST database of handwritten digits, by Yann LeCun, Corinna Cortes and Christopher J. C. Burges.
- **Copy used to build this kit:** `mnist.npz` from the Keras datasets mirror
  (https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz), 11,490,434 bytes,
  SHA-256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
  `tools/get_full_mnist.py` downloads and checks this same file.
- **Licence:** Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0),
  https://creativecommons.org/licenses/by-sa/3.0/

## What was changed

The files in `data/` are adaptations of MNIST, made for teaching:

- **sampled:** balanced random subsets (seed 2026), converted from numpy arrays to CSV text with a header;
- **downsampled:** 14 × 14 versions, each pixel the floor average of a 2 × 2 block;
- **recombined:** test images placed side by side as five-digit postcode strips, and fictional business
  metadata (valid postcodes, depots) added around them;
- **deliberately corrupted** (some labs only): scanner exports with injected defects such as inverted images,
  wrong labels and broken text.

## What this means for you

You may copy, share and adapt these files, for example in your own projects, as long as you:

1. credit MNIST and its authors, link to the licence, and say that the data was changed;
2. share any changed version of the data itself (an adaptation) under the same licence, CC BY-SA 3.0.

A credit line you can paste:

> Data: MNIST (LeCun, Cortes, Burges), CC BY-SA 3.0, adapted. Garabato Parcels is fictional.

This licence covers the data files only. It does not cover the notebooks, code or slides of the course.

## No personal data

Garabato Parcels, its depots (Chamartín, Retiro, Vallecas), its scanners, its operators and every piece of
metadata in these files are invented. Any e-mail address or phone number that appears in a lab file is fictional
and uses a reserved example domain or number range.
