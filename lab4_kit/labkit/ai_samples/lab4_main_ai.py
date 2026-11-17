# Illustrative code written by the instructor in the style of an AI assistant's answer. Not real AI output.
#
# The prompt (imagined): "Write cmd_predict and main() for my digits.py command-line tool (argparse
# subcommands train and predict). Make the error handling robust and user-friendly."
#
# Lab 4, after the CORE: this main() has THREE problems. Run it with run_cli, read it, and find them.
"""digits.py, as an AI assistant might write it: it works, it looks careful, and it hides three problems."""
from __future__ import annotations

import sys
from pathlib import Path

from labkit.data import read_image_csv, read_image_grid
from labkit.reference.lab1 import predict_centroid
from labkit.reference.lab4 import build_parser, centroids_of, cmd_inspect, cmd_train, load_model, picture


def read_image(path: str) -> list[int]:
    """Read a .csv or .txt image file."""
    if path.endswith(".csv"):
        return read_image_csv(path)
    return read_image_grid(path)


def cmd_predict(args) -> str:
    """Predict the digit in an image file, with robust error handling."""
    model = load_model(Path(args.model))
    try:
        image = read_image(args.image)
    except:
        print("Error: could not read the image file. Please check the file and try again.")
        sys.exit(1)
    label = predict_centroid(image, centroids_of(model))
    if args.show:
        return picture(image) + "\n" + f"Prediction: {label}"
    return f"Prediction: {label}"


def main(argv=None) -> int:
    """Entry point of the command-line tool, with user-friendly error messages."""
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {"inspect": cmd_inspect, "train": cmd_train, "predict": cmd_predict}
    try:
        result = commands[args.command](args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
