"""web_app.py: Garabato's digit reader as a small web API (Lab 4 · ADVANCED BLD2; Theory S15-S16).

The drawing page tools/draw_digit.html sends {"pixels": [784 numbers]} to POST /predict and shows the answer.
This file is that API, written with FastAPI. Run it from a terminal in the kit folder:

    uv sync --group web          # once (it needs internet): installs FastAPI and uvicorn
    python web_app.py            # or: uv run python web_app.py

then open http://127.0.0.1:8000/ in a browser on this computer, draw a digit and press "Ask the server (Lab 4)".
http://127.0.0.1:8000/docs lists the API and lets you try it. Ctrl+C in the terminal stops the server.

You write two things, under the banner: the Drawing model (what /predict accepts) and the predict endpoint.
Everything marked "Provided" is already implemented: use it, don't modify it.
"""
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from labkit.lab4_tools import top3_scorer

KIT = Path(__file__).resolve().parent
MODEL_FILE = KIT / "models" / "centroids.json"      # made by the CORE: python digits.py train ...
PAGE = KIT / "tools" / "draw_digit.html"

# ----- Provided (don't modify) ---------------------------------------------------------------------------------
app = FastAPI(title="Garabato digits", description="Reads a digit drawn in tools/draw_digit.html.")
top3 = top3_scorer(MODEL_FILE)       # pixels -> the 3 closest average digits, e.g. [[7, 1.0], [1, 0.85], [9, 0.71]]
Pixel = Annotated[int, Field(ge=0, le=255)]         # one pixel: a whole number from 0 to 255 (like Field(gt=0), S16)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """GET /: the drawing page (Provided)."""
    return PAGE.read_text(encoding="utf-8")


# ===== YOUR CODE HERE =====  (BLD2: the contract and the endpoint)
class Drawing(BaseModel):
    """What POST /predict accepts: {"pixels": [...]}, exactly 784 pixels."""
    # TODO: one field, pixels: a list of Pixel, exactly 784 of them: Field(min_length=..., max_length=...)
    pass


# TODO: register predict as the POST /predict route (one decorator line, as in Theory S16)
def predict(drawing: Drawing) -> dict:
    """POST /predict: answer {"top3": top3(drawing.pixels)}. FastAPI calls it only with a valid Drawing."""
    pass


# ----- Provided: run the server ---------------------------------------------------------------------------------
if __name__ == "__main__":           # python web_app.py runs the server; import web_app (the notebook) doesn't
    import uvicorn

    if not MODEL_FILE.is_file():
        raise SystemExit("error: models/centroids.json not found: train a model first, e.g. "
                         "python digits.py train data/mnist_train_10k.csv.gz --out models/centroids.json")
    uvicorn.run(app, host="127.0.0.1", port=8000)   # 127.0.0.1: only this computer can reach the server
