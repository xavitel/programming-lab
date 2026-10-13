# Start here · Lab 1 · Read One Digit

Welcome! This folder holds everything you need for Lab 1: the notebook `lab1_read_one_digit.ipynb`, the helper package
`labkit/`, the data and a few tools. The CORE uses only Python's standard library, so if you already have
**Python 3.10 or newer**, there is nothing to install.

Nothing in this lab is graded or collected. Garabato Parcels is a fictional company.

## Pre-flight: 3 minutes, before class

1. **Extract the zip.** Windows: right-click `lab1_kit.zip` > **Extract All...** > **Extract**.
   macOS: double-click `lab1_kit.zip` in Finder.
2. **Open the folder in VS Code:** **File > Open Folder...** and choose the extracted `lab1_kit` folder.
3. **Open the notebook and select a Python 3.10+ kernel** (Route A, step 3 below).
4. **Run the first code cell** (click it, then press **Shift+Enter**) and wait until it prints `READY`.

Didn't get `READY`? The cell prints `STOP:` and the fix. Try the fix, then the table at the end of this file.
Still stuck? Bring the message to class, or use Route B (Colab).

## Choose a route

| Route | Use it when | Time |
|---|---|---|
| **A** · VS Code with a Python you already have | you have VS Code and Python 3.10+ (for example the theory course `.venv`) | about 2 min |
| **A′** · `uv sync` | you use uv, or Route A shows no Python 3.10+ | about 3 min, needs internet once |
| **B** · Google Colab | nothing works on your laptop | about 3 min, needs internet and a Google account |
| **C** · Pair with a neighbour | in class, while your laptop is being fixed | none |

There are no screenshots in this guide: every button and menu is named exactly as it appears on screen.

### Route A · VS Code with a Python you already have

1. **Extensions.** VS Code needs the **Python** and **Jupyter** extensions, both published by Microsoft.
   Open **View > Extensions** (Windows: Ctrl+Shift+X; macOS: Cmd+Shift+X), search for each name and click
   **Install** if it isn't installed yet.
2. **Open the kit folder.** **File > Open Folder...** (on both Windows and macOS) and choose the folder that contains
   `labkit` and `lab1_read_one_digit.ipynb`. If Windows created two nested folders (`lab1_kit\lab1_kit`), either one works.
   Then click `lab1_read_one_digit.ipynb` in the **Explorer** side bar (Windows: Ctrl+Shift+E; macOS: Cmd+Shift+E).
3. **Select the kernel.** At the top right of the notebook, click the kernel button. It says
   **Select Kernel**, or shows the Python it's using (for example `Python 3.9.6`).
   - If the list starts with **Select Another Kernel...**, click that first.
   - Click **Python Environments...**.
   - Pick a Python **3.10 or newer** (3.11+ recommended: it runs the lab about 3x faster), for example the theory course `.venv` (`Python 3.14`), or the one marked
     **Recommended** if it's 3.10+. On macOS, skip `/usr/bin/python3` when it says 3.9.
   - If VS Code asks to install **ipykernel**, click **Install** (it needs internet, once).
4. **Run the first code cell:** click inside it and press **Shift+Enter** (or click the ▶ button on its left).
   It checks your Python, finds the kit and prints `READY`.

Want to know which Pythons you have? Open **Terminal > New Terminal** in VS Code and type `py -0p` (Windows) or
`python3 --version` (macOS).

### Route A′ · `uv sync` (a fresh environment just for this kit)

1. Install uv once (needs internet), then close and reopen the terminal:
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS (Terminal): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. In VS Code, with the kit folder open: **Terminal > New Terminal**, then type `uv sync`. It creates a `.venv`
   folder inside the kit with Python 3.12 (the kit's `.python-version` asks for it: it runs the lab about 3x
   faster than 3.10) and `ipykernel`. uv uses your Python 3.12 if you have one, or downloads it.
3. Kernel button > **Select Another Kernel...** > **Python Environments...** > `.venv`. If `.venv` isn't listed
   yet, run **Developer: Reload Window** from the Command Palette (Windows: Ctrl+Shift+P; macOS: Cmd+Shift+P) and
   try again.
4. Only for the Challenges that use numpy: `uv sync --group extras` adds numpy and matplotlib.

### Route B · Google Colab

1. Go to `https://colab.research.google.com`, sign in, then **File > Upload notebook** and choose `lab1_read_one_digit.ipynb`
   from the extracted folder.
2. In the left side bar, click the **Files** icon (a folder). Click **Upload to session storage** (the page with an
   up arrow) and choose **`lab1_kit.zip`**, the zip file itself, not the folder. Wait until it appears in the list.
3. Run the first code cell (**Shift+Enter**). It finds the zip, extracts it and prints `READY`. If Colab warns that
   the notebook was not authored by Google, click **Run anyway**: it's the course notebook from IE Campus.
4. Colab forgets uploaded files when the session ends. Before you leave, use **File > Download > Download .ipynb**
   to keep your work. Next time, upload the notebook and the zip again.

### Route C · Pair with a neighbour

Share one laptop for now; Javier will fix your setup during the work block. Work alone or with a neighbour, your
choice.

## Running cells: Interrupt and Restart

- **Run a cell:** Shift+Enter. While a cell runs, it shows a spinning icon (VS Code) or `[*]`.
- **Interrupt** stops the running cell and keeps everything else in memory. Use it first when a cell runs for more
  than a minute. VS Code: the **Interrupt** button in the notebook toolbar at the top. Colab: the stop button on the
  cell, or **Runtime > Interrupt execution**.
- **Restart** wipes the kernel's memory: use it when Interrupt doesn't help, or when the notebook behaves
  strangely. VS Code: the **Restart** button in the notebook toolbar. Colab: **Runtime > Restart session**.
  After a restart, run the setup cell, the **Load the data** cell, then the cells with your `def`s. Quickest:
  click the cell you want to run, then **Execute Above Cells** (VS Code: the arrow next to the cell's run
  button) or **Runtime > Run before** (Colab).
- Your edits live in the notebook file: save with Ctrl+S (macOS: Cmd+S).

## The AI dial

A recommendation, not a ban:

- **CORE:** AI as a **tutor** or explainer. Ask *why*, not for the answer: Hint 4 of each task gives you a prompt.
  Try on your own for 7–10 minutes first.
- **OPTIONAL and ADVANCED tasks:** AI as a **reviewer**. It may comment on your code; keep only what you can explain
  line by line.
- **The Challenge (ADVANCED):** AI as a **pair programmer**. Verify with checks and tests, and say how you used it.
- **Always:** you own every line you run; verify it with `check()` or your own test; **never paste personal data
  into an AI tool**.

## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| `STOP: the notebook was opened from inside the zip file.` | Windows can show files inside a zip without extracting them, but Python can't use them from there. | Close the notebook. Right-click the zip > **Extract All...** > **Extract**, then **File > Open Folder...** on the extracted folder. |
| `STOP: this kernel runs Python 3.9...` (or older) | The kernel is too old. The macOS system Python (`/usr/bin/python3`) is often 3.9. | Route A, step 3: pick a 3.10+ kernel. None listed? Use Route A′, or install Python 3.12+ from python.org (Windows: tick **Add python.exe to PATH**). |
| `STOP: I can't find the labkit folder.` | Only the notebook was opened, or it was moved out of the kit folder. | **File > Open Folder...** on the whole kit folder, and keep the notebook next to `labkit`. |
| `'python' is not recognized as an internal or external command` (Windows) or `command not found: python` (macOS) | Windows uses the `py` launcher; macOS calls it `python3`. | Type `py` (Windows) or `python3` (macOS) instead of `python`, or `uv run python` with Route A′. |
| A cell runs forever (spinning icon or `[*]`) | A very long loop (for example over far more images than the task asks for) or a `while` loop that never ends. | **Interrupt**. Still busy after 10 seconds? **Restart**, then run the setup cell again. |
| "I fixed it but the check still fails" | The check uses the version of your function from the last time you ran its `def` cell. | Run the cell with `def` again, then the check. Still odd? **Restart**, then run the setup cell, the **Load the data** cell and your `def` cells again (**Execute Above Cells** does all of them). |
| `NameError: name 'check' is not defined` | The setup cell hasn't run since the kernel started. | Run the setup cell (the first code cell) again. |
| `NameError: name 'train_images' is not defined` (or `test_images`) | The **Load the data** cell hasn't run since the kernel started. | Run it, or click the cell you need and use **Execute Above Cells** (Colab: **Runtime > Run before**). |
| The setup cell says a data file is missing or damaged | The zip was only partly extracted, or a file was changed. | Delete the kit folder and extract the zip again. |
| **Select Kernel** shows no Python, or there's no **Python Environments...** | The Python or Jupyter extension is missing, or no Python is installed. | Install both extensions (Route A, step 1). Then install Python 3.12+ from python.org, or use Route A′. |
| VS Code asks to install `ipykernel` and you're offline | The chosen environment has no Jupyter kernel package. | Choose another 3.10+ environment, or use Route C now and Route A′ at home. |
| `Note: Anaconda 'base' detected.` | Information only. | Fine, as long as it's Python 3.10+. |

## What's in this folder

- `lab1_read_one_digit.ipynb`: the lab notebook. Start at the top.
- `labkit/`: the helper package behind `check()`, `rescue()` and the provided tools. You can read it:
  `labkit/reference/` holds clear reference versions of the CORE and OPTIONAL tasks, the ones `rescue()` hands you.
- `data/`: the data files and their description (`data/README.md`).
- `tools/`: `draw_digit.html` (draw your own digit in a browser) and `get_full_mnist.py` (only for numpy Challenges).
- `LEAGUE.md`: the rules of the opt-in Pure-Python League.
- `LICENSE-DATA.md`: the data licence (MNIST, CC BY-SA 3.0, adapted).
- `pyproject.toml`, `uv.lock` and `.python-version`: used only by Route A′.
- Some labs add their own files (for example a script or sample images); the notebook tells you when you need them.
