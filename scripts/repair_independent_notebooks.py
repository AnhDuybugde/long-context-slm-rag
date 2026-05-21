from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "independent_variants"

SIMPLE_SETUP_CELL = '''# Simple Kaggle/Colab setup. Run this cell first.
# Do not force reinstall Kaggle's scientific stack; only install packages if missing.
import importlib.metadata as importlib_metadata
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "datasets": "datasets",
    "pyarrow": "pyarrow",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "torch": "torch",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "tqdm": "tqdm",
}

missing = [package for module, package in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import sklearn
import torch
from sentence_transformers import SentenceTransformer

def version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"

print("Dependency check OK:")
print("python", sys.version.split()[0])
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("torch", torch.__version__)
print("transformers", version("transformers"))
print("sentence-transformers", version("sentence-transformers"))
'''

LEIDEN_SETUP_CELL = '''# Simple Kaggle/Colab setup. Run this cell first.
# Do not force reinstall Kaggle's scientific stack; only install packages if missing.
import importlib.metadata as importlib_metadata
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "datasets": "datasets",
    "pyarrow": "pyarrow",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "torch": "torch",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "igraph": "igraph",
    "leidenalg": "leidenalg",
}

missing = [package for module, package in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import sklearn
import torch
from sentence_transformers import SentenceTransformer

def version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"

print("Dependency check OK:")
print("python", sys.version.split()[0])
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("torch", torch.__version__)
print("transformers", version("transformers"))
print("sentence-transformers", version("sentence-transformers"))
print("igraph", version("igraph"))
print("leidenalg", version("leidenalg"))
'''


REPLACEMENTS = {
    "The first run installs pinned binary packages and restarts the kernel.\n# After the kernel reconnects, rerun this cell once and then continue.": (
        "If packages are installed or changed, this cell stops with a clear restart message.\n"
        "# Restart the session manually, then rerun from this cell once."
    ),
    '"numpy": "2.0.2"': '"numpy": "1.26.4"',
    '"scipy": "1.14.1"': '"scipy": "1.13.1"',
    '"scikit-learn": "1.6.1"': '"scikit-learn": "1.5.2"',
    'print("Restarting the kernel now. After it reconnects, rerun this cell once.")\n    os._exit(0)': (
        'raise SystemExit("Restart the kernel/session manually, then rerun this cell once.")'
    ),
    'print("Removed conflicting packages:", ", ".join(installed_conflicts))\n    raise SystemExit("Restart the kernel/session manually, then rerun this cell once.")': (
        'print("Removed conflicting packages:", ", ".join(installed_conflicts))'
    ),
    "        self.model.to(self.device)\n": "        self.model.to(self.device)\n        self.model.eval()\n",
    "        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=2)\n": (
        "        with torch.inference_mode():\n"
        "            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)\n"
    ),
    "        outputs = self.generator.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=2)\n": (
        "        with torch.inference_mode():\n"
        "            outputs = self.generator.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=1)\n"
    ),
}


def update_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source")
        if isinstance(source, list):
            text = "".join(source)
        elif isinstance(source, str):
            text = source
        else:
            continue
        original = text
        if index == 1 and cell.get("cell_type") == "code" and "Kaggle/Colab setup" in text:
            text = LEIDEN_SETUP_CELL if "raptor_leiden_abstractive" in path.name else SIMPLE_SETUP_CELL
        for old, new in REPLACEMENTS.items():
            text = text.replace(old, new)
        if text != original:
            cell["source"] = text.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main() -> None:
    for path in sorted(NOTEBOOK_DIR.glob("qasper_*_standalone.ipynb")):
        if update_notebook(path):
            print(path)


if __name__ == "__main__":
    main()
