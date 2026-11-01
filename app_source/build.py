# Builds the Kronos Forecast app for the machine you run it on (PyInstaller can't cross-build):
#   Windows -> dist/KronosForecast/KronosForecast.exe
#   macOS   -> dist/KronosForecast/KronosForecast        (run on a Mac)
#   Linux   -> dist/KronosForecast/KronosForecast        (run on Linux)
#   pip install -r requirements.txt pyinstaller  &&  python build.py
import os
import sys

import PyInstaller.__main__

HERE = os.path.dirname(os.path.abspath(__file__))
sep = ";" if sys.platform == "win32" else ":"
os.chdir(HERE)
PyInstaller.__main__.run([
    "kronos_app.py",
    "--name", "KronosForecast",
    "--onedir",            # one folder: starts fast (a single-file build would unpack ~1 GB on every launch)
    "--noconfirm", "--clean",
    "--add-data", f"assets{sep}assets",
    "--add-data", f"model{sep}model",
    "--hidden-import", "kronos_paths",
    "--hidden-import", "model.kronos",
    "--hidden-import", "model.module",
    # huggingface_hub imports safetensors lazily AND checks it is installed by reading its package
    # metadata, so both the code and the metadata have to be bundled explicitly.
    "--collect-submodules", "safetensors",
    "--copy-metadata", "safetensors",
    "--copy-metadata", "torch",
] + [a for m in (  # never used by the app; only pulled in because they happen to be installed
    "matplotlib", "tkinter", "IPython", "pytest", "numba", "llvmlite", "pyarrow", "scipy", "duckdb", "PIL",
    "lxml", "gevent", "altair", "pydantic", "pydantic_core", "sklearn", "lightgbm", "cryptography", "hf_xet",
    "pythonwin", "win32com", "jupyter", "notebook", "polars", "sqlalchemy", "openpyxl", "bs4", "plotly",
) for a in ("--exclude-module", m)])
