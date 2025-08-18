# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
import importlib.util

# -- Path setup --------------------------------------------------------------
# Add the project root so "import src" works
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Debug: check if modules can be found
for module in [
    "src.inference",
    "src.encode_latent_transformer",
    "src.train_autoencoder",
]:
    spec = importlib.util.find_spec(module)
    print(f"[SPHINX DEBUG] {module} found:", spec is not None)

# -- Project information -----------------------------------------------------
project = "2025_AI_Gait_Identification"
copyright = "2025, Jose Campos Cuiña"
author = "Jose Campos Cuiña"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

language = "english"

# If some heavy dependencies break imports, mock them here:
# autodoc_mock_imports = ["tensorflow", "torch", "sklearn"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
