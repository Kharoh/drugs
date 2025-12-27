"""Sphinx configuration for the drugs project."""

import os
import sys


# Add project to path for autodoc
ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
	sys.path.insert(0, SRC)


project = "drugs"
author = "Kharoh"

extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.napoleon",
	"sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autoclass_content = "class"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
