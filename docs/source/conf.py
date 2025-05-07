# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Add the project root directory

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'APLUS ML'
copyright = '2025, Michael Wornow'
author = 'Michael Wornow'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # Automatically document from docstrings
    'sphinx.ext.napoleon',   # Support Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',   # Link to source code
    'sphinx.ext.coverage',   # Check documentation coverage
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = [
    'numpy',
    'pandas',
    'matplotlib',
    'sklearn',
    'pydot',
    'simpy',
    'mpire',
    'mizani',
    'plotnine',
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

# -- Options for search output ----------------------------------------------
html_search_language = 'en'
html_search_options = {'type': 'default'}

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)