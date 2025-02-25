# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
    'mpire'
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

