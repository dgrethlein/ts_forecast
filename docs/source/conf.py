# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# def setup(app):
#     app.add_css_file('style.css')


# -- Project information -----------------------------------------------------

project = 'ts_forecast'
copyright = '2026, David Grethlein'
author = 'David Grethlein'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.graphviz',
              'pyan.sphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'pydata_sphinx_theme'
# html_static_path = ['_static']

# html_css_files = ['style.css']

html_theme_options = {
    # "navigation_depth" : 10,
    # "sidebar_span" : 6
}

viewcode_line_numbers = True

add_module_names = False


# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True


# sphinx-rtd-dark-mode settings
# default_dark_mode = True
# default_dark_mode = False


# pyan3 settings
graphviz_output_format = "svg"
# graphviz_dot_options = ["Gnewrank=True"]

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python'        : ('https://docs.python.org/3', None),
    'pandas'        : ('https://pandas.pydata.org/docs/', None),
    'numpy'         : ('https://numpy.org/doc/stable/', None),
    'matplotlib'    : ('https://matplotlib.org/stable/', None),
    'sklearn'       : ('https://scikit-learn.org/stable/', None)
}
