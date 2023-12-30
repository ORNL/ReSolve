# -*- coding: utf-8 -*-
#
# ReSolve documentation build configuration file, created by
# sphinx-quickstart on Thu Nov 2 13:13:18 2023.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import shlex



# Modify Doxyfile for ReadTheDocs compatibility
with open('./doxygen/Doxyfile.in', 'r') as f:
    fdata = f.read()
fdata = fdata.replace('@PROJECT_SOURCE_DIR@', '.')
with open('./doxygen/Doxyfile.in', 'w') as f:
    f.write(fdata)
# The output directory needs to point to a directory within ../_readthedocs/ 
# by default readthedocs checks for html files within ../_readthedocs/ folder
# ../readthedocs folder does not exist locally only on the readthedocs server.
with open('./doxygen/Doxyfile.in', 'a') as f:
    f.write("\nOUTPUT_DIRECTORY=../_readthedocs/html/doxygen")

        # Call doxygen
    # from subprocess import call
    # call(['doxygen', "./doxygen/Doxyfile.in"])


# Get current directory
conf_directory = os.path.dirname(os.path.realpath(__file__))


# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinxcontrib.jquery',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'm2r2'
]

# The main toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = [os.path.join(conf_directory, 'sphinx/_templates')]

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.html', '.md']

project = 'ReSolve'
copyright = '2023, UT-Battelle, LLC, and Battelle Memorial Institute'
author = 'Kasia Świrydowicz, Slaven Peles'
release = '1.0.0'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

root_doc = 'index'

# -- Option for numbering figures/tables/etc.-----------------------------------
# Note: numfig requires Sphinx (1.3+)
numfig = True

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'English'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [ 
                     'cmake/blt/docs',
                     'thirdparty',
                     'doxygen-awesome-css'
                ]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
try:
    import sphinx_rtd_theme
except:
    html_theme = 'classic'
    html_theme_options = {
        'codebgcolor': 'lightgrey',
        'stickysidebar': 'true'
    }
    html_theme_path = []
else:
    html_theme = 'sphinx_rtd_theme'
    html_theme_options = {}
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


# Output file base name for HTML help builder.
htmlhelp_basename = 'ReSolve'

# primal, quest, sphinx:
# override wide tables in RTD theme
# (Thanks to https://rackerlabs.github.io/docs-rackspace/tools/rtd-tables.html)
# These folders are copied to the documentation's HTML output
html_static_path = ['./sphinx/style']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'theme_overrides.css',
]

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
#'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'ReSolve.tex', u'ReSolve Documentation',
   u'ORNL', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'resolve', u'ReSolve Documentation',
     [u'ORNL'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'ReSolve', u'ReSolve Documentation',
   'ReSolve Team', 'ReSolve', 'ReSolve is a library of GPU-resident linear solvers.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False
