# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../../lsdo_project_template/core'))     # for autodoc

# -- Project information -----------------------------------------------------

project = 'lsdo_project_template'
copyright = '2023, Anugrah'
author = 'Anugrah'
version = '0.1'
# release = 0.1.0rtc


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension", # autoapi is not needed when using autodoc
    # "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon", # another extension to read numpydoc style but 'numpydoc' extension looks better
    "numpydoc", # numpydoc already includes autodoc
    # "myst_parser", # compiles .md, .myst files
    "myst_nb", # compiles .md, .myst, .ipynb files
    "sphinx.ext.viewcode", # adds the source code for classes and functions in auto generated api ref
    "sphinxcontrib.collections", # adds files from outside src and executes functions before Sphinx builds
]

myst_enable_extensions = [
    # "amsmath",
    # "colon_fence",
    # "deflist",
    "dollarmath", # allow parsing: Inline math: $...$ , and Display (block) math: $$...$$
                  # Additionally if myst_dmath_allow_labels=True is set (the default):
                  # Display (block) math with equation label: $$...$$ (1)
    # "html_image",
]

autoapi_dirs = ["../../modopt/core"]

root_doc = 'welcome' # default: 'index'

# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'markdown',
#     '.ipynb': 'Jupyter notebook',
#     }

# source_parsers = {'.md': 'myst_nb',
#                 '.ipynb': 'myst_nb',
#                 }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme' # other theme options: 'alabaster', 'classic', 'sphinxdoc', 'nature', 'bizstyle', ...

# html_theme_options for sphinx_rtd_theme
html_theme_options = {
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    # 'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',   # other valid colors: 'white', ...
    # toc options
    'collapse_navigation': False,   # default: True
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': True     # default: False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Function used by collections for converting .py files from examples
# to .ipynb and writing those into `_temp/target/` directory before Sphinx builds

def py2nb(config):
    examples = [filename for filename in os.listdir(config['from']) if filename.startswith("ex_")]
    # os.mkdir(config['target']) 
    for ex in examples:
        nb = new_notebook()
        with open(config['from']+ex) as f:
            code = f.read()

        ex_name = ex[3:-3]
        nb.cells.append(new_markdown_cell('# '+ ex_name))
        nb.cells.append(new_code_cell(code))
        nbformat.write(nb, config['target']+ex[:-3]+'.ipynb')

    return

collections = {
    
    # copy_tutorials collection copies the contents inside `/tutorials` 
    # directory into `/src/_temp/tutorials`
   'copy_tutorials': {
      'driver': 'copy_folder',
      'source': '../examples/tutorials', # source relative to path of makefile, not wrt /src
      'target': 'tutorials/',
      'ignore': [],
    #   'active': True,         # default: True. If False, this collection is ignored during doc build.
    #   'safe': True,           # default: True. If True, any problem will raise an exception and stops the build.
      'clean': True,            # default: True. If False, no cleanup is done before collections get executed.
      'final_clean': True,      # default: True. If True, a final cleanup is done at the end of a Sphinx build.
    #   'tags': ['my_collection', 'dummy'],     # List of tags, which trigger an activation of the collection.
                                        # Should be used together with active set to False, 
                                        # otherwise the collection gets always executed.
                                        # Use -t tag option of sphinx-build command to trigger related collections.
                                        # e.g. : `sphinx-build -b html -t dummy . _build/html`
   },

    # convert_examples collection converts the contents inside `/examples` 
    # directory and writes into `/src/_temp/examples`
    'convert_examples': {
      'driver': 'writer_function',  # uses custom WriterFunctionDriver written by Anugrah
      'from'  : '../examples/',     # source relative to path of makefile, not wrt /src
      'source': py2nb,              # custom function written above in `conf.py`
      'target': 'examples/',        # target was a file for original FunctionDriver, e.g., 'target': 'examples/temp.txt'
                                    # the original FunctionDriver was supposed to write only 1 file.
    #   'active': True,   
    #   'safe': True,         
      'clean': True,       
      'final_clean': True,      
    #   'write_result': True,   # this prevents original FunctionDriver from writing to the target file
   },
}

collections_target = '_temp'        # default : '_collections', the default storage location for all collections
collections_clean  = True           # default : True, all configured target locations get wiped out at the beginning
                                    # can be overwritten for individual collection by setting value for the 'clean' key
collections_final_clean  = True     # default : True, all collections start their clean-up routine after a Sphinx build is done
                                    # can be overwritten for individual collection by setting value for the 'final_clean' key