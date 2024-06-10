
# Documentation

Before you build the docs by running the `make html` command in the `/docs` directory, 
ensure that you have installed modOpt with the `pip install -e .[docs]` command.

## Writing
Start by modifying the documentation pages by editing `.md` files in the `/src` directory.
Customize/add/remove pages from the template according to the updates made to the package.

For automatically generated API references, add docstrings to your modules, classes, functions, etc., and
then edit the list of directories containing files with docstrings intended for automatic API generation. 
This can be done by editing the line `autoapi_dirs = ["../../modopt/core"]` 
in `conf.py` in the `/src` directory.

Add Python files or Jupyter notebooks for examples with minimal explanations, and detailed Jupyter notebooks for 
tutorials into the `/examples` and `/tutorials` directories. 
Filenames for examples should start with'ex_'.
Add your examples and tutorials into the toctrees in `examples.md` and `tutorials.md` respectively.

## Building
Once you have all the source code written for your documentation, on the terminal/command line, run `make html`.
This will build all the html pages locally and you can verify if the documentation was built as intended by
opening the `docs/_build/html/welcome.html` on your browser.

## Hosting
### If you are doing a completely new version of docs from a new/forked repository for modOpt
On the /lsdolab *Read the Docs* account, **import** your project **manually** from github repository, and link the `/docs` directory.
The default website address will be generated based on your *Read the Docs* project name as `https://<proj_name>.readthedocs.io/`.
You can also customize the URL on *Read the Docs*, if needed.

### If you are just making modifications to the existing docs
Make sure to update the `docs/requirements-docs.txt` with dependencies for *Read the Docs* to build 
the documentation exactly as in your local build.
Optionally, edit the `.readthedocs.yml` in the project root directory for building with specific operating systems or versions of Python.
After you commit and push, *Read the Docs* will build your package on its servers and once its complete,
you will see your documentation online.