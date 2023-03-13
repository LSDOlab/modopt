
# Documentation

If you are not interested in using this repository as a template but only want to use the documentation template, 
you can just copy the `/docs` directory and the `.readthedocs.yaml` file into your package root.
However, make sure you have all the dependencies mentioned in the `setup.py` file installed before you build your
documentation.

## Writing
Start by modifying the documentation pages by editing `.md` files in the `/src` directory.
Customize/add/remove pages from the template according to your package's requirements.

For automatically generated API references, add docstrings to your modules, classes, functions, etc., and
then edit the list of directories containing files with docstrings intended for automatic API generation. 
This can be done by editing the line `autoapi_dirs = ["../../lsdo_project_template/core"]` 
in `conf.py` in the `/src` directory.

Add Python files for examples and Jupyter notebooks for tutorials into the main project repository. 
Filenames for examples should start with'ex_'.
Add your examples and tutorials to the toctrees in `examples.md` and `tutorials.md` respectively.

## Building
Once you have all the source code written for your documentation, on the terminal/command line, run `make html`.
This will build all the html pages locally and you can verify if the documentation was built as intended by
opening the `docs/_build/html/welcome.html` on your browser.

## Hosting
On your/lsdolab *Read the Docs* account, **import** your project **manually** from github repository, and link the `/docs` directory.
Make sure to edit `requirements.txt` with dependencies for *Read the Docs* to build the documentation exactly
as in your local build.
Optionally, edit the `.readthedocs.yml` in the project root directory for building with specific operating systems or versions of Python.
After you commit and push, *Read the Docs* will build your package on its servers and once its complete,
you will see your documentation online.
The default website address will be generated based on your *Read the Docs* project name as `https://<proj_name>.readthedocs.io/`.
You can also customize the URL on *Read the Docs*, if needed.
