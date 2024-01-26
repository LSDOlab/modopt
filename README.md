# Modopt

[![Forks](https://img.shields.io/github/forks/LSDOlab/modopt.svg)](https://github.com/LSDOlab/modopt/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/modopt.svg)](https://github.com/LSDOlab/modopt/issues)

Please visit the [documentation](https://modopt.readthedocs.io/) for usage instructions.

## Installation

### Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/lsdolab/modopt.git
```
To update modopt to the latest version, run on the terminal or command line
```sh
pip uninstall modopt
pip install git+https://github.com/lsdolab/modopt.git
```

### Installation instructions for developers
To install `modopt`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/lsdolab/modopt.git
pip install -e ./modopt
```
To update modopt to the latest version, run on the terminal or command line
```sh
cd /path/to/modopt
git pull
```
To run the tests, install and run `pytest` with
```sh
pip install pytest
pytest
```

# For Developers
For updating documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.

# License
This project is licensed under the terms of the **GNU Lesser General Public License v3.0**.
