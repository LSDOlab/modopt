# Modopt

<!---
[![Python](https://img.shields.io/pypi/pyversions/modopt)](https://img.shields.io/pypi/pyversions/modopt)
[![Pypi](https://img.shields.io/pypi/v/modopt)](https://pypi.org/project/modopt/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
[![GitHub Actions Test Badge](https://github.com/LSDOlab/modopt/actions/workflows/actions.yml/badge.svg)](https://github.com/modopt/modopt/actions)
-->

<!-- [![Forks](https://img.shields.io/github/forks/LSDOlab/modopt.svg)](https://github.com/LSDOlab/modopt/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/modopt.svg)](https://github.com/LSDOlab/modopt/issues) -->
<!-- [![Pypi version](https://img.shields.io/pypi/v/modopt)](https://pypi.org/project/modopt/) -->
[![GitHub Actions Test Badge](https://github.com/LSDOlab/modopt/actions/workflows/install_test.yml/badge.svg)](https://github.com/LSDOlab/modopt/actions)
[![Coverage Status](https://coveralls.io/repos/github/LSDOlab/modopt/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/LSDOlab/modopt?branch=main)
[![Documentation Status](https://readthedocs.org/projects/modopt/badge/?version=latest)](https://modopt.readthedocs.io/en/latest/?badge=main)
[![License](https://img.shields.io/badge/License-GNU_LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

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
