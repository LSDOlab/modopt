# modOpt

[![GitHub Actions Test Badge](https://github.com/LSDOlab/modopt/actions/workflows/install_test.yml/badge.svg)](https://github.com/LSDOlab/modopt/actions)
[![Coverage Status](https://coveralls.io/repos/github/LSDOlab/modopt/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/LSDOlab/modopt?branch=main)
[![Documentation Status](https://readthedocs.org/projects/modopt/badge/?version=latest)](https://modopt.readthedocs.io/en/latest/?badge=main)
[![License](https://img.shields.io/badge/License-GNU_LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

<!---
[![Python](https://img.shields.io/pypi/pyversions/modopt)](https://img.shields.io/pypi/pyversions/modopt)
[![Pypi](https://img.shields.io/pypi/v/modopt)](https://pypi.org/project/modopt/)
[![Pypi version](https://img.shields.io/pypi/v/modopt)](https://pypi.org/project/modopt/)
[![Forks](https://img.shields.io/github/forks/LSDOlab/modopt.svg)](https://github.com/LSDOlab/modopt/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/modopt.svg)](https://github.com/LSDOlab/modopt/issues) -->

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
<!-- ## Testing

## Usage -->

## Documentation
For API reference and more details on installation and usage, visit the [documentation](https://modopt.readthedocs.io/).

## Citation
If you use modOpt in your work, please use the following reference for citation:

```
@article{joshy2024modopt,
  title={modOpt: A modular development environment and library for optimization algorithms},
  author={Joshy, Anugrah Jo and Hwang, John T},
  journal={arXiv preprint arXiv:2410.12942},
  year={2024},
  doi={10.48550/arXiv.2410.12942}
}
```

## Bugs, feature requests, questions
Please use the [GitHub issue tracker](https://github.com/LSDOlab/modopt/issues) for reporting bugs, requesting new features, or any other questions.

## Contributing
We always welcome contributions to modOpt. 
Please refer the [`CONTRIBUTING.md`](https://github.com/LSDOlab/modopt/blob/main/CONTRIBUTING.md) 
file for guidelines on how to contribute.

## License
This project is licensed under the terms of the [GNU Lesser General Public License v3.0](https://github.com/LSDOlab/modopt/blob/main/LICENSE.txt).
