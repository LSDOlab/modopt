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

To install the latest commit from the main branch, run the following command in the terminal
```sh
pip install git+https://github.com/lsdolab/modopt.git@main
```

### Uninstalling
To uninstall modOpt, run
```sh
pip uninstall modopt
```

### Upgrading
To upgrade to the latest commit, first uninstall modopt and then reinstall using
```sh
pip uninstall modopt
pip install git+https://github.com/lsdolab/modopt.git@main
```

### Installing and upgrading in development mode

To install `modopt` in development mode, first clone the repository and then install using
```sh
git clone https://github.com/lsdolab/modopt.git
pip install -e ./modopt
```
The `-e` flag installs the package in editable mode, 
allowing you to modify the source code without reinstallation.

To upgrade to the latest commit in development mode, run
```sh
cd /path/to/modopt
git pull
```

## Testing
To verify that the package works correctly, install `pytest` using
```sh
pip install pytest
```
and run the following command from the project's root directory:
```sh
pytest -m basic
```
The -m basic flag runs only the basic test cases, excluding 
visualization tests and tests for interfaces with optional dependencies 
such as Jax, CasADi, and others.

## Usage 

Here is a simple example that minimizes `x^2 + y^2` subject to `x + y = 1`.

```python
import numpy as np
import modopt as mo

# `v = [x, y]` is the vector of variables

# Define the problem functions
def objective(v):               # Objective function
    return v[0]**2 + v[1]**2

def constraint(v):              # Constraint function
    return np.array([v[0] + v[1]])

# Define the problem constants
x0 = np.array([1., 1.]) # Initial guess for the variables
cl = np.array([1.])     # Vector of lower bounds for the constraints
cu = np.array([1.])     # Vector of upper bounds for the constraints

# Create the problem and optimizer objects
problem   = mo.ProblemLite(x0=x0, obj=objective, con=constraint, cl=cl, cu=cu)
optimizer = mo.SLSQP(problem, solver_options={'ftol':1e-6})

# Solve the optimization problem and view the results
optimizer.solve()
optimizer.print_results()
```

We used the SLSQP optimizer to solve this problem.
Note that we did not provide functions for computing the objective gradient
or the constraint Jacobian.
In the absence of user-provided derivatives, `ProblemLite` estimates them
using first-order finite differences.
However, it is more efficient if the user provides the functions with exact derivatives.
For more complex examples with user-defined derivatives,
please see the [Examples](https://modopt.readthedocs.io/en/latest/src/examples.html)
in the documentation.

This example is provided only to help users to get started with modOpt.
See the [documentation](https://modopt.readthedocs.io/) for more advanced features
of modOpt.

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
