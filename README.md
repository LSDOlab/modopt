# modOpt: A MODular development environment and library for OPTimization algorithms

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

modOpt is a modular development environment and library for optimization
algorithms, written in Python.
It is a platform designed to support research and education
in the field of numerical optimization.
Its modular development environment facilitates the construction of 
optimization algorithms using self-contained modules.
When implementing new algorithms, developers can reuse stable and efficient modules 
already available in modOpt, eliminating the need to build these components from scratch. 
Similarly, existing algorithms in modOpt can be customized for specific applications
by modifying only the relevant modules.

modOpt as a library includes several gradient-based and gradient-free optimization algorithms.
It provides interfaces to more than a dozen general-purpose optimizers, 
along with fully transparent implementations of several educational optimization algorithms.
Additionally, modOpt offers various features to support students, optimization practitioners, 
and advanced developers.
For instance, it includes built-in visualization and recording capabilities, 
interfaces to modeling frameworks such as JAX, CasADi, OpenMDAO and CSDL, and
an interface to the CUTEst test problem set.
It also provides various utilities for testing and benchmarking algorithms, 
and postprocessing optimization results.

modOpt is supported on Linux, macOS, and Windows.
The general-purpose optimizers available in modOpt include SLSQP, PySLSQP, OpenSQP,
SNOPT, IPOPT, Trust-Constr, InteriorPoint, BFGS, L-BFGS-B, Nelder-Mead, COBYLA, COBYQA, and CVXOPT.
The ConvexQPSolvers optimizer provides an interface to more than 15 QP solvers
available through the `qpsolvers` package.
Note that PySLSQP, SNOPT, IPOPT, COBYQA, CVXOPT, and qpsolvers must be 
installed separately if users wish to utilize them.
Similarly, the modeling frameworks JAX, CasADi, OpenMDAO, and CSDL 
need to be installed separately, if needed.

## Installation

To install the latest commit from the main branch, run the following command in the terminal:
```sh
pip install git+https://github.com/lsdolab/modopt.git@main
```

To install modOpt with all interfaced open-source optimizers, run:
```sh
pip install "modopt[open_source_optimizers] @ git+https://github.com/lsdolab/modopt.git@main"
```
This will install `pyslsqp`, `IPOPT`, `cobyqa`, `cvxopt`, and the `qpsolvers` package,
along with the QP solvers `quadprog` and `osqp`.
For details on obtaining a copy of `SNOPT`, visit the 
[SNOPT documentation](https://modopt.readthedocs.io/en/latest/src/performant_algs/snopt.html).

To install modOpt with JAX, CasADi, OpenMDAO, CSDL, and CSDL_alpha, run:
```sh
pip install "modopt[jax,casadi,openmdao,csdl,csdl_alpha] @ git+https://github.com/lsdolab/modopt.git@main"
```
Remove any dependencies from the list `[jax,casadi,openmdao,csdl,csdl_alpha]` above if they are not needed.

To uninstall modOpt, run:
```sh
pip uninstall modopt
```

To upgrade to the latest commit, uninstall modOpt and then reinstall it using:
```sh
pip uninstall modopt
pip install git+https://github.com/lsdolab/modopt.git@main
```

## Installation in development mode

To install modOpt in development mode, clone the repository and install it using:
```sh
git clone https://github.com/lsdolab/modopt.git
pip install -e ./modopt
```
The `-e` flag installs the package in editable mode, 
allowing you to modify the source code without needing to reinstall it.

To upgrade to the latest commit in development mode, navigate to the modOpt directory and run:
```sh
git pull
```

## Testing
To verify that the installed package works correctly, install `pytest` using:
```sh
pip install pytest
```
Then, run the following command from the project's root directory:
```sh
pytest -m basic
```
The `-m basic` flag runs only the basic test cases, excluding
visualization tests and tests for interfaces with
optional dependencies such as JAX, CSDL, OpenMDAO, and others.

## Usage 

The example below is provided to help users get started with modOpt.
For information on more advanced features, refer to the 
[documentation](https://modopt.readthedocs.io/).
The following example minimizes $x^2 + y^2$ subject to
the constraint $x + y = 1$.

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
However, it is more efficient if the user provides the functions for the exact derivatives.

For more complex examples, such as building models in various modeling languages, using different optimizers,
or developing new optimizers in modOpt with built-in modules,
visit the [Examples](https://modopt.readthedocs.io/en/latest/src/examples.html)
section of the documentation.

## Documentation
For API reference and more details on installation and usage, visit the [documentation](https://modopt.readthedocs.io/).

## Citation
If you use modOpt in your work, please use the following reference for citation:

```
@article{joshy2026modopt,
  title={modOpt: A modular development environment and library for optimization algorithms},
  author={Joshy, Anugrah Jo and Hwang, John T},
  journal={Advances in Engineering Software},
  volume={213},
  pages={104084},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.advengsoft.2025.104084}
}
```

## Bugs, feature requests, questions
Please use the [GitHub issue tracker](https://github.com/LSDOlab/modopt/issues) 
for reporting bugs, requesting new features, or any other questions.

## Contributing
We always welcome contributions to modOpt. 
Please refer the [`CONTRIBUTING.md`](https://github.com/LSDOlab/modopt/blob/main/CONTRIBUTING.md) 
file for guidelines on how to contribute.

## License
This project is licensed under the terms of the [GNU Lesser General Public License v3.0](https://github.com/LSDOlab/modopt/blob/main/LICENSE.txt).
