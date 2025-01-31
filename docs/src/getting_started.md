# Getting Started
This page provides instructions for installing modOpt
and running a minimal example.

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