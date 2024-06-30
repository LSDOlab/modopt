# CVXOPT

To use the CVXOPT solver, first install the '*cvxopt*' package with `pip install cvxopt`.
You can then follow the same process as for other optimizers
except when importing the optimizer.

```{Warning}
CVXOPT can only solve convex optimization problems.
Therefore, users should ensure their problems are convex before applying the optimizer.
ModOpt does not perform any checks to determine if the user-defined problem is indeed
a convex optimization problems.
Users must ensure, at a minimum, that the equality constraints are linear.
```

Import the optimizer as shown in the following code:

```py
from modopt import CVXOPT
```

Solver options could be set by just passing them within the `solver_options` 
dictionary when instantiating the CVXOPT optimizer object.
For example, we can set the maximum number of iterations `maxiters` 
and the convergence tolerance `abstol` for the algorithm as shown below.
```py
optimizer = CVXOPT(prob, solver_options={'maxiters': 100, 'abstol': 1e-9})
```

A limited number of options are available for the CVXOPT solver, as given in the following table.
More details about the nonlinear convex optimization algorithm and the solver options can be found 
**[here](https://cvxopt.org/userguide/solvers.html?highlight=parameters#algorithm-parameters)**.

```{list-table} CVXOPT solver options
:header-rows: 1
:name: cvxopt_options

* - Option
  - Type (default value)
  - Description
* - `show_progress`
  - *bool* (`True`)
  - Set to `False` to turn off the console output.
* - `maxiters`
  - *int* (`100`)
  - Maximum number of iterations.
* - `abstol`
  - *float* (`1e-7`)
  - Absolute accuracy.
* - `reltol`
  - *float* (`1e-6`)
  - Relative accuracy.
* - `feastol`
  - *float* (`1e-7`)
  - Tolerance for the feasibility conditions.
* - `refinement`
  - *int* (`1`)
  - Number of iterative refinement steps to use \
    when solving the KKT equations.
```


```{Note}
Like any other gradient-based optimizer, CVXOPT requires users to define the 
objective gradient and constraint Jacobian for their problem.
It additionally requires the Lagrangian or objective Hessian depending on whether
the problem has constraints or not.
However, for small problems, users can leverage the finite difference approximations for these
derivatives implemented in the `Problem`/`ProblemLite` classes in modOpt.
```