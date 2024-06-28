# ConvexQPSolvers

To use `ConvexQPSolvers`, first install the qpsolvers package with `pip install qpsolvers[wheels_only]`.
You can then follow the same process as for other optimizers
except when importing the optimizer.
You can import the optimizer as shown in the following code:

```py
from modopt import ConvexQPSolvers
```

Unlike the other solvers in modOpt, `ConvexQPSolvers` is a unified interface to
multiple convex QP solvers supported by the `qpsolvers` package.
Therefore, you need to install the relevant QP solver before you
can start optimizing your QP problems.
The QP solvers supported by the `qpsolvers` package are: 

- `'clarabel'`
- `'cvxopt'`
- `'daqp'`
- `'ecos'`
- `'gurobi'`
- `'highs'`
- `'hpipm'`
- `'mosek'`
- `'osqp'`
- `'piqp'`
- `'proxqp'`
- `'qpalm'`
- `'qpoases'`
- `'qpswift'`
- `'quadprog'`
- `'scs'`
- `'nppro'`

Solver options could be set by just passing them within the `solver_options` 
dictionary when instantiating the `ConvexQPsolvers` optimizer object.
Since `ConvexQPsolvers` contain several QP solver interfaces, the user
***must*** specify their `solver` choice within the `solver_options`.
The only two global options (meaning options that are not specific to your QP `solver` choice) are:

```{list-table} ConvexQPSolvers solver options
:header-rows: 1
:name: qpsolvers_options

* - Option
  - Type (default value)
  - Description
* - `solver`
  - *str* (None)
  - The QP solver to be used. Must always be specified. \
    Default value *None* will raise an error.
* - `verbose`
  - *bool* (True)
  - Set to `True` to print out extra information.

```

Options specific to the selected `solver` can also be specified and the complete list of options for all
supported solvers are available from the 
**[qpsolvers documentation](https://qpsolvers.github.io/qpsolvers/supported-solvers.html)**.
For example, we can set the maximum number of iterations `max_iter` 
and the runtime limit `time_limit` (in seconds) for the `osqp` solver as shown below.

```py
optimizer = ConvexQPSolvers(
    prob,
    solver_options={'solver':'osqp', 'max_iter': 1000, 'time_limit': 120}
    )
```

````{Note}
Since `ConvexQPsolvers` contain several QP solver interfaces, 
when calling the 'modopt.optimize()` function, the user has to specify the solver twice as shown below.
```py
from modopt import optimize

results = optimize(
    prob, 
    solver='ConvexQPSolvers', 
    solver_options={'solver': 'quadprog', 'verbose': False}
    )
```
````