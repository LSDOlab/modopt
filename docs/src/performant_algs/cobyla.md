# COBYLA

Constrained Optimization BY Linear Approximation, also known as COBYLA, is a gradient-free optimization algorithm.
This solver uses the 'COBYLA' algorithm from the Scipy library.
```{note}
`COBYLA` cannot solve problems with equality constraints.
Please use other gradient-free algorithms such as `COBYQA` 
if your problem has equality constraints.
For better efficiency, we recommend using general nonlinear programming algorithms
such as `PySLSQP` or `IPOPT` if first order derivative information is available 
for the objective and constraints of your problem.
```

To use the `COBYLA` solver, start by importing it as shown in the following code:

```py
from modopt import COBYLA
```

Options could be set by just passing them within the `solver_options` dictionary when 
instantiating the `COBYLA` optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the absolute tolerance for the constraint violations `catol` as shown below.

```py
optimizer = COBYLA(prob, solver_options={'maxiter':1000, 'catol':1e-6})
```

A limited number of options are available for the `COBYLA` solver in modOpt as given in the following table.
For more information on the Scipy 'COBYLA' algorithm, visit
**[Scipy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html)**.

```{list-table} COBYLA solver options
:header-rows: 1
:name: cobyla_options

* - Option
  - Type (default value)
  - Description
* - `maxiter`
  - *int* (`1000`)
  - Maximum number of function evaluations.
* - `rhobeg`
  - *float*(`1.0`)
  - Reasonable initial changes to the variables.
* - `tol`
  - *float* (`1e-4`)
  - Final accuracy in the optimization (not precisely guaranteed). \
    This is a lower bound on the size of the trust region.
* - `catol`
  - *float* (`2e-4`)
  - Absolute tolerance for the constraint violations.
* - `disp`
  - *bool* (`False`)
  - Set to `True` to print convergence messages. \
    If `False`, no console outputs will be generated.
* - `callback`
  - *callable* (`None`)
  - Function to be called after each iteration. \
    The function is called as`callback(xk)`, where `xk` is the \
    optimization variable vector from the current iteration.
```