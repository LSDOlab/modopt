# SLSQP

To use SLSQP solver from the Scipy library, 
you can follow the same process as for other optimizers except when importing the optimizer.
Import the optimizer as shown in the following code:

```py
from modopt import SLSQP
```

Options could be set by just passing them within the `solver_options` dictionary when 
instantiating the SLSQP optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the precision goal `ftol` for the final solution as shown below.

```py
optimizer = SLSQP(prob, solver_options={'maxiter':20, 'ftol':1e-6})
```

A limited number of options are available for the SLSQP solver in modOpt as given in the following table.
For more information on the Scipy SLSQP algorithm, visit
**[Scipy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)**.

```{list-table} SLSQP solver options
:header-rows: 1
:name: slsqp_options

* - Option
  - Type (default value)
  - Description
* - `maxiter`
  - *int* (`100`)
  - Maximum number of iterations.
* - `ftol`
  - *float* (`1e-6`)
  - Precision goal for the final solution.
* - `disp`
  - *bool* (`False`)
  - Set to `True` to print convergence messages. \
    If `False`, no console outputs will be generated.
* - `callback`
  - *callable* (`None`)
  - Function to be called after each major iteration. \
    The function is called as `callback(xk)`, where `xk` is the \
    optimization variable vector from the current major iteration.
```