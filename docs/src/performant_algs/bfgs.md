# BFGS

The Broyden-Fletcher-Goldfarb-Shanno algorithm, also known as the BFGS algorithm, 
is a gradient-based optimization algorithm.
This solver uses the 'BFGS' algorithm from the Scipy library.
```{note}
`BFGS` is a quasi-Newton optimization algorithm for unconstrained problems.
Therefore, it does not support bounds or constraints.
Please use general nonlinear programming algorithms such as `PySLSQP` or `IPOPT`, 
if your problem has bounds or constraints.
```

To use the `BFGS` solver, start by importing it as shown in the following code:
```py
from modopt import BFGS
```

Options could be set by just passing them within the `solver_options` dictionary when 
instantiating the `BFGS` optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the tolerance on the gradient norm `gtol` as shown below.

```py
optimizer = BFGS(prob, solver_options={'maxiter':1000, 'gtol':1e-6})
```

The options available for the `BFGS` solver in modOpt are given in the following table.
For more information on the Scipy 'BFGS' algorithm, visit
**[Scipy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html)**.

```{list-table} BFGS solver options
:header-rows: 1
:name: bfgs_options

* - Option
  - Type (default value)
  - Description
* - `maxiter`
  - *int* (`200`)
  - Maximum number of iterations.
* - `gtol`
  - *float*(`1e-5`)
  - Terminate successfully if: `norm[gradient] <= gtol`.
* - `xrtol`
  - *float* (`0.0`)
  - Terminate successfully if \
    `norm[alpha*pk] <= (norm[xk] + xrtol) * xrtol`, \
    where `alpha*pk` is the step computed and `xk` is \
    the current optimization variable vector.
* - `norm`
  - *float* (`np.inf`)
  - Order of the norm to be used with the two termination criteria above. \
    `np.inf` represents maximum and `-np.inf` represents minimum.
* - `c1`
  - *float* (`1e-4`)
  - Armijo condition parameter.
* - `c2`
  - *float* (`0.9`)
  - Curvature condition parameter. Must satisfy `0 < c1 < c2 < 1`.
* - `hess_inv0`
  - *np.ndarray* \
    (`np.identity(n)`)
  - Initial estimate of the objective Hessian inverse. \
    Default is the identity matrix.
* - `return_all`
  - *bool* (`False`)
  - Set to `True` to return a list containing the best solution from \
    each major iteration in the final results dict.
* - `disp`
  - *bool* (`False`)
  - Set to `True` to print convergence messages. \
    If `False`, no console outputs will be generated.
* - `callback`
  - *callable* (`None`)
  - Function to be called after each major iteration. \
    The function is called as`callback(xk, fk)`, where `xk` is the \
    optimization variable vector from the current major iteration, \
    and `fk` is the corresponding objective value.
```