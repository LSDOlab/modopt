# LBFGSB

The L-BFGS-B (Limited-memory BFGS with Bound constraints) algorithm, 
is a gradient-based optimization algorithm.
This solver uses the L-BFGS-B algorithm from the Scipy library.
```{note}
LBFGSB is a quasi-Newton optimization algorithm for large-scale bound-constrained problems.
Therefore, it does not support other types of constraints.
Please use general nonlinear programming algorithms like PySLSQP or IPOPT, 
if your problem has constraints other than optimization variable bounds.
```

To use LBFGSB, start by importing the optimizer as shown in the following code:
```py
from modopt import LBFGSB
```

Options could be set by just passing them within the `solver_options` dictionary when 
instantiating the LBFGSB optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the tolerance on the projected gradient `gtol` as shown below.

```py
optimizer = LBFGSB(prob, solver_options={'maxiter':1000, 'gtol':1e-6})
```

The options available for the LBFGSB solver in modOpt are given in the following table.
For more information on the Scipy L-BFGS-B algorithm, visit
**[Scipy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)**.

```{list-table} LBFGSB solver options
:header-rows: 1
:name: lbfgsb_options

* - Option
  - Type (default value)
  - Description
* - `maxfun`
  - *int* (`1000`)
  - Maximum number of function (objective) evaluations.
* - `maxiter`
  - *int* (`200`)
  - Maximum number of iterations.
* - `maxls`
  - *int* (`20`)
  - Maximum number of line search steps per major iteration.
* - `maxcor`
  - *int* (`10`)
  - Maximum number of variable metric corrections used to define \
    the limited memory Hessian approximation. \
    *LBFGSB* does not store the full Hessian but uses this many \
    terms to construct an approximation when required.
* - `ftol`
  - *float* (`2.22e-9`)
  - Terminate successfully if: \
    `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol`
* - `gtol`
  - *float*(`1e-5`)
  - Terminate successfully if: \
    `max{|proj g_i | i = 1, ..., n} <= gtol`, where `proj g_i` \
    is the i-th component of the projected gradient.
* - `iprint`
  - *int* (`-1`)
  - Verbosity of the console output.

    *<   0* suppresses all console outputs. \
    *=   0* prints a short summary upon completion. \
    *0<iprint<99* also prints `f^k` and `|proj g^k|` at every *iprint*-th iteration. \
    *=  99* also prints more scalar variables at each iteration. \
    *= 100* also prints more Cauchy search data, changes in the active set and final `x`. \
    *> 100* also prints vectors `x`, `g`, and Cauchy `x` at every iteration.
* - `callback`
  - *callable* (`None`)
  - Function to be called after each major iteration. \
    The function is called as`callback(xk, fk)`, where `xk` is the \
    optimization variable vector from the current major iteration, \
    and `fk` is the corresponding objective value.
```