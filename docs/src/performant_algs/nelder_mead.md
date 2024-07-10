# NelderMead

The Nelder-Mead simplex algorithm is a local search, gradient-free optimization algorithm.
This solver uses the 'Nelder-Mead' algorithm from the Scipy library.
```{note}
`NelderMead` can solve only bound-constrained problems.
Please use other gradient-free algorithms such as `COBYLA` or `COBYQA` 
if your problem has constraints other than simple variable bounds.
For better efficiency, we recommend using general nonlinear programming algorithms
such as `PySLSQP` or `IPOPT` if first order derivative information is available 
for the objective and constraints of your problem.
```

To use the `NelderMead` solver, start by importing it as shown in the following code:

```py
from modopt import NelderMead
```

Options could be set by just passing them within the `solver_options` dictionary when 
instantiating the `NelderMead` optimizer object.
For example, we can set the maximum number of function evaluations `maxiter` 
and the minimum absolute error in `x_best` between iterations `xatol` as shown below.

```py
optimizer = NelderMead(prob, solver_options={'maxiter':1000, 'catol':1e-6})
```

The  options available for the `NelderMead` solver in modOpt are given in the following table.
For more information on the Scipy 'Nelder-Mead' algorithm, visit
**[Scipy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)**.

```{list-table} NelderMead solver options
:header-rows: 1
:name: nelder_mead_options

* - Option
  - Type (default value)
  - Description
* - `maxfev`
  - *int* (`1000`)
  - Maximum number of function evaluations.
* - `maxiter`
  - *int* (`1000`)
  - Maximum number of iterations. The optimization will stop as \
    soon as either of `maxfev` or `maxiter` is reached.
* - `xatol`
  - *float* (`1e-4`)
  - Terminate if absolute error in `x_best` between iterations is \
    less than `xatol`.
* - `fatol`
  - *float* (`1e-4`)
  - Terminate if absolute error in `f(x_best)` between iterations \
    is less than `fatol`. For convergence, both `xatol` and \
    `fatol` need to be satisfied.
* - `adaptive`
  - *bool* (`False`)
  - Set to `True` to adapt algorithm parameters to the \
    dimensionality of the problem. \
    Useful for high-dimensional problems.
* - `initial_simplex`
  - *np.ndarray* (`None`)
  - Initial simplex coordinates of shape `(n+1,n)`where `n=len(x)`. \
    If given, overrides `x0`. \
    `initial_simplex[j,:]` should contain the coordinates of \
    the *j*-th vertex of the `n+1` vertices of the simplex.
* - `return_all`
  - *bool* (`False`)
  - Set to `True` to return a list containing the best solution \
    from each iteration in the final results dict.
* - `disp`
  - *bool* (`False`)
  - Set to `True` to print convergence messages. \
    If `False`, no console outputs will be generated.
* - `callback`
  - *callable* (`None`)
  - Function to be called after each iteration. The function is \
    called as `callback(xk, fk)`, where `xk` is the optimization \
    variable vector from the current iteration, and `fk` is the \
    corresponding objective value.
```