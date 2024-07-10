# COBYQA

Constrained Optimization BY Quadratic Approximations, also known as COBYQA, 
is a gradient-free optimization algorithm developed to improve upon 
the COBYLA algorithm.
COBYQA employs a derivative-free, trust-region SQP approach, 
utilizing quadratic models derived from underdetermined interpolation.
Unlike COBYLA, COBYQA also supports equality constraints and can handle
general nonlinear optimization problems.
```{note}
For better efficiency, we recommend using general nonlinear programming algorithms
such as PySLSQP or IPOPT, if first order derivative information is available 
for the objective and constraints of your problem.
```

To use the `COBYQA` solver, start by importing it as shown in the following code:

```py
from modopt import COBYQA
```

Options could be set by just passing them within the `solver_options` dictionary when 
instantiating the `COBYQA` optimizer object.
For example, we can set the maximum number of function evaluations `maxfev` 
and the tolerance on the maximum constraint violation `feasibility_tol` as shown below.

```py
optimizer = COBYQA(prob, solver_options={'maxiter':1000, 'feasibility_tol':1e-6})
```

The options available for the `COBYQA` solver in modOpt are given in the following table.
For more information on the COBYQA algorithm and for advanced options, visit
**[COBYQA documentation](https://www.cobyqa.com/stable/ref/generated/cobyqa.minimize.html)**.

```{list-table} COBYQA solver options
:header-rows: 1
:name: cobyqa_options

* - Option
  - Type (default value)
  - Description
* - `maxfev`
  - *int* (`500`)
  - Maximum number of function evaluations.
* - `maxiter`
  - *int* (`1000`)
  - Maximum number of iterations.
* - `target`
  - *float*(`-np.inf`)
  - Target value for the objective function. \
    Terminate succesfully if the objective function value of a \
    feasible point `xk` (see `feasibility_tol` below) is less than \
    or equal to `target`, i.e., `f(xk)<=target`.
* - `feasibility_tol`
  - *float* (`1e-8`)
  - Tolerance on the maximum constraint violation. \
    A point is considered feasible if the maximum constraint violation \
    at a point is less than or equal to this `feasibility_tol`.
* - `radius_init`
  - *float* (`1.0`)
  - Initial trust region radius. \
    Typically, this value should be in the order of one tenth of \
    the greatest expected change to `x0`.
* - `radius_final`
  - *float* (`1e-6`)
  - Final trust region radius. \
    Specifies the accuracy needed in the final values of the variables. 
* - `nb_points`
  - *int* (`2*n+1`)
  - Number of interpolation points used to build the quadratic model \
    of the objective and constraint functions. \
    Must satisfy `0<nb_points<=(n+1)*(n+2)//2`.
* - `scale`
  - *bool* (`False`)
  - Set to `True` to scale the variables according to the bounds. \
    If `True` and if all the lower and upper bounds are finite, \
    the variables are scaled to be within the range `[-1,+1]`. \
    If any of the lower/upper bounds is infinite, the variables are not scaled.
* - `filter_size`
  - *int* (`1e6`)
  - Maximum number of points in the filter. \
    The filter is used to select the best point returned by \
    the optimization procedure.
* - `store_history`
  - *bool* (`False`)
  - Set to `True` to store the history of the objective function values \
    and maximum constraint violations.
* - `history_size`
  - *int* (`1e6`)
  - Maximum number of function evaluations to store in the history.
* - `debug`
  - *bool* (`False`)
  - Set to `True` to perform additional checks during optimization. \
    Should be used only for debugging purposes and \
    is not recommended for general users.
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