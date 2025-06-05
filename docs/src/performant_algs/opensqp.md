# OpenSQP

The `OpenSQP` solver is a gradient-based optimization algorithm that uses
sequential quadratic programming (SQP) to solve general nonlinear programming problems.
It uses a damped BFGS algorithm to approximate the Hessian of the Lagrangian and
an augmented Lagrangian merit function for the line search.
The `OpenSQP` solver is fully implemented in modOpt.
The source code is available 
**[here](https://github.com/LSDOlab/modopt/blob/main/modopt/core/optimization_algorithms/opensqp.py)**.

```{note}
For built-in algorithms like `OpenSQP`,
there are no "solver-specific" options,
as the entire solver is integrated within modOpt.
As a result, all solver options are set directly using keyword
arguments, rather than being passed separately in a
`solver_options` dictionary as in other performant algorithms.
```

To use the `OpenSQP` solver, start by importing it as shown in the following code:

```py
from modopt import OpenSQP
```

Options could be set by passing them as kwargs when 
instantiating the OpenSQP optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the optimality tolerance `opt_tol` shown below.

```py
optimizer = OpenSQP(prob, maxiter=20, opt_tol=1e-8)
```

See the table below for the full list of options.

```{list-table} OpenSQP (solver) options
:header-rows: 1
:name: opensqp_options

* - Option
  - Type (default value)
  - Description
* - `recording`
  - *bool* (`False`)
  - If `True`, records all outputs from the optimization. \
    This needs to be enabled for hot-starting the same problem later, \
    if the optimization is interrupted.
* - `hot_start_from`
  - *str* (`None`)
  - The record file from which to hot-start the optimization.
* - `hot_start_atol`
  - *float* (`0.0`)
  - The absolute tolerance check for the inputs \
    when reusing outputs from the hot-start record.
* - `hot_start_rtol`
  - *float* (`0.0`)
  - The relative tolerance check for the inputs \
    when reusing outputs from the hot-start record.
* - `visualize`
  - *list* (`[]`)
  - The list of scalar variables to visualize during the optimization.
* - `turn_off_outputs`
  - *bool* (`False`)
  - If `True`, prevents modOpt from generating any output files.
* - `maxiter`
  - *int* (`1000`)
  - Maximum number of major iterations.
* - `opt_tol`
  - *float*(`1e-7`)
  - Optimality tolerance.
* - `feas_tol`
  - *float* (`1e-7`)
  - Feasibility tolerance. Terminate successfully only if the \
    scaled maximum constraint violation is less than `feas_tol`.
* - `aqp_primal_feas_tol`
  - *float* (`1e-8`)
  - Tolerance for the primal feasibility of the augmented QP subproblem.
* - `aqp_dual_feas_tol`
  - *float* (`1e-8`)
  - Tolerance for the dual feasibility of the augmented QP subproblem.
* - `aqp_time_limit`
  - *float* (`5.0`)
  - Time limit for augmented QP solution in seconds.
* - `ls_min_step`
  - *float* (`1e-14`)
  -  Minimum step size for the line search.
* - `ls_max_step`
  - *float* (`1.0`)
  -  Maximum step size for the line search.
* - `ls_maxiter`
  - *int* (`10`)
  - Maximum number of iterations for the line search.
* - `ls_eta_a`
  - *float* (`1e-4`)
  - Armijo (sufficient decrease condition) parameter for the line search.
* - `ls_eta_w`
  - *float* (`0.9`)
  - Wolfe (curvature condition) parameter for the line search.
* - `ls_alpha_tol`
  - *float* (`1e-14`)
  - Relative tolerance for an acceptable step in the line search.
* - `readable_outputs`
  - *list* (`[]`)
  - List of outputs to be written to readable text output files. \
    Available outputs are: 'major', 'obj', 'x', 'lag_mult', 'slacks', \
    'constraints', 'opt', 'feas', 'sum_viol', 'max_viol', 'time', \
    'nfev', 'ngev', 'step', 'rho', 'merit', 'elastic', 'gamma', 'low_curvature'.
```