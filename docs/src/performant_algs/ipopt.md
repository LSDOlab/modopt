# IPOPT

IPOPT (Interior Point OPTimizer) is a gradient-based optimization algorithm that uses an 
interior point method to solve general nonlinear programming problems.
It can utilize second-order derivative information in the form of the Hessian of 
the objective for unconstrained problems or the Hessian of the Lagrangian for constrained 
problems. This solver uses the 'ipopt' algorithm from the CasADi library.

To use the `IPOPT` solver in modOpt, first install CasADi with `pip install casadi`.
You can then import the `IPOPT` solver from modOpt as shown in the following code:

```py
from modopt import IPOPT
```

Options for the IPOPT solver are available **[here](https://coin-or.github.io/Ipopt/OPTIONS.html)**.
Solver options could be set by just passing them within the `solver_options` 
dictionary when instantiating the IPOPT optimizer object.
For example, we can set the maximum number of iterations `max_iter` 
and the convergence tolerance `tol` for the algorithm as shown below.

```py
optimizer = IPOPT(prob, solver_options={'max_iter': 100, 'tol': 1e-6})
```

Some of the most commonly applicable options are given below:

```{list-table} IPOPT solver options
:header-rows: 1
:name: ipopt_options

* - Option
  - Type (default value)
  - Description
* - `max_iter`
  - *int* (`1000`)
  - Maximum number of iterations.
* - `max_wall_time`
  - *float* (`1e+20`)
  - Maximum number of walltime clock seconds.
* - `max_cpu_time`
  - *float* (`1e+20`)
  - Maximum number of cpu seconds.
* - `tol`
  - *float* (`1e-8`)
  - Convergence tolerance for the algorithm.
* - `print_level`
  - *int* (`5`)
  - Controls the verbosity level for the console output. \
    Valid range is between `0` and `12`. \
    The output becomes more detailed as this value increases.
* - `output_file`
  - *str* \
    (`"ipopt_output.txt"`)
  - Output filename.
* - `file_print_level`
  - *int* (`5`)
  - Controls the verbosity level for the output file. \
    Valid range is between `0` and `12`. \
    The file output becomes more detailed as this value increases.
* - `file_append`
  - *str* (`"no"`)
  - Determines whether to append to the output file. \
    Valid values are `"yes"` or `"no"`.
* - `print_user_options`
  - *str* (`"yes"`)
  - Determines whether to print all options set by the user. \
    Valid values are `"yes"` or `"no"`.
* - `print_advanced_options`
  - *str* (`"no"`)
  - Determines whether to print the advanced options also. \
    Valid values are `"yes"` or `"no"`.
* - `linear_solver`
  - *str* (`"mumps"`)
  - Linear solver to use for computing the step direction. \
    Valid values are "ma27",  "ma57",  "ma77",  "ma86",  "ma97",  \
    "pardiso", "pardisomkl", "spral", "wsmp", "mumps", or "custom".
* - `hessian_approximation`
  - *str* \
    (`"limited-memory"`)
  - Set `"exact"` to force IPOPT to use user-defined second derivatives \
    (objective/Lagrangian Hessian for unconstrained/constrained problems). \
    Default is "limited-memory" which uses a limited-memory quasi-Newton \
    Hessian approximation.
* - `print_timing_statistics`
  - *str* (`"no"`)
  - Set `"yes"` to measure and print time spent on selected tasks.
* - `derivative_test`
  - *str* (`"none"`)
  - To perform a derivative check at the at the initial guess `x0` \
    before the optimization. Possible values are:
    * `"none"`              : turn off derivative test
    * `"first-order"`       : check first derivatives 
    * `"second-order"`      : check both first and second derivatives
    * `"only-second-order"` : check only second derivatives
* - `derivative_test_print_all`
  - *str* (`"no"`)
  - Set `"yes"` to print information for all derivatives tested. \
    Default is `"no"` which prints information only for derivatives with \
    `relative error > derivative_test_tolerance`.
* - `derivative_test_perturbation`
  - *float* (`1e-8`)
  - Relative size of the finite difference perturbation in derivative test.
* - `derivative_test_tolerance`
  - *float* (`1e-4`)
  - Indicate derivatives as wrong if: \
    `relative error > derivative_test_tolerance`.

```