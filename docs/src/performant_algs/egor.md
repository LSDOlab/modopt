# Egor (EGObox)

Egor is a gradient-free global optimizer from the [EGObox library](https://github.com/relf/EGObox).
As a [bayesian optimizer](https://en.wikipedia.org/wiki/Bayesian_optimization), it is used to optimize expensive-to-evaluate black-box functions. 

The modOpt `Egor` wrapper supports continuous design variables with finite bounds and
nonlinear constraints with all standard bound forms:

- upper-bounded constraints `c(x) <= u`
- lower-bounded constraints `c(x) >= l`
- equality constraints `c(x) = v`
- double-sided constraints `l <= c(x) <= u`

```{warning}
Egor in modOpt currently requires finite lower and upper bounds on every design variable.
Problems with unbounded variables are rejected.
```

Before using `Egor`, install `egobox` (version 0.38 or later):

```sh
pip install egobox
```

or install via modOpt extras:

```sh
pip install "modopt[egobox]"
```

To use the `Egor` solver in modOpt, import it as:

```py
from modopt import Egor
```

Then instantiate it with optional `solver_options`:

```py
optimizer = Egor(prob, solver_options={"max_iters": 50, "n_doe": 10, "seed": 42})
```

modOpt forwards Egor options through two paths:

- constructor options are passed to `egobox.Egor(...)`
- runtime options (`max_iters`, `seed`, `outdir`, `warm_start`, `hot_start`,
  `run_info`, `timeout`, `verbose`, `stop_on_error`, `fcstrs`, `fcstr_specs`)
  are passed to `Egor.minimize(...)`

```{note}
For constrained problems, do not pass `solver_options['cstr_specs']` directly.
The modOpt wrapper builds `cstr_specs` automatically from `cl` and `cu`.
```

```{list-table} Egor solver options in modOpt
:header-rows: 1
:name: egor_options

* - Option
  - Type (default value)
  - Description
* - `max_iters`
  - *int* (`20`)
  - Maximum number of Egor iterations. Passed to \
    `minimize()`.
* - `gp_config`
  - *egobox.GpConfig*, *dict*, or `None` (`None`)
  - GP configuration used by the optimizer, see \
    GpConfig for details. When `None`, the egobox \
    default is used.
* - `infill_n_start`
  - *int* or `None` (`None`)
  - Number of runs of infill strategy optimizations; \
    the best result is taken. When `None`, the \
    egobox default is used. Replaces the deprecated \
    `n_start` option.
* - `n_doe`
  - *int* (`0`)
  - Number of samples of initial LHS sampling, used \
    when DOE is not provided by the user. When 0, \
    the number of points is computed automatically \
    regarding the number of input variables of the \
    function under optimization.
* - `x_doe`
  - *None*, *list*, *tuple*, or *ndarray* (`None`)
  - Initial DOE inputs, shape `(ns, nx)`. When \
    `y_doe` is not given, the `ns` points are \
    evaluated first. Replaces the deprecated `doe` \
    option (`doe[:, :nx]`).
* - `y_doe`
  - *None*, *list*, *tuple*, or *ndarray* (`None`)
  - Outputs at `x_doe`, shape `(ns, 1 + n_cstr)`: \
    the objective, then the raw values of the \
    modOpt constraints with at least one finite \
    bound, in order. Requires `x_doe`.
* - `infill_strategy`
  - *egobox.InfillStrategy* (`LOG_EI`)
  - Infill criterion used to decide the next \
    promising point.
* - `feasible_infill_strategy`
  - *egobox.FeasibleInfillStrategy* (`NONE`)
  - Strategy to take feasibility into account in the \
    infill criterion (Expected Feasible Improvement): \
    `NONE`, `EFI_P`, or `EFI_FE`. `EFI_P` and \
    `EFI_FE` require `infill_strategy` to be `EI`, \
    `WB2`, or `WB2S`.
* - `cstr_infill`
  - *bool* (`False`)
  - Activates the constrained infill criterion, \
    where the product of probability of feasibility \
    is used as a factor of the infill criterion.
* - `cstr_strategy`
  - *egobox.ConstraintStrategy* (`MC`)
  - Constraint management strategy for infill; use \
    the mean value or the upper trusted bound.
* - `qei_config`
  - *egobox.QEiConfig*, *dict*, or `None` (`None`)
  - Configuration for parallel qEI, also known as \
    batch or multipoint evaluation. `q` points are \
    selected at each iteration of the EGO algorithm.
* - `infill_optimizer`
  - *egobox.InfillOptimizer* (`COBYLA`)
  - Internal optimizer used to optimize the infill \
    criterion; either `COBYLA` or `SLSQP`.
* - `trego`
  - *object* (`None`)
  - TREGO configuration to activate TREGO strategy \
    for global optimization.
* - `coego_n_coop`
  - *int* (`0`)
  - Number of cooperative component groups used by \
    the CoEGO algorithm.
* - `target`
  - *float* or `None` (`None`)
  - Known optimum used as a stopping criterion. \
    When `None`, no target is used.
* - `failsafe_strategy`
  - *egobox.FailsafeStrategy* (`REJECTION`)
  - Strategy to handle objective computation failure \
    (NaN values or errors): `REJECTION`, \
    `IMPUTATION`, or `VIABILITY`.
* - `seed`
  - *int* or `None` (`None`)
  - Random generator seed to allow computation \
    reproducibility.
* - `outdir`
  - *str* or `None` (`None`)
  - Directory to write optimization history and use \
    as a search path for warm-start DOE.
* - `warm_start`
  - *bool* (`False`)
  - Start by loading initial DOE from `outdir`.
* - `hot_start`
  - *bool*, *int*, or `None` (`None`)
  - When `True`, `hot_start` behaves like \
    `hot_start = 0` with no iteration extension. \
    When `hot_start >= 0`, the optimizer state is \
    saved and optimization restarts from a previous \
    checkpoint.
* - `run_info`
  - *object* or `None` (`None`)
  - Optional run information object used for \
    logging and saving results.
* - `timeout`
  - *float*, *int*, or `None` (`None`)
  - Optional timeout in seconds. The optimization \
    stops when the elapsed time exceeds this \
    duration.
* - `verbose`
  - *int*, *egobox.Verbose*, or `None` (`None`)
  - Logging verbosity level. Default is `None`, \
    which means `Verbose.ERROR` and possible \
    control by the `EGOBOX_LOG` environment \
    variable.
* - `stop_on_error`
  - *bool* (`False`)
  - If `True`, terminate the optimization when the \
    objective function raises an error. Otherwise, \
    the error is handled according to \
    `failsafe_strategy`.
* - `cstr_tols`
  - *None*, *float*, *list*, *tuple*, or *ndarray* \
    (`None`)
  - Constraint violation tolerances: a scalar, or one \
    value per modOpt constraint with at least one \
    finite bound. Each one is set as the tolerance \
    of the corresponding `CstrSpec`. Use `CstrSpec` \
    tolerances in `fcstr_specs` for function \
    constraints. For equality and double-sided \
    constraints, the single value applies to both \
    internal `c(x) <= 0` constraints. Replaces the \
    deprecated `cstr_tol` option.
* - `cstr_specs`
  - *None*, *list*, or *tuple* (`None`)
  - Optional list of `CstrSpec` objects describing \
    how each surrogate-modeled constraint should be \
    interpreted.
* - `fcstrs`
  - *list* or *tuple* (`[]`)
  - List of constraints defined as functions.
* - `fcstr_specs`
  - *list* or *tuple* (`[]`)
  - Optional list of `CstrSpec` objects, one per \
    function constraint.
```

```{note}
`n_start`, `doe` and `cstr_tol` are deprecated, use `infill_n_start`, `x_doe`/`y_doe`
and `cstr_tols` instead.
They still work but emit a `DeprecationWarning`.
```

In the results, `y_doe` holds the objective followed by the raw values of the
modOpt constraints with at least one finite bound, as for the `y_doe` option.

```{note}
Detailed information on `egobox` objects can be retrieved using the python interpreter. See example below. 
```
```bash
> python
>>> import egobox
>>> help(egobox.GpConfig) 
```
