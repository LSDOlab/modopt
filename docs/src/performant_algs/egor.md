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

Before using `Egor`, install `egobox`:

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
- runtime options are passed to `Egor.minimize(...)`

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
  - Maximum number of Egor iterations. Passed to `minimize()`.
* - `gp_config`
  - *egobox.GpConfig*
  - Surrogate model configuration.
* - `n_start`
  - *int* (`20`)
  - Number of starts for infill optimization.
* - `n_doe`
  - *int* (`0`)
  - Number of initial DOE points when DOE is not explicitly provided.
* - `doe`
  - *None*, *list*, *tuple*, or *ndarray* (`None`)
  - Initial DOE data.
* - `infill_strategy`
  - *egobox.InfillStrategy* (`LOG_EI`)
  - Infill criterion used during the EGO loop.
* - `cstr_infill`
  - *bool* (`False`)
  - Whether to use constrained infill strategy.
* - `cstr_strategy`
  - *egobox.ConstraintStrategy* (`MC`)
  - Constraint handling strategy for infill.
* - `qei_config`
  - *egobox.QEiConfig*
  - Configuration for qEI/multipoint sampling.
* - `infill_optimizer`
  - *egobox.InfillOptimizer* (`COBYLA`)
  - Internal optimizer for infill subproblems.
* - `trego`
  - *object* (`None`)
  - TREGO configuration.
* - `coego_n_coop`
  - *int* (`0`)
  - Number of cooperative groups for CoEGO mode.
* - `target`
  - *float* (`-max_float`)
  - Target objective value for early stopping.
* - `failsafe_strategy`
  - *egobox.FailsafeStrategy* (`REJECTION`)
  - Strategy when objective evaluations fail.
* - `seed`
  - *int* or `None` (`None`)
  - Random seed (forwarded to `minimize`).
* - `outdir`
  - *str* or `None` (`None`)
  - Runtime output directory for Egobox artifacts/checkpoints (forwarded to `minimize`).
* - `warm_start`
  - *bool* (`False`)
  - Runtime warm-start control (forwarded to `minimize`).
* - `hot_start`
  - *int* or `None` (`None`)
  - Runtime hot-start/checkpoint continuation control (forwarded to `minimize`).
* - `run_info`
  - *object* or `None` (`None`)
  - Optional Egobox run information object forwarded to `minimize`.
* - `timeout`
  - *float*, *int*, or `None` (`None`)
  - Runtime timeout forwarded to `minimize`.
* - `verbose`
  - *int*, *egobox.Verbosity*, or `None` (`None`)
  - Runtime verbosity level forwarded to `minimize`.
* - `cstr_tol`
  - *None*, *list*, *tuple*, or *ndarray* (`None`)
  - Constraint tolerance values used by Egobox.
* - `cstr_specs`
  - *None*, *list*, or *tuple* (`None`)
  - Constraint specs passed to the Egor constructor. For constrained modOpt problems, this is auto-generated from `cl`/`cu`.
* - `fcstrs`
  - *list* or *tuple* (`[]`)
  - Additional function constraints forwarded to `minimize`.
* - `fcstr_specs`
  - *list* or *tuple* (`[]`)
  - Function-constraint specs forwarded to `minimize` when supported by the installed egobox version.
```

```{note}
Detailed information on egobox objects can be retrieved using the python interpreter. See example below. 
```
```bash
> python
>>> help(egobox.GpConfig) 
```
