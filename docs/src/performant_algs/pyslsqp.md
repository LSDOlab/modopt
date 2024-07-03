# PySLSQP

Before using PySLSQP, make sure to have pyslsqp installed on your machine.
You can use `pip install pyslsqp` to install from PyPI.
To use PySLSQP, you can follow the same process for other optimizers
except when importing the optimizer.

You need to import the optimizer as shown in the following code:

```py
from modopt import PySLSQP
```

Options could be set by just passing them within the `solver_options` dictionary  when 
instantiating the PySLSQP optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the accuracy goal `acc` for the final solution as shown below.

```py
optimizer = PySLSQP(prob, solver_options={'maxiter': 20, 'acc': 1e-6})
```

The complete list of options available for the PySLSQP solver in modOpt are given in the following table.
Please visit the **[pySLSQP documentation](https://pyslsqp.readthedocs.io)** for more details.

```{list-table} PySLSQP solver options
:header-rows: 1
:name: pyslsqp_options

* - Option
  - Type (default value)
  - Description
* - `maxiter`
  - *int* (`100`)
  - Maximum number of iterations.
* - `acc`
  - *float* (`1e-6`)
  - Accuracy (optimality) of the solution.
* - `iprint`
  - *int* (`1`)
  - Verbosity of the console output.\
    *iprint <= 0* suppresses all console outputs. \
    *iprint  = 1* prints the final result summary upon completion. \
    *iprint >= 2* prints the status of each major iteration and \
    the final result summary.
* - `callback`
  - *callable* (`None`)
  - Function to be called after each major iteration. \
    The function is called as`callback(x)`, where ``x`` is the \
    optimization variable vector from the current major iteration.
* - `summary_filename`
  - *str* (`"slsqp_summary.out"`)
  - Name of the readable summary file.
* - `visualize`
  - *bool* (`False`)
  - Set to `True` to visualize the optimization process. \
    Only major iterations are visualized.
* - `visualize_vars`
  - *list* (see description)
  - List of scalar variables to visualize. \
    Default value is `[‘objective’, ‘optimality’, ‘feasibility’]`. \
    Available variables are ['x[i]', 'objective', 'optimality', \
    'feasibility', 'constraints[i]', 'gradient[i]', 'multipliers[i]',\
    'jacobian[i,j]']
* - `keep_plot_open`
  - *bool* (`False`)
  - If `True`, the plot window will remain open after optimization.
* - `save_figname`
  - *str* (`"slsqp_plot.pdf"`)
  - Name of the file to save the visualized plot.
* - `save_itr`
  - *str* (`None`)
  - Set to `"all"` to save all iterations. \
    Set to `"major"` to save only major iterations. \
    By default, `save_itr` is None, and no iterations are saved.
* - `save_vars`
  - *list* (see description)
  - List of variables to save. \
    Default value is `['x', ‘objective’, ‘optimality’, ‘feasibility’,
    ‘step’, ‘mode’, ‘iter’, ‘majiter’, ‘ismajor’]`. \
    The full list of available variables are ['x', 'objective', \
    'optimality', 'feasibility', ‘step’, ‘mode’, ‘iter’, ‘majiter’, \
    ‘ismajor’, 'constraints', 'gradient', 'multipliers', 'jacobian']
* - `save_filename`
  - *str* (`"slsqp_recorder.hdf5"`)
  - Name of the saved file.
* - `load_filename`
  - *str* (`None`)
  - File to load the previous solution/iterates for warm/hot start. \
    If `None`, `load_filename` is assumed as the `save_filename`. \
    If `load_filename` is same as `save_filename`, \
    the newly generated file will be saved as: \
    ‘save_filename without extension’ + ‘_warm.hdf5’ or ‘_hot.hdf5’.
* - `warm_start`
  - *bool* (`False`)
  - Set to `True` to use the previous solution from the saved file \
    from the last optimization as the initial guess.
* - `hot_start`
  - *bool* (`False`)
  - If `True`, PySLSQP will use the saved objective, constraints, \
    gradient, and Jacobian values from the previous optimization \
    until the iterations reach the last saved iteration. \
    Note that this only works if `save_itr` for the previous \
    optimization was set to ‘all’. This is useful when the obj, \
    con, grad, and jac functions are expensive to compute and the \
    optimization process was interrupted during the prior run.
```