# SNOPT

SNOPT (Sparse Nonlinear OPTimizer) is a gradient-based optimization algorithm that uses a 
sequential quadratic programming (SQP) to solve general nonlinear programming problems.
It uses the BFGS algorithm to approximate the Hessian of the Lagrangian.

To use the `SNOPT` solver in modOpt, you need to have it installed on your machine along with its Python interface.
The binaries can be obtained from **[SNOPT's official website](https://ccom.ucsd.edu/~optimizers/downloads/)**,
and the Python interface from **[GitHub](https://github.com/snopt/snopt-python/tree/snopt-only)**.
Once these are installed correctly, you can import the `SNOPT` solver from modOpt as shown in the following code:

```py
from modopt import SNOPT
```

Options could be set by passing them within the `solver_options` dictionary when 
instantiating the SNOPT optimizer object as shown below.

```py
snopt_options = {
    'Major iterations': 100, 
    'Major optimality': 1e-9, 
    'Major feasibility': 1e-8
    }
optimizer = SNOPT(prob, solver_options=snopt_options)
```

## modOpt-specific solver options for SNOPT
The modOpt-SNOPT interface adds two more options in addition to 
the standard options available with the SNOPT solver.
These additional options are described in the table below.
```{list-table}  ModOpt-specific SNOPT solver options
:header-rows: 1
:name: snopt_modopt_options

* - Option
  - Type (default value)
  - Description
* - `append2file`
  - *bool* (`False`)
  - If `True`, new outputs will be appended to the previous \
    output files. Otherwise, new outputs will be written \
    to newly created files.
* - `continue_on_failure`
  - *bool* (`False`)
  - If `True`, during failed function calls, SNOPT will reduce \
    the step size and continue the optimization. Otherwise, \
    SNOPT exits optimization on function call failure.
```

## Standard solver options for SNOPT
The complete list of standard options, their types, and default values for SNOPT are shown in the table below.
For more details on the SNOPT optimization algorithm or its usage, see
**[SNOPT User's Guide](https://ccom.ucsd.edu/~optimizers/static/pdfs/sndoc7.pdf)**.

```{list-table} SNOPT solver options
:header-rows: 1
:name: snopt_options

* - Option
  - Type
  - Default value
* - `'Start type'`
  - *str*
  - `'Cold'`
* - `'Specs filename'`
  - *str*
  - `None`
* - `'Print filename'`
  - *str*
  - problem.name + `'_SNOPT_print.out'`
* - `'Print frequency'`
  - *int*
  - `None`
* - `'Print level'`
  - *int*
  - `None`
* - `'Summary'`
  - *str*
  - `'yes'`
* - `'Summary filename'`
  - *str*
  - problem.name + `'_SNOPT_summary.out'`
* - `'Summary frequency'`
  - *int*
  - `None`
* - `'Solution file'`
  - *int*
  - `None`
* - `'Solution filename'`
  - *str*
  - `'SNOPT_solution.out'`
* - `'Solution print'`
  - *bool*
  - `None`
* - `'Major print level'`
  - *int*
  - `None`
* - `'Minor print level'`
  - *int*
  - `None`
* - `'Sticky parameters'`
  - *int*
  - `None`
* - `'Suppress'`
  - *int*
  - `None`
* - `'Time limit'`
  - *float*
  - `None`
* - `'Timing level'`
  - *int*
  - `None`
* - `'System information'`
  - *int*
  - `None`
* - `'Verify level'`
  - *int*
  - `None`
* - `'Max memory attempts'`
  - *int*
  - `10`
* - `'Total character workspace'`
  - *int*
  - `None`
* - `'Total integer workspace'`
  - *int*
  - `None`
* - `'Total real workspace'`
  - *int*
  - `None`
* - `'Proximal point'`
  - *int*
  - `None`
* - `'Major feasibility'`
  - *float*
  - `None`
* - `'Major optimality'`
  - *float*
  - `None`
* - `'Minor feasibility'`
  - *float*
  - `None`
* - `'Minor optimality'`
  - *float*
  - `None`
* - `'Minor phase1'`
  - *float*
  - `None`
* - `'Feasibility tolerance'`
  - *float*
  - `None`
* - `'Optimality tolerance'`
  - *float*
  - `None`
* - `'Iteration limit'`
  - *int*
  - `None`
* - `'Major iterations'`
  - *int*
  - `None`
* - `'Minor iterations'`
  - *int*
  - `None`
* - `'CG tolerance'`
  - *float*
  - `None`
* - `'CG preconditioning'`
  - *int*
  - `None`
* - `'CG iterations'`
  - *int*
  - `None`
* - `'Crash option'`
  - *int*
  - `None`
* - `'Crash tolerance'`
  - *float*
  - `None`
* - `'Debug level'`
  - *int*
  - `None`
* - `'Derivative level'`
  - *int*
  - `None`
* - `'Derivative linesearch'`
  - *int*
  - `None`
* - `'Derivative option'`
  - *int*
  - `None`
* - `'Elastic objective'`
  - *int*
  - `None`
* - `'Elastic mode'`
  - *int*
  - `None`
* - `'Elastic weight'`
  - *float*
  - `None`
* - `'Elastic weightmax'`
  - *float*
  - `None`
* - `'Hessian frequency'`
  - *int*
  - `None`
* - `'Hessian flush'`
  - *int*
  - `None`
* - `'Hessian type'`
  - *int*
  - `None`
* - `'Hessian updates'`
  - *int*
  - `None`
* - `'Infinite bound'`
  - *float*
  - `1.0e+20`
* - `'Major step limit'`
  - *float*
  - `None`
* - `'Unbounded objective'`
  - *float*
  - `None`
* - `'Unbounded step'`
  - *float*
  - `None`
* - `'Linesearch tolerance'`
  - *float*
  - `None`
* - `'Linesearch debug'`
  - *int*
  - `None`
* - `'LU swap'`
  - *float*
  - `None`
* - `'LU factor tolerance'`
  - *float*
  - `None`
* - `'LU update tolerance'`
  - *float*
  - `None`
* - `'LU density'`
  - *float*
  - `None`
* - `'LU singularity'`
  - *float*
  - `None`
* - `'New superbasics'`
  - *int*
  - `None`
* - `'Partial pricing'`
  - *int*
  - `None`
* - `'Penalty parameter'`
  - *float*
  - `None`
* - `'Pivot tolerance'`
  - *float*
  - `None`
* - `'Reduced Hessian limit'`
  - *int*
  - `None`
* - `'Superbasics limit'`
  - *int*
  - `None`
* - `'Scale option'`
  - *int*
  - `None`
* - `'Scale tolerance'`
  - *float*
  - `None`
* - `'Scale print'`
  - *int*
  - `None`
* - `'Verbose'`
  - *bool*
  - `True`
```