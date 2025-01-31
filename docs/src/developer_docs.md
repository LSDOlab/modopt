# Developer Docs

This page provides guidance for developers to get started with advanced use cases of modOpt.

## Developing New Optimizers
If you are interested in using modOpt to implement new optimizers,
refer to the [New Optimizer Development](./optimizer_development.ipynb) section to get started.
For benchmarking your newly implemented optimizer, see the [Benchmarking](./benchmarking.ipynb) section.
The [Educational Algorithms](./educational_algs.md) section contains several references 
for building gradient-based, gradient-free, and discrete optimization algorithms.
Explore the [Examples](./examples.md) section for demonstration problems that show how to
utilize the built-in educational algorithms.

## Interfacing External Optimizers
All optimizers in modOpt must inherit from the abstract [**`Optimizer`**](./api/optimizer.md) class.
As a result, tools for checking first derivatives, visualizing, recording, and hot-starting
are automatically inherited by correctly interfaced external optimizers.
Every optimizer in modOpt is instantiated with a problem object, which should be derived
from the [**`Problem`**](./api/problem.md) or [**`ProblemLite`**](./api/problem_lite.md) classes. 
See the [UML class diagram](#uml-class-diagram-for-modopt) at the bottom of this page for
a quick overview of the `Optimizer`, `Problem`, and `ProblemLite` classes.

The `Optimizer` base class provides essential tools for recording, 
visualization, and hot-starting an optimization.
The `record` attribute manages the systematic recording of the optimization, while the `visualizer`
attribute enables real-time visualization of the optimization process. 
The `Optimizer` base class also implements a `check_first_derivatives` method 
to verify the correctness of the user-defined derivatives in the provided problem object.

Subclasses of `Optimizer` must implement an `initialize` method that sets the `solver_name` and
declares any optimizer-specific options. 
Developers are required to define the `available_outputs` attribute within the `initialize` method. 
This attribute specifies the data that the optimizer will provide after each iteration 
of the algorithm by calling the `update_outputs` method. 
Developers must also define a `setup` method to handle any pre-processing of the problem data and
configuration of the optimizer’s modules.

The core of an optimizer in modOpt lies in the `solve` method. 
This method implements the numerical algorithm and iteratively calls 
the `'compute_'` methods from the problem object.
Upon completion of the optimization, the `solve` method should assign a `results` attribute 
that holds the optimization results in the form of a dictionary. 
Developers may optionally implement a `print_results` method to override 
the default implementation provided by the base class and
customize the presentation of the results.

```{note}
Since HDF5 files from optimizer recording are incompatible with text editors, 
developers can provide users with the `readable_outputs` 
option during optimizer instantiation to export optimizer-generated
data as plain text files. 
For each variable listed in `readable_outputs`, a separate file is generated,
with rows representing optimizer iterations.
While creating a new optimizer, developers may declare this option so that
users are able to take advantage of this feature already implemented in
the `Optimizer` base class.
The list of variables allowed for `readable_outputs` is any
subset of the keys in the `available_outputs` attribute.
```

Developers may need to implement additional methods for setting up constraints, their bounds, and derivatives,
depending on the API of the external optimizer.
The [external_libraries](https://github.com/LSDOlab/modopt/tree/main/modopt/external_libraries) directory 
contains a variety of references for interfacing external optimizers with modOpt. 
These include QP solvers, convex optimizers, and general gradient-free and gradient-based nonlinear optimizers.
For benchmarking a newly interfaced external optimizer against existing optimizers in the library, 
see the [Benchmarking](./benchmarking.ipynb) section.

## Interfacing External Models or Modeling Frameworks

Lightweight interfaces to external models or modeling frameworks can be
built using the [**`ProblemLite`**](./api/problem_lite.md) class.
For a basic demonstration and usage, refer to the [**`ProblemLite`**](./modeling/problem_lite.ipynb) section.
For more information on the class, see the [**`ProblemLite`**](./developer_docs/problem_lite.md) page for developers.
For references on modeling interfaces built using `ProblemLite`, check out
[JaxProblem](https://github.com/LSDOlab/modopt/blob/main/modopt/external_libraries/jax/jax_problem.py)
and [CasADiProblem](https://github.com/LSDOlab/modopt/blob/main/modopt/external_libraries/casadi/casadi_problem.py).

Developers can use the [**`Problem`**](./api/problem.md) class 
to build more intrusive and flexible modeling interfaces.
For a basic demonstration and usage, refer to the [**`Problem`**](./modeling/problem.ipynb) section.
For more information on the class, see the [**`Problem`**](./developer_docs/problem.md) page for developers.
For references on modeling interfaces built using `Problem`, check out
[CSDLProblem](https://github.com/LSDOlab/modopt/blob/main/modopt/external_libraries/csdl/csdl_problem.py),
[CSDLAlphaProblem](https://github.com/LSDOlab/modopt/blob/main/modopt/external_libraries/csdl/csdl_alpha_problem.py),
and [OpenMDAOProblem](https://github.com/LSDOlab/modopt/blob/main/modopt/external_libraries/openmdao/openmdao_problem.py).

## Interfacing Test Problem Sets
Test problem collections can be interfaced with modOpt using the `Problem` or `ProblemLite` classes,
following the same methodology discussed in the section 
[Interfacing External Models or Modeling Frameworks](#interfacing-external-models-or-modeling-frameworks) above. 
For reference, check out 
[CUTEstProblem](https://github.com/LSDOlab/modopt/blob/main/modopt/external_libraries/pycutest/cutest_problem.py),
an interface to the CUTEst test problem collection built using the `Problem` class.

## UML class diagram for modOpt

```{figure} /src/images/modopt_full_uml.png
:figwidth: 100 %
:align: center
:alt: modopt_full_uml

_**UML class diagram for modOpt**_
```

```{toctree}
:hidden:

developer_docs/problem
developer_docs/problem_lite
developer_docs/optimizer
```


<!-- ## Developer API Reference
This section provides the developer API reference for modOpt.

### modopt.Optimizer

```{eval-rst}

.. autoclass:: modopt.Optimizer
    :members: __init__, check_first_derivatives, solve, print_results, update_outputs, run_postprocessing

```
### modopt.ProblemLite

```{eval-rst}

.. autoclass:: modopt.ProblemLite
    :members: __init__

```

### modopt.Problem

```{eval-rst}

.. autoclass:: modopt.Problem
    :members: __init__, __str__, initialize, setup, setup_derivatives, 
              add_design_variables, add_objective, add_constraints,
              declare_lagrangian, declare_objective_gradient, declare_lagrangian_gradient,
              declare_constraint_jacobian, declare_constraint_jvp, declare_constraint_vjp,
              declare_objective_hessian, declare_lagrangian_hessian, 
              declare_objective_hvp, declare_lagrangian_hvp,
              compute_objective, compute_constraints, compute_lagrangian,
              compute_objective_gradient, compute_lagrangian_gradient, compute_constraint_jacobian,
              compute_constraint_jvp, compute_constraint_vjp, 
              compute_objective_hessian, compute_lagrangian_hessian,
              compute_objective_hvp, compute_lagrangian_hvp,
              use_finite_differencing
    :private-members: _compute_objective, _compute_constraints, _compute_lagrangian,
                      _compute_objective_gradient, _compute_lagrangian_gradient, 
                      _compute_constraint_jacobian,
                      _compute_constraint_jvp, _compute_constraint_vjp,
                      _compute_objective_hessian, _compute_lagrangian_hessian,
                      _compute_objective_hvp, _compute_lagrangian_hvp
``` -->