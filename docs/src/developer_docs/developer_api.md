# Developer API Reference
This section provides the developer API reference for modOpt.

## modopt.Optimizer

<!-- ```{eval-rst}

.. autoclass:: modopt.Optimizer
    :members: __init__, check_first_derivatives, solve, print_results, update_outputs, run_postprocessing

``` -->
## modopt.ProblemLite

<!-- ```{eval-rst}

.. autoclass:: modopt.ProblemLite
    :members: __init__

``` -->

## modopt.Problem

<!-- ```{eval-rst}

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