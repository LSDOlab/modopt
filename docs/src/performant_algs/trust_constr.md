# TrustConstr

`TrustConstr` is a gradient-based optimization algorithm that uses a trust-region 
interior point method or an equality-constrained sequential quadratic programming (SQP) 
method to solve a problem depending on whether the problem has inequality constraints or not.
It can utilize second-order derivative information in the form of the Hessian of the objective 
for unconstrained problems or the Hessian of the Lagrangian for constrained problems.
`TrustConstr` can also use objective Hessian-vector products when the 
objective Hessian is unavailable.
This solver uses the 'trust-constr' algorithm from the Scipy library.

To use the `TrustConstr` solver in modOpt, start by importing it as shown in the following code:

```py
from modopt import TrustConstr
```

Options could be set by just passing them within the `solver_options` dictionary when 
instantiating the `TrustConstr` optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the tolerance on the barrier parameter `barrier_tol` as shown below.

```py
optimizer = TrustConstr(prob, solver_options={'maxiter':100, 'barrier_tol':1e-8})
```

The options available for the `TrustConstr` solver in modOpt as given in the following table.
For more information on the Scipy 'trust-constr' algorithm, visit
**[Scipy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html)**.

```{list-table} TrustConstr solver options
:header-rows: 1
:name: trust_constr_options

* - Option
  - Type \
    (default)
  - Description
* - `maxiter`
  - *int* \
    (`500`)
  - Maximum number of iterations.
* - `gtol`
  - *float* \
    (`1e-8`)
  - Terminate successfully when both the infinity norm (max abs value) of \
    the Lagrangian gradient and the constraint violation are less than `gtol`.
* - `xtol`
  - *float* \
    (`1e-8`)
  - Terminate successfully when `tr_radius < xtol`, where `tr_radius` \
    is the trust region radius.
* - `barrier_tol`
  - *float* \
    (`1e-8`)
  - When inequality constraints are present, the algorithm will terminate \
    only when the barrier parameter decays below `barrier_tol`. 
* - `initial_tr_radius`
  - *float* \
    (`1.`)
  - Initial trust radius. \
    The trust radius is automatically updated throughout the optimization, \
    with `initial_tr_radius` being its initial value.
    <!-- The trust radius gives the maximum distance between \
    solution points in consecutive iterations. It reflects the trust the algorithm \
    puts in the local approximation of the optimization problem. \
    For an accurate local approximation the trust-region should be large, and \
    for an approximation valid only close to the current point it should be a small one. \ -->
* - `initial_constr_penalty`
  - *float* \
    (`1.`)
  - Initial constraints penalty parameter for computing the merit function.
    <!-- The penalty parameter is used for defining the merit function: \
    `M(x) = fun(x) + constr_penalty * constr_norm_l2(x)`, \
    where `constr_norm_l2(x)` is the l2 norm of the constraint values. \ -->
    The penalty parameter is automatically updated during the optimization, \
    with `initial_constr_penalty` being its initial value.
    <!-- The penalty parameter is used for balancing \
    the requirements of decreasing the objective function and satisfying the constraints. \ -->  
    <!-- The merit function is used for accepting or rejecting trial points and `constr_penalty` \
    weights the two conflicting goals of reducing objective function and constraints. \ -->
* - `initial_barrier_parameter`
  - *float* \
    (`0.1`)
  - Initial barrier parameter for the barrier subproblem. \
    Used only when inequality constraints are present.
    <!-- For addressing problems `min_x f(x)` subject to `c(x) <= 0`, \
    the algorithm introduces slack variables, solving the problem \
    `min_(x,s) f(x) + barrier_parameter*sum(ln(s))` subject to `c(x) + s = 0`. \ -->
    The subproblem is solved for decreasing values of `barrier_parameter` \
    and with decreasing tolerances for the subproblem termination. 
    <!-- starting with `initial_barrier_parameter` for the barrier parameter \
    and `initial_barrier_tolerance` for the barrier tolerance. \ -->
    Both `barrier_parameter` and `barrier_tolerance` are updated \
    with the same prefactor.
* - `initial_barrier_tolerance`
  - *float* \
    (`0.1`)
  - Initial tolerance for the barrier subproblem termination. \
    See the description of `initial_barrier_parameter` above for more details.
* - `factorization_method`
  - *str* \
    (`None`)
  - Method to be used for factorizing the constraint Jacobian. \
    Use `None` (default) for the auto selection or one of:
    * `'NormalEquation'` (requires scikit-sparse), 
    * `'AugmentedSystem'`, 
    * `'QRFactorization'`, or 
    * `'SVDFactorization'`.

    'NormalEquation' and 'AugmentedSystem' can be used only with sparse constraints. \
    The projections required by the algorithm will be computed using, respectively, \
    the normal equation and the augmented system approaches. \
    'NormalEquation' computes the Cholesky factorization of `A@A.T` and \
    'AugmentedSystem' performs the LU factorization of an augmented system. \
    They usually provide similar results. 'AugmentedSystem' is used by default for sparse matrices. \
    \
    'QRFactorization' and 'SVDFactorization' can be used only with dense constraints. \
    They compute the required projections using, respectively, QR and SVD factorizations. \
    'SVDFactorization' can cope with Jacobians with deficient row rank and \
    will be used whenever other factorization methods fail \
    (which may imply the conversion of sparse matrices to a dense format when required). \
    By default, 'QRFactorization' is used for dense matrices.
* - `sparse_jacobian`
  - *bool* \
    (`None`)
  - If `True`, represent the constraint Jacobian as sparse. 

    If `False`, represent the Jacobian as dense. \
    If `None` (default), the Jacobian won't be converted (will use user-provided format).
* - `ignore_exact_hessian`
  - *bool* \
    (`False`)
  - If `True`, the algorithm will ignore exact Hessians even if it is available, \
    and use only gradient information to approximate the Hessian. \
    If `False` (default), the algorithm will use objective/Lagrangian Hessian for \
    unconstrained/constrained problems whenever available.
* - `verbose`
  - *int* \
    (`0`)
  - Verbosity of the console output. \
    `0` suppresses all console outputs. \
    `1` displays a termination report. \
    `2` also displays progress during iterations in the form of a table. \
    `3` displays progress during iterations with more columns in the table.
* - `callback`
  - *callable* \
    (`None`)
  - Function to be called after each iteration. \
    The function is called as `callback(intermediate_result)`, \
    where `intermediate_result` is a dictionary containing values \
    for different variables from the current iteration. \
    The values correspond to the following keys: \
    '*x*', '*optimality*', '*constr_violation*', '*fun*', '*grad*', '*lagrangian_grad*',\
    '*constr*', '*jac*', '*v*', '*nit*', '*cg_niter*', '*nfev*', '*njev*', '*nhev*', \
    '*constr_nfev*', '*constr_njev*', '*constr_nhev*', '*tr_radius*', '*constr_penalty*', \
    '*barrier_parameter*', '*barrier_tolerance*', '*cg_stop_cond*', and '*execution_time*'. \
    See [scipy documentation](https://docs.scipy.org/doc/scipy-1.14.0/reference/optimize.minimize-trustconstr.html)
    for a description of each of the keys.
```