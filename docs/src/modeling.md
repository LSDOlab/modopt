# Modeling

modOpt allows great flexibility in modeling the optimization functions,
i.e., the objective and constraint functions, and their derivatives.
<!-- There are mainly six different ways in which models can be built. -->
Models can primarily be built using one of six classes: 
`ProblemLite`, `Problem`, `CSDLAlphaProblem`, `OpenMDAOProblem`, 
`JaxProblem`, or `CasADiProblem`.
Note that the support for the older version of CSDL through `CSDLProblem` 
is now deprecated.

If you are deciding which of these to use for your problem, here are some
general rules of thumb.

1. If your problem functions and derivatives are already written in some 
modeling language other than Jax, OpenMDAO, CasADi, or CSDL,
you might want to use `Problem` or `ProblemLite`.
`ProblemLite` provides a simple and straightforward way to wrap 
the problem functions and their derivatives for use with modOpt,
while `Problem` offers greater flexibility for interfacing with external functions.
`ProblemLite` is more beginner-friendly, as it abstracts away the
object-oriented programming required with the `Problem` class.
2. If your problem functions and derivatives can be written as simple equations
in Python, then `Problem` or `ProblemLite` might be good options.
Both can handle sparsity in the derivative matrices but `Problem` offers
more advanced array management techniques.
<!-- and the problem size (number of variables and constraints) is not too large 
to benefit from sparse matrix formats -->
<!-- However, if the problem can benefit from sparsity in the  -->
3. If your problem functions consist of a long sequence of explicit equations
(potentially involving linear solvers), hand-derivation of 
the problem derivatives become challenging.
For such problems, Jax and CasADi are excellent options.
CasADi supports sparse matrices and includes built-in solvers for disciplines 
such as optimal control, while Jax is generally more efficient
in memory and time, and supports GPU acceleration.
4. Lastly, if your problem involves very large and complex numerical models
involving multiple disciplines and nonlinear solvers, then you might want to
consider CSDL or OpenMDAO as it is designed to facilitate modeling of 
complex systems like an aircraft.
OpenMDAO requires the user to provide partial derivatives for individual components
in the model, while CSDL fully automates the derivative computation.
CSDL, however, tends to be less memory-efficient for large models involving *for loops*.

In any of the above cases, if the problem functions are not 
expensive to evaluate and the problem size is small,
numerical approximation of the derivatives using finite-difference 
or complex step methods may work well.
However, note that round-off errors can sometimes affect the optimization
convergence.
Numerical derivatives are rarely necessary, as CSDL, CasADi, and Jax already provide exact derivatives
through automatic differentiation, which users can access without any additional
effort.

Please visit the following pages for more guidance on using any
of the six modeling options discussed above.

```{toctree}
:caption: Modeling options
:maxdepth: 1

modeling/problem_lite
modeling/problem
modeling/csdl
modeling/csdl_alpha
modeling/openmdao
modeling/jax
modeling/casadi
```