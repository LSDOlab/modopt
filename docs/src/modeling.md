# Modeling

modOpt offers great flexibility in modeling the optimization functions
(objective and constraint functions), and their derivatives.
<!-- There are mainly six different ways in which models can be built. -->
Models can primarily be built using one of six classes: 
`ProblemLite`, `Problem`, `CSDLAlphaProblem`, `OpenMDAOProblem`, 
`JaxProblem`, or `CasADiProblem`.
The latter four classes serve as interfaces to the
modeling languages CSDL_alpha, OpenMDAO, Jax, and CasADi, respectively.
Note that support for the older version of CSDL via `CSDLProblem` 
is now deprecated.

If you are deciding which of these to use for your problem, here are some
general rules of thumb.

1. If your problem functions and derivatives are already written in some 
modeling language other than Jax, OpenMDAO, CasADi, or CSDL,
you might consider using `Problem` or `ProblemLite`.
`ProblemLite` provides a simple and straightforward way to wrap 
the problem functions and their derivatives for use within modOpt,
while `Problem` provides greater flexibility for interfacing with external functions.
`ProblemLite` is more beginner-friendly, as it abstracts away the
object-oriented programming required with the `Problem` class.

2. If your problem functions and derivatives can be written as simple equations
in Python, then `Problem` and `ProblemLite` are good options.
Both can handle sparsity in the derivative matrices, but `Problem` offers
more advanced array management techniques.
<!-- and the problem size (number of variables and constraints) is not too large 
to benefit from sparse matrix formats -->
<!-- However, if the problem can benefit from sparsity in the  -->

3. If your problem functions consist of a long sequence of explicit equations
(potentially involving linear solvers), hand-deriving derivatives 
can become challenging.
In such cases, Jax and CasADi are excellent options.
CasADi supports sparse matrices and includes built-in solvers for disciplines 
such as optimal control.
Jax is generally more efficient in memory and time and supports GPU acceleration.

4. Lastly, if your problem involves very large and complex numerical models
spanning multiple disciplines and utilizing nonlinear solvers,
consider using CSDL or OpenMDAO, as they are designed to facilitate modeling of 
large and complex systems, such as aircraft.
OpenMDAO requires users to provide partial derivatives for individual components
in the model, whereas CSDL fully automates derivative computation.
However, CSDL tends to be less memory-efficient for large models involving *for loops*.

```{note}
For all six modeling options, if the problem functions are not 
expensive to evaluate and the problem size is small,
numerical approximation of the derivatives using finite-difference 
or complex step methods may work well.
However, note that round-off errors can sometimes affect 
optimization convergence.
Numerical derivatives are rarely necessary, 
as CSDL, CasADi, and Jax can provide exact derivatives
through automatic differentiation, which users can access 
and utilize without any additional effort.
```

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