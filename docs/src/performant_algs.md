# Performant optimization algorithms

Unlike the instructional algorithms, users have an easy, alternate way to optimize using
the performant algorithms.
This minimal API allows users to solve the optimization problem using a single line of code once the problem is defined.
A simple example is shown below.

```py
import numpy as np
import modopt as mo

x0 = np.array([50., 5.])
xl = np.array([0., -np.inf])
cl = np.array([1., 1.])
cu = np.array([1., np.inf])
c_scaler = np.array([10., 100.])
def obj(x):
    return np.sum(x**4)
def grad(x):    
    return 4 * x ** 3
def con(x):
    return np.array([x[0] + x[1], x[0] - x[1]])
def jac(x):
    return np.array([[1., 1], [1., -1]])
    
problem = mo.ProblemLite(
    x0, 
    obj=obj, 
    grad=grad, 
    con=con, 
    jac=jac, 
    cl=cl, 
    cu=cu,
    xl=xl,
    c_scaler = c_scaler,
    name='ineq_constrained_lite'
    )

results = mo.optimize(
    problem, 
    solver='IPOPT', 
    solver_options={'max_iter': 100, 'tol': 1e-6}
    )
```

The standard, but slightly less convenient way of optimizing the above problem would be as follows.
```py
optimizer = mo.IPOPT(
    problem, 
    solver_options={'max_iter': 100, 'tol': 1e-6}
    )
results = optimizer.solve()
```

Although slightly more verbose, using the optimizer classes as shown above can be more beneficial.
It allows access to additional optimizer information and provides more debugging options,
such as verifying the correctness of the user-provided first derivatives 
with `optimizer.check_first_derivatives(x=x0, step=1e-6)`.

Please visit the following pages for more information on any specific optimizer.

```{toctree}
:maxdepth: 1

performant_algs/slsqp
performant_algs/pyslsqp
performant_algs/sqp
performant_algs/snopt
performant_algs/ipopt
performant_algs/qpsolvers
```