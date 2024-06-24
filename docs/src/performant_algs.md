# Performant optimization algorithms

Unlike the instructional algorithms, users have an easy, alternate way to optimize using
the performant algorithms.
This minimal API allows users to solve the optimization problem using a single line of code once the problem is defined.
A simple example is shown below.

```py
from modopt import ProblemLite, optimize

x0 = np.array([50., 5.])
cl = np.array([1.])
cu = np.array([np.inf])
def obj(x):
    return np.sum(x**4)
def grad(x):    
    return 4 * x ** 3
def con(x):
    return np.array([x[0] - x[1]])
def jac(x):
    return np.array([[1., -1]])
    
prob = ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, cl=cl, cu=cu, name='ineq_constrained_lite')

results = optimize(prob, solver='IPOPT', solver_options={'max_iter': 100, 'tol': 1e-6})
```

The standard, more involved but less convenient way of optimizing the above problem would be as follows.
```py
from modopt import IPOPT

optimizer = IPOPT(prob, solver_options={'max_iter': 100, 'tol': 1e-6})
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
```