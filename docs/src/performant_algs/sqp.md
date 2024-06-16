# SQP (Sequential Quadratic Programming)

While using the builtin SQP optimizer, you can follow the same process as in the previous section
except when importing the optimizer.

You need to import the optimizer as shown in the following code:

```py
from modopt import SQP
```

Options available are: `maxiter`, `opt_tol`, and `feas_tol`.
Options could be set by just passing them as kwargs when 
instantiating the SQP optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the optimality tolerance `opt_tol` shown below.

```py
optimizer = SQP(prob, maxiter=20, opt_tol=1e-8)
```