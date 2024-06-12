# SLSQP

While using scipy library you can follow the same process for other optimizers
except when importing the optimizer.

You need to import the optimizer as shown in the following code 
(here we use the SLSQP optimizer):

```py
from modopt import SLSQP
```

Options are available
**[here](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp)**.
Options could be set by just passing them as kwargs when 
instantiating the SLSQP optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the precision goal `ftol` for the objective as shown below.

```py
optimizer = SLSQP(prob, maxiter=20, ftol=1e-6)
```