# PySLSQP

Make sure to have pyslsqp installed on your machine.
You can use `pip install pyslsqp` to install from PyPI.
To use pyslsqp, you can follow the same process for other optimizers
except when importing the optimizer.

You need to import the optimizer as shown in the following code:

```py
from modopt import PySLSQP
```

Options are available
**[here](https://pyslsqp.readthedocs.io/en/latest/src/api_pages/optimize.html#pyslsqp.optimize)**.
Options could be set by just passing them within a `solver_options` dictionary  when 
instantiating the PySLSQP optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the accuracy goal `acc` for the final solution as shown below.

```py
optimizer = PySLSQP(prob, solver_options={'maxiter': 20, 'acc': 1e-6})
```