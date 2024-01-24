# SNOPT

While using SNOPT library you can follow the same process for other optimizers
except when importing the optimizer.

You need to import the optimizer as shown in the following code:

```py
from modopt import SNOPT
```

Options are available
**[here](https://github.com/LSDOlab/modopt/blob/main/modopt/external_libraries/snopt/snopt_optimizer.py#L22)**.
Options could be set by just passing them as kwargs when 
instantiating the SNOPT optimizer object as shown below.

```py
optimizer = SNOPT(  prob, 
                    Major_iterations = 100, 
                    Major_optimality=1e-9, 
                    Major_feasibility=1e-8)
```