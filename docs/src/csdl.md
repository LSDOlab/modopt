# Solving csdl problems

## Define a problem in csdl

This example does not intend to cover the features of csdl.
For more details and tutorials on csdl, please refer to **[csdl documentation](https://lsdolab.github.io/csdl/)**.
In this example, we solve a constrained problem given by

$$
\underset{x_1, x_2 \in \mathbb{R}}{\text{minimize}} \quad x_1^2 + x_2^2

\newline
\text{subject to} \quad x_1 \geq 0
\newline
\quad \quad \quad \quad x_1 + x_2 = 1
\newline
\quad \quad \quad \quad x_1 - x_2 \geq 1
$$

We know the solution of this problem is $x_1=1$, and $x_2=0$.
However, we start from an intial guess of $x_1=0$, and $x_2=0.0$ for the purposes of this tutorial.

The problem model is written in csdl as follows:

```py
from csdl import Model

# minimize x^2 + y^2 subject to x>=0, x+y=1, x-y>=1.

class QuadraticFunc(Model):
    def initialize(self):
        pass

    def define(self):
        # add_inputs
        x = self.create_input('x', val=1.)
        y = self.create_input('y', val=1.)

        z = x**2 + y**2

        # add_outputs
        self.register_output('z', z)

        constraint_1 = x + y
        constraint_2 = x - y
        self.register_output('constraint_1', constraint_1)
        self.register_output('constraint_2', constraint_2)

        # define optimization problem
        self.add_design_variable('x', lower=0.)
        self.add_design_variable('y')
        self.add_objective('z')
        self.add_constraint('constraint_1', equals=1.)
        self.add_constraint('constraint_2', lower=1.)
```

Once your model is setup in csdl, create a `Simulator` object in csdl and 
translate the `Simulator` object to a `CSDLProblem` object in modOpt.

```py
from csdl_om import Simulator

# Create a Simulator object for your model
sim = Simulator(QuadraticFunc())

from modopt import CSDLProblem

# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLProblem(
    problem_name='quadratic',
    simulator=sim,
)
```

Once your problem is translated to modOpt, import your preferred optimizer from
the respective library in modOpt and solve it, following the standard procedure.
Here we will use the `SLSQP` optimizer from the scipy library.

```py
from modopt import SLSQP

# Setup your preferred optimizer (SLSQP) with the Problem object 
# Pass in the options for your chosen optimizer
optimizer = SLSQP(prob, solver_options={'maxiter':20})

# Check first derivatives at the initial guess, if needed
optimizer.check_first_derivatives(prob.x0)

# Solve your optimization problem
optimizer.solve()

# Print results of optimization
optimizer.print_results()
```


## Recommended optimizers for csdl problems

### 1 . SLSQP

Import the SLSQP optimizer as shown below:

```py
from modopt import SLSQP
```

Options available can be found from the 
**[scipy docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp)**.
Options could be set by just passing them as kwargs when 
instantiating the SLSQP optimizer object.
For example, we can set the maximum number of iterations `maxiter` 
and the precision goal `ftol` for the objective as shown below.

```py
optimizer = SLSQP(prob, solver_options={'maxiter':20, 'ftol':1e-6})
```

### 2. SQP

Import the SQP optimizer as shown below:

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


### 3. SNOPT

Import the SNOPT optimizer as shown below:

```py
from modopt import SNOPT
```
Options could be set by just passing them as kwargs when 
instantiating the SNOPT optimizer object.

```py
snopt_options = {
    'Major iterations': 100, 
    'Major optimality': 1e-9, 
    'Major feasibility': 1e-8
    }
optimizer = SNOPT(prob, solver_options=snopt_options)
```
