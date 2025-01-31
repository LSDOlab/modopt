# `Problem`

The `Problem` class encapsulates the optimization problem definition and 
is designed as an abstract class. 
Users formulate their specific problem by subclassing `Problem`. 
The attributes and methods accessed by the optimizer are automatically generated 
based on how the user defines the subclass.
The subclass definition begins with the `initialize` method, which assigns a name to the
problem and declares any model-specific options. 
The `setup` method follows, adding the objective, design variables, and constraints 
using the add objective, `add_design_variables`, and `add_constraints` methods, respectively. 
Note that within modOpt, ‘design variables’ are synonymous with ‘optimization variables’. 
Users can also specify scaling for all variables and functions, 
as well as lower and upper bounds for variables and constraints, while adding them to the problem.

For gradient-based optimizers, users must define the `setup_derivatives` method to declare the
derivatives they will provide later. 
Depending on the problem and optimizer, these derivatives may
include objective gradients, constraint Jacobians, or objective/Lagrangian Hessians. 
Optimizers that can directly use the Lagrangian and its derivatives benefit from users 
declaring these as well. 
Similarly, users can also declare matrix-vector products (Jacobian-vector products (JVPs),
vector-Jacobian products (VJPs), objective/Lagrangian Hessian-vector products (HVPs)) when
using optimizers that can leverage them.

For a problem with distinct design variables and constraints, these quantities can be declared separately. 
They also require separate derivative declarations. 
The `Problem` class employs efficient
array management techniques to handle sparsity in constraint Jacobians and Hessians, allowing
for computational and memory savings when sub-Jacobians or sub-Hessians with different sparsity
patterns are declared individually.

Once all functions are declared, users must define the methods to compute these functions, even
if they are constants. 
This requirement ensures consistency and prevents users from inadvertently
omitting necessary definitions. 
For example, if an objective and its gradient are declared, the user
must define the compute objective and compute objective gradient methods. 
Method names are kept verbose instead of using abbreviations, to be maximally explicit and to avoid ambiguities.
The full list of optimization functions supported by the `Problem` class is listed in the 
[UML class diagram](../developer_docs.md/#uml-class-diagram-for-modopt).
If certain derivatives are challenging to formulate, users can invoke the `use_finite_differencing` method
within the corresponding `‘compute_’` method to automatically apply a first-order finite difference
approximation.

When instantiated, a subclass object generates the necessary attributes for the optimizer, such
as the initial guess vector, scaling vectors, and bound vectors corresponding to the concatenated
vectors for the design variables and constraints. 
The `Problem` class also includes methods that wrap the user-defined `‘compute_’` methods 
for the optimizer, applying the necessary scaling beforehand. 
These wrapped `‘compute_’` methods are prefixed with an underscore. 
When recording and visualizing, these wrapper methods additionally update the record and the `Visualizer` objects
with the latest values from each `‘compute_’` call.

Strong interfaces to various modeling languages and test-suite problems are established through
subclasses of the `Problem` class. 
For instance, `OpenMDAOProblem`, `CSDLProblem`, and `CUTEstProblem` inherit from `Problem`. 
These subclasses serve as wrappers that transform the models written in
their respective languages into models suitable for the optimizers in modOpt.