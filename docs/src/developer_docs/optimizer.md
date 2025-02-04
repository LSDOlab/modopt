# `Optimizer`

All optimizers in modOpt must inherit from the abstract [**`Optimizer`**](../api/optimizer.md) class.
As a result, tools for checking first derivatives, visualizing, recording, and hot-starting
are automatically inherited by all optimizers in the library.
Every optimizer in modOpt is instantiated with a problem object, which should be derived
from the [**`Problem`**](../api/problem.md) or [**`ProblemLite`**](../api/problem_lite.md) classes. 
See the [UML class diagram](../developer_docs.md/#uml-class-diagram-for-modopt) for
a quick overview of the `Optimizer`, `Problem`, and `ProblemLite` classes.

The `Optimizer` base class provides essential tools for recording, 
visualization, and hot-starting an optimization.
The `record` attribute manages the systematic recording of the optimization, while the `visualizer`
attribute enables real-time visualization of the optimization process. 
The `Optimizer` base class also implements a `check_first_derivatives` method 
to verify the correctness of the user-defined derivatives in the provided problem object.

Subclasses of `Optimizer` must implement an `initialize` method that sets the `solver_name` and
declares any optimizer-specific options. 
Developers are required to define the `available_outputs` attribute within the `initialize` method. 
This attribute specifies the data that the optimizer will provide after each iteration 
of the algorithm by calling the `update_outputs` method. 
Developers must also define a `setup` method to handle any pre-processing of the problem data and
configuration of the optimizer’s modules.

The core of an optimizer in modOpt lies in the `solve` method. 
This method implements the numerical algorithm and iteratively calls 
the `'compute_'` methods from the problem object.
Upon completion of the optimization, the `solve` method should assign a `results` attribute 
that holds the optimization results in the form of a dictionary. 
Developers may optionally implement a `print_results` method to override 
the default implementation provided by the base class and
customize the presentation of the results.

Developers may need to implement additional methods for setting up constraints, 
their bounds, and derivatives, depending on their optimizer.

```{note}
Since HDF5 files from optimizer recording are incompatible with text editors, 
developers can provide users with the `readable_outputs` 
option during optimizer instantiation to export optimizer-generated
data as plain text files. 
For each variable listed in `readable_outputs`, a separate file is generated,
with rows representing optimizer iterations.
While creating a new optimizer, developers may declare this option so that
users are able to take advantage of this feature already implemented in
the `Optimizer` base class.
The list of variables allowed for `readable_outputs` is any
subset of the keys in the `available_outputs` attribute.
```