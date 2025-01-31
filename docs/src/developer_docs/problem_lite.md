# `ProblemLite`

Many optimization problems, e.g., academic test problems designed to evaluate specific aspects
of optimization algorithms, are simple and do not require the advanced array management techniques 
provided by the `Problem` class.
For such problems, it is often more efficient for users to define the optimization functions 
in pure Python and provide these functions,
along with constants such as the initial guess, bounds, and scalers, directly to modOpt. 
Furthermore, some practitioners may prefer to use the optimizers in modOpt without involving the
`Problem` class, particularly if they have already modeled their optimization functions outside of modOpt.

In these scenarios, subclassing the `Problem` class can be excessive and result in unnecessarily verbose code. 
The `ProblemLite` class is designed to address this situation. Unlike `Problem`,
`ProblemLite` is a concrete class that can be instantiated directly, without requiring any subclassing. 
`ProblemLite` objects act as containers for optimization constants and functions defined externally, streamlining the process for users primarily focused on using the optimizers. 
Additionally, `ProblemLite` abstracts away the more advanced object-oriented programming concepts, 
making it more accessible to beginners.

Despite its simplicity, `ProblemLite` replicates the same interface as `Problem` when interacting
with optimizers. 
It supports recording and visualization, and can automatically generate derivatives using finite differences 
if the derivatives requested by the optimizer are not provided. 
It also facilitates lightweight interfaces to optimization functions implemented in other
modeling languages. 
For example, the `JaxProblem` and `CasadiProblem` in modOpt are implemented as subclasses of `ProblemLite`.

For the full list of optimization functions supported by `ProblemLite`,  refer to the 
[UML class diagram](../developer_docs.md/#uml-class-diagram-for-modopt).