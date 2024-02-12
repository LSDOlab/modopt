# Welcome to modopt documentation!

ModOpt is a modular development environment and library for optimization
algorithms written fully in Python.
ModOpt is primarily developed for easy and fast development of 
gradient-based optimization algorithms through modular construction of 
algorithms, and testing with built-in problems and interfaced test-suites.
Since modOpt breaks down algorithms into self-contained components such as
line-searches and Hessian approximations, students can develop new or modified versions 
of existing algorithms by replacing or modifying these components.
It also enables students to perform comparative studies of their versions with a 
standard algorithm.

### Modopt as a library of optimization algorithms

Modopt allows optimization practitioners to define the computational models that provide
the objective, constraints, and the derivatives for their optimization problems 
using one of the following three options:
1. the built-in `Problem()` class,
2. **Computational System Design Language** (CSDL), or
3. **OpenMDAO** modeling framework

Once the model is defined, it needs to be optimized. 
The users can pick an optimizer of their choice from the
library of optimization algorithms available in modOpt.

<!-- ![modopt_lib](/src/images/modopt_lib.png "Modopt as a library") -->
```{figure} /src/images/modopt_lib.png
:figwidth: 80 %
:align: center
:alt: modopt_lib

_**Modopt as a library**_
```

### Modopt as a development environment for optimization algorithms

There are a number of transparent modules available within the package
which users can leverage to develop new or modified optimization algorithms.
Custom optimization algorithms can be developed in a modular fashion 
using the `Optimizer()` class in modOpt.
Benchmarking against built-in standard algorithms or interfaced external optimizers
are also possible.

<!-- ![modopt_env](/src/images/modopt_env.png "Modopt as a development environment") -->
<!-- <img src="/images/modopt_env.png" alt='modopt_env' title="Modopt as a development environment" width="150" height="100"/> -->
<!-- <p align="center"> -->
<!-- <img src="images/modopt_env.png" alt='modopt_env'> -->
<!-- </p> -->
```{figure} /src/images/modopt_env.png
:figwidth: 80 %
:align: center
:alt: modopt_lib

_**Modopt as a development environment**_
```


<!-- # Cite us
```none
@article{lsdo2023,
        Author = { Anugrah Jo Joshy, and John T. Hwang},
        Journal = {Name of the journal},
        Title = {A modular development environment and library for optimization
        algorithms},
        pages = {0123},
        year = {2024},
        issn = {0123-4567},
        doi = {https://doi.org/}
        }
``` -->

<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/tutorials
src/csdl
src/opt_algs
src/examples
src/contributing
src/changelog
src/api
```
