# Welcome to modOpt

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

## modOpt as a library of optimization algorithms

modOpt allows optimization practitioners to define the computational models that provide
the objective, constraints, and the derivatives for their optimization problems 
using one of the following three options:
1. the built-in `Problem()` class,
2. **Computational System Design Language** (CSDL), or
3. **OpenMDAO** modeling framework

Once the model is defined, it needs to be optimized. 
The users can pick an optimizer of their choice from the
library of optimization algorithms available in modOpt.

<!-- ![modopt_lib](/src/images/modopt_lib.png "modOpt as a library") -->
```{figure} /src/images/modopt_lib.png
:figwidth: 80 %
:align: center
:alt: modopt_lib

_**modOpt as a library**_
```

## modOpt as a development environment for optimization algorithms

There are a number of transparent modules available within the package
which users can leverage to develop new or modified optimization algorithms.
Custom optimization algorithms can be developed in a modular fashion 
using the `Optimizer()` class in modOpt.
Benchmarking against built-in standard algorithms or interfaced external optimizers
are also possible.

<!-- ![modopt_env](/src/images/modopt_env.png "modOpt as a development environment") -->
<!-- <img src="/images/modopt_env.png" alt='modopt_env' title="modOpt as a development environment" width="150" height="100"/> -->
<!-- <p align="center"> -->
<!-- <img src="images/modopt_env.png" alt='modopt_env'> -->
<!-- </p> -->
```{figure} /src/images/modopt_env.png
:figwidth: 80 %
:align: center
:alt: modopt_lib

_**modOpt as a development environment**_
```

## Getting Started
To install and start using modOpt, please read the [Getting Started](src/getting_started.md) page.

## Citation
If you use modOpt in your work, please use the following reference for citation:

```bibtex
@article{joshy2024modopt,
  title={modOpt: A modular development environment and library for optimization algorithms},
  author={Joshy, Anugrah Jo and Hwang, John T},
  journal={arXiv preprint arXiv:2410.12942},
  year={2024},
  doi={10.48550/arXiv.2410.12942}
}
```

<!-- ## References

```{bibliography} src/references.bib
``` -->

<!-- Remove/add custom pages from/to toc as per your package's requirement -->
<!-- src/basic -->

```{toctree}
:maxdepth: 2
:caption: Contents

src/getting_started
src/modeling
src/optimizers
src/benchmarking
src/postprocessing
src/tutorials
src/examples
src/api
src/contributing
src/changelog
src/license
```

```{toctree}
:maxdepth: 2
:caption: Performant algorithms
:hidden:

src/performant_algs/slsqp
src/performant_algs/pyslsqp
src/performant_algs/cobyla
src/performant_algs/bfgs
src/performant_algs/lbfgsb
src/performant_algs/nelder_mead
src/performant_algs/cobyqa
src/performant_algs/trust_constr
src/performant_algs/sqp
src/performant_algs/snopt
src/performant_algs/ipopt
src/performant_algs/qpsolvers
src/performant_algs/cvxopt
```

```{toctree}
:maxdepth: 2
:caption: Educational algorithms
:hidden:

src/educational_algs/steepest_descent
src/educational_algs/newton
src/educational_algs/quasi_newton

src/educational_algs/newton_lagrange
src/educational_algs/l2_penalty_eq

src/educational_algs/nelder_mead_simplex
src/educational_algs/pso
src/educational_algs/simulated_annealing

src/educational_algs/std_algs
```