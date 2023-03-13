# Welcome to mod-opt documentation

![alt text](/images/lsdolab.png "Title displayed")

`mod-opt` is a modular development environment and library for optimization
algorithms.
`mod-opt` is primarily developed for easy and fast development of 
gradient-based optimization algorithms through modular construction of 
algorithms and testing with built-in problems and interfaces with test-suites.

Benchmarking with built-in standard algorithms or interfaced external optimizer
libraries are also possible.
Custom optimization algorithms can be developed in a modular fashion using the Optimizer() class in modOpt.
Custom optimization problems to be solved with any of the optimizers available can be written using the Problem() class in modOpt. Optimization problems written in csdl modeling language is also supported.

# Cite us
```none
@article{lsdo2023,
        Author = { Anugrah Jo Joshy, and John T. Hwang},
        Journal = {Name of the journal},
        Title = {A modular development environment and library for optimization
        algorithms},
        pages = {0123},
        year = {2023},
        issn = {0123-4567},
        doi = {https://doi.org/}
        }
```

<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

getting_started
tutorials
custom_1
custom_2
examples
api
autoapi/index
```
