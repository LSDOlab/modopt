# Changelog

This page lists the key changes made between consecutive versions of modOpt.
A more detailed description of major changes between releases can be found
in `RELEASE_NOTES.md` in the project root directory.

***********************************
## Version 0.2.0 (April 22, 2026)

### Feature Additions
- Full support for NumPy 2.0
- Builtin `InteriorPoint` algorithm as an educational solver
- Builtin `OpenSQP` algorithm as a performant solver
- Support Hessian and multiplier initialization in `OpenSQP`
- Update CUTEst table with newest problems, tags, classifications
- Allow filtering CUTEst problems based on tags and return problem metadata
- Support adder in `CSDLAlphaProblem`
- Implement caching in `AugmentedLagrangian`

### Bug Fixes
- Rounding issue in performance profiling
- Issue with callable checks in solver_options
- Reference bug in `NewtonLagrange`, `MeritFunction` cache

### Miscellaneous
- Provide `out_dir` argument in Optimizer base class
- Include `return_status`, `success`, and `stats` in IPOPT results
- Update docs to use latest `sphinx-collections`
- Remove workflow for the old, github-hosted docs

***********************************
## Version 0.1.0 (February 3, 2025)

- Initial production release of modOpt