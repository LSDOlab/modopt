import time

import numpy as np

from modopt import Optimizer
from modopt.utils.options_dictionary import OptionsDictionary


class Egor(Optimizer):
    """
    Class that interfaces modOpt with the ``egobox`` package's Egor optimizer.
    Egor is a gradient-free efficient global optimization solver that requires
    finite bounds on every design variable and supports nonlinear inequality
    constraints through functions of the form ``c(x) <= 0``.

    Parameters
    ----------
    problem : Problem or ProblemLite
        Object containing the problem to be solved.
    recording : bool, default=False
        If ``True``, record all outputs from the optimization.
    out_dir : str, optional
        The directory to store all the output files generated from the optimization.
    hot_start_from : str, optional
        The record file from which to hot-start the optimization.
    hot_start_atol : float, default=0.
        The absolute tolerance check for the inputs when reusing outputs from
        the hot-start record.
    hot_start_rtol : float, default=0.
        The relative tolerance check for the inputs when reusing outputs from
        the hot-start record.
    visualize : list, default=[]
        The list of scalar variables to visualize during the optimization.
    keep_viz_open : bool, default=False
        If ``True``, keep the visualization window open after the optimization is complete.
    turn_off_outputs : bool, default=False
        If ``True``, prevent modOpt from generating any output files.

    solver_options : dict, default={}
        Dictionary containing the options to be passed to the Egor solver.
        Supported options are ``'max_iters'``, ``'gp_config'``, ``'n_start'``,
        ``'n_doe'``, ``'doe'``, ``'infill_strategy'``, ``'cstr_infill'``,
        ``'cstr_strategy'``, ``'qei_config'``, ``'infill_optimizer'``,
        ``'trego'``, ``'coego_n_coop'``, ``'target'``, ``'outdir'``,
        ``'warm_start'``, ``'hot_start'``, ``'failsafe_strategy'``,
        ``'seed'``, and ``'cstr_tol'``.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are ``'x'`` and ``'obj'``.
    """

    def initialize(self):
        try:
            import egobox as egx
        except ImportError as exc:
            raise ImportError(
                "'egobox' could not be imported. Install egobox using 'pip install egobox' "
                "or 'pip install modopt[egobox]' for using Egor optimizer."
            ) from exc

        self.egx = egx
        self.solver_name = "egobox-egor"
        self.options.declare("solver_options", types=dict, default={})

        self.default_solver_options = {
            "max_iters": (int, 20),
            "gp_config": (object, egx.GpConfig()),
            "n_start": (int, 20),
            "n_doe": (int, 0),
            "doe": ((type(None), list, tuple, np.ndarray), None),
            "infill_strategy": (object, egx.InfillStrategy.LOG_EI),
            "cstr_infill": (bool, False),
            "cstr_strategy": (object, egx.ConstraintStrategy.MC),
            "qei_config": (object, egx.QEiConfig()),
            "infill_optimizer": (object, egx.InfillOptimizer.COBYLA),
            "trego": (object, None),
            "coego_n_coop": (int, 0),
            "target": (float, -np.finfo(float).max),
            "outdir": ((type(None), str), None),
            "warm_start": (bool, False),
            "hot_start": ((type(None), int), None),
            "failsafe_strategy": (object, egx.FailsafeStrategy.REJECTION),
            "seed": ((type(None), int), None),
            "cstr_tol": ((type(None), list, tuple, np.ndarray), None),
        }

        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[0], default=value[1])

        self.available_outputs = {
            "x": (float, (self.problem.nx,)),
            "obj": float,
        }
        self.options.declare(
            "readable_outputs", values=([], ["x"], ["obj"], ["x", "obj"]), default=[]
        )

        self.x0 = self.problem.x0 * 1.0
        self.obj = self.problem._compute_objective
        self.active_callbacks = ["obj"]

        if self.problem.constrained:
            self.con = self.problem._compute_constraints
            self.active_callbacks += ["con"]

    def setup(self):
        self.solver_options.update(self.options["solver_options"])
        self.options_to_pass = self.solver_options.get_pure_dict()
        self.max_iters = self.options_to_pass.pop("max_iters")

        self._validate_problem()
        self._setup_xspecs()
        self._setup_constraints()
        self._normalize_solver_options()

    def _validate_problem(self):
        if "obj" not in self.problem.user_defined_callbacks:
            raise ValueError(
                "Egor requires an objective function. Feasibility problems are not supported."
            )

        if not np.all(np.isfinite(self.problem.x_lower)) or not np.all(
            np.isfinite(self.problem.x_upper)
        ):
            raise ValueError(
                "Egor requires finite lower and upper bounds on all design variables."
            )

        if np.any(self.problem.x_lower >= self.problem.x_upper):
            raise ValueError(
                "Each Egor design variable must satisfy x_lower < x_upper."
            )

        if (
            self.problem.constrained
            and "con" not in self.problem.user_defined_callbacks
        ):
            raise ValueError(
                "Constraint function is required for constrained problems when using Egor."
            )

    def _setup_xspecs(self):
        self.xspecs = np.column_stack(
            (self.problem.x_lower, self.problem.x_upper)
        ).tolist()

    def _setup_constraints(self):
        self.fcstrs = []
        self._constraint_specs = []

        if not self.problem.constrained:
            return

        cl = self.problem.c_lower
        cu = self.problem.c_upper

        for index, (lower, upper) in enumerate(zip(cl, cu)):
            if np.isfinite(lower) and np.isfinite(upper) and np.isclose(lower, upper):
                raise ValueError(
                    "Egor does not support equality constraints. "
                    "Use a different solver (PySLSQP, IPOPT, etc.) or reformulate the problem with inequalities."
                )

            if np.isfinite(upper):
                self._constraint_specs.append(
                    {"index": index, "kind": "upper", "bound": upper}
                )
            if np.isfinite(lower):
                self._constraint_specs.append(
                    {"index": index, "kind": "lower", "bound": lower}
                )

        if self._constraint_specs:
            self.fcstrs = [self._make_fcstr(spec) for spec in self._constraint_specs]

    def _normalize_solver_options(self):
        if self.options_to_pass["doe"] is not None:
            self.options_to_pass["doe"] = np.asarray(
                self.options_to_pass["doe"], dtype=float
            )

        cstr_tol = self.options_to_pass["cstr_tol"]
        if cstr_tol is not None:
            if not self.fcstrs:
                raise ValueError(
                    "solver_options['cstr_tol'] was provided but the problem has no inequality constraints."
                )

            cstr_tol = np.asarray(cstr_tol, dtype=float).reshape(-1)
            if cstr_tol.size != len(self.fcstrs):
                raise ValueError(
                    f"solver_options['cstr_tol'] must have length {len(self.fcstrs)} for Egor after "
                    "converting the modOpt constraint bounds to c(x) <= 0 form."
                )
            self.options_to_pass["cstr_tol"] = cstr_tol.tolist()

        if (
            self.fcstrs
            and self.options_to_pass["infill_optimizer"]
            == self.egx.InfillOptimizer.SLSQP
        ):
            self.check_if_callbacks_are_declared(
                "jac", "Constraint Jacobian", "Egor with SLSQP infill optimizer"
            )
            self.jac = self.problem._compute_constraint_jacobian
            self.active_callbacks += ["jac"]

    def _make_fcstr(self, spec):
        def fcstr(x, gradient=False):
            x = np.asarray(x, dtype=float)
            if gradient:
                return self._constraint_gradient(x, spec)
            return self._constraint_value(x, spec)

        return fcstr

    def _constraint_value(self, x, spec):
        values = np.asarray(self.con(x), dtype=float).reshape(-1)
        value = values[spec["index"]]
        if spec["kind"] == "upper":
            return float(value - spec["bound"])
        return float(spec["bound"] - value)

    def _constraint_gradient(self, x, spec):
        jacobian = np.asarray(self.jac(x), dtype=float)
        gradient = jacobian[spec["index"]].reshape(-1)
        if spec["kind"] == "upper":
            return gradient
        return -gradient

    def _egor_objective(self, x):
        # Egor can evaluate several candidate points at once. Convert batched inputs
        # to repeated modOpt objective evaluations and return a column vector.
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.asarray([[self.obj(x)]], dtype=float)
        if x.ndim != 2:
            raise ValueError(
                "Egor objective callback expects input with shape (n, nx) or (nx,)."
            )

        values = [self.obj(xi) for xi in x]
        return np.asarray(values, dtype=float).reshape((-1, 1))

    def solve(self):
        constructor_options = self.options_to_pass.copy()
        minimize_kwargs = {"max_iters": self.max_iters, "fcstrs": self.fcstrs}

        if "seed" in constructor_options and constructor_options["seed"] is not None:
            minimize_kwargs["seed"] = constructor_options.pop("seed")

        start_time = time.time()
        optimizer = self.egx.Egor(self.xspecs, **constructor_options)
        egor_output = optimizer.minimize(self._egor_objective, **minimize_kwargs)
        self.total_time = time.time() - start_time

        status = getattr(egor_output, "status", None)
        egor_result = getattr(egor_output, "result", egor_output)

        x_opt = np.asarray(egor_result.x_opt, dtype=float).reshape(-1)
        y_opt = np.asarray(egor_result.y_opt, dtype=float).reshape(-1)
        objective = float(y_opt[0])

        message = "Optimization completed successfully."
        if status is not None and hasattr(status, "exit"):
            message = str(status.exit)

        self.update_outputs(x=x_opt, obj=objective)

        self.results = {
            "x": x_opt,
            "fun": objective,
            "success": True,
            "message": message,
            "x_doe": np.asarray(egor_result.x_doe, dtype=float),
            "y_doe": np.asarray(egor_result.y_doe, dtype=float),
            "total_time": self.total_time,
        }

        if status is not None:
            self.results["status"] = status

        if self.problem.constrained:
            self.results["constraints"] = np.asarray(self.con(x_opt), dtype=float)

        self.run_post_processing()

        return self.results

    def print_results(self, optimal_variables=False, doe=False, all=False):
        output = "\n\tSolution from Egor:"
        output += "\n\t" + "-" * 100

        output += f"\n\t{'Problem':25}: {self.problem_name}"
        output += f"\n\t{'Solver':25}: {self.solver_name}"
        output += f"\n\t{'Success':25}: {self.results['success']}"
        output += f"\n\t{'Message':25}: {self.results['message']}"
        output += f"\n\t{'Total time':25}: {self.total_time}"
        output += f"\n\t{'Objective':25}: {self.results['fun']}"
        output += self.get_callback_counts_string(25)

        if self.problem.constrained:
            output += f"\n\t{'Constraints':25}: {self.results['constraints']}"
        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"
        if doe or all:
            output += f"\n\t{'DOE size':25}: {self.results['x_doe'].shape[0]}"

        output += "\n\t" + "-" * 100
        print(output)
