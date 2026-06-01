import time
import inspect

import numpy as np

from modopt import Optimizer
from modopt.utils.options_dictionary import OptionsDictionary


class Egor(Optimizer):
    """
    Class that interfaces modOpt with the ``egobox`` package's Egor optimizer.
    Egor is a gradient-free efficient global optimization solver that requires
    finite bounds on every design variable and supports nonlinear constraints
    with bound specifications (``<=``, ``>=``, equality, and double-sided).

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
        Supported options are ``'max_iters'``, ``'gp_config'``, ``'n_cstr'``, ``'n_start'``,
        ``'n_doe'``, ``'doe'``, ``'infill_strategy'``, ``'cstr_infill'``,
        ``'cstr_strategy'``, ``'qei_config'``, ``'infill_optimizer'``,
        ``'trego'``, ``'coego_n_coop'``, ``'target'``, ``'outdir'``,
        ``'warm_start'``, ``'hot_start'``, ``'failsafe_strategy'``,
        ``'seed'``, ``'cstr_tol'``, ``'run_info'``, ``'timeout'``, ``'verbose'``,
        ``'fcstrs'``, and ``'fcstr_specs'``.
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
            "n_cstr": (int, 0),
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
            "cstr_specs": ((type(None), list, tuple), None),
            "run_info": ((type(None), object), None),
            "timeout": ((type(None), float, int), None),
            "verbose": ((type(None), int, object), None),
            "fcstrs": ((list, tuple), []),
            "fcstr_specs": ((list, tuple), []),
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
        self._constraint_specs = []

        if not self.problem.constrained:
            return

        cl = self.problem.c_lower
        cu = self.problem.c_upper

        for index, (lower, upper) in enumerate(zip(cl, cu)):
            if not np.isfinite(lower) and not np.isfinite(upper):
                continue

            if np.isfinite(lower) and np.isfinite(upper):
                if np.isclose(lower, upper):
                    spec = self.egx.CstrSpec.eq(float(lower))
                else:
                    spec = self.egx.CstrSpec.btw(float(lower), float(upper))
            elif np.isfinite(upper):
                spec = self.egx.CstrSpec.leq(float(upper))
            else:
                spec = self.egx.CstrSpec.geq(float(lower))

            self._constraint_specs.append({"index": index, "spec": spec})

    def _normalize_solver_options(self):
        user_n_cstr = self.options_to_pass.get("n_cstr", 0)
        if user_n_cstr < 0:
            raise ValueError("solver_options['n_cstr'] must be >= 0.")

        user_cstr_specs = self.options_to_pass.get("cstr_specs")
        if user_cstr_specs is not None and not self.problem.constrained:
            raise ValueError(
                "solver_options['cstr_specs'] requires a constrained modOpt problem with a constraint callback."
            )

        if user_cstr_specs is not None and self.problem.constrained:
            raise ValueError(
                "Do not pass solver_options['cstr_specs'] for constrained modOpt problems. "
                "Constraint specs are generated automatically from problem.c_lower/c_upper."
            )

        if self.options_to_pass["doe"] is not None:
            self.options_to_pass["doe"] = np.asarray(
                self.options_to_pass["doe"], dtype=float
            )

        if self.problem.constrained:
            computed_n_cstr = len(self._constraint_specs)
            if user_n_cstr not in (0, computed_n_cstr):
                raise ValueError(
                    f"solver_options['n_cstr'] must be 0 or {computed_n_cstr} for this constrained modOpt problem."
                )

            self.options_to_pass["cstr_specs"] = [
                spec["spec"] for spec in self._constraint_specs
            ]
            self.options_to_pass["n_cstr"] = computed_n_cstr
        elif user_cstr_specs is None:
            if user_n_cstr != 0:
                raise ValueError(
                    "solver_options['n_cstr'] > 0 requires a constrained modOpt problem with constraint outputs."
                )
            self.options_to_pass.pop("cstr_specs", None)
            self.options_to_pass["n_cstr"] = 0

        cstr_tol = self.options_to_pass["cstr_tol"]
        if cstr_tol is not None:
            if not self._constraint_specs:
                raise ValueError(
                    "solver_options['cstr_tol'] was provided but the problem has no inequality constraints."
                )

            cstr_tol = np.asarray(cstr_tol, dtype=float).reshape(-1)
            if cstr_tol.size != len(self._constraint_specs):
                raise ValueError(
                    f"solver_options['cstr_tol'] must have length {len(self._constraint_specs)} for Egor after "
                    "converting the modOpt constraint bounds to c(x) <= 0 form."
                )
            self.options_to_pass["cstr_tol"] = cstr_tol.tolist()

    def _egor_fun(self, x):
        # Egor can evaluate several candidate points at once. Convert batched inputs
        # to repeated modOpt objective/constraint evaluations and return a matrix
        # with columns [obj, c_1, ..., c_n].
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape((1, -1))
        if x.ndim != 2:
            raise ValueError(
                "Egor objective callback expects input with shape (n, nx) or (nx,)."
            )

        obj_vals = [self.obj(xi) for xi in x]
        outputs = np.asarray(obj_vals, dtype=float).reshape((-1, 1))

        if self.problem.constrained:
            con_vals = [np.asarray(self.con(xi), dtype=float).reshape((1, -1)) for xi in x]
            con_block = np.vstack(con_vals)
            selected = con_block[:, [spec["index"] for spec in self._constraint_specs]]
            outputs = np.hstack((outputs, selected))

        return outputs

    def solve(self):
        constructor_options = self.options_to_pass.copy()
        minimize_kwargs = {"max_iters": self.max_iters}

        # Runtime-only controls are passed via minimize().
        # NOTE: Egobox accepts seed in the constructor, but this is deprecated upstream.
        for option_name in ("run_info", "timeout", "seed"):
            value = constructor_options.pop(option_name, None)
            if value is not None:
                minimize_kwargs[option_name] = value

        # Shared controls exist on both constructor and minimize in current Egobox API.
        for option_name in ("outdir", "warm_start", "hot_start", "verbose"):
            value = constructor_options.get(option_name, None)
            if value is not None:
                minimize_kwargs[option_name] = value

        # Function constraints are optional and independent of modOpt's grouped constraints.
        fcstrs = list(constructor_options.pop("fcstrs", []))
        if fcstrs:
            minimize_kwargs["fcstrs"] = fcstrs

        fcstr_specs = list(constructor_options.pop("fcstr_specs", []))

        minimize_signature = inspect.signature(self.egx.Egor.minimize)
        minimize_parameters = set(minimize_signature.parameters.keys())
        if fcstr_specs:
            if "fcstr_specs" not in minimize_parameters:
                raise ValueError(
                    "solver_options['fcstr_specs'] was provided, but the installed egobox version "
                    "does not support the fcstr_specs argument in Egor.minimize()."
                )
            minimize_kwargs["fcstr_specs"] = fcstr_specs

        start_time = time.time()
        optimizer = self.egx.Egor(self.xspecs, **constructor_options)
        egor_output = optimizer.minimize(self._egor_fun, **minimize_kwargs)
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
