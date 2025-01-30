import numpy as np
import time

from modopt import Optimizer
from modopt.line_search_algorithms import ScipyLS, Minpack2LS, BacktrackingArmijo
# from modopt.approximate_hessians import BFGS
from modopt.approximate_hessians import BFGSScipy as BFGS


class QuasiNewton(Optimizer):
    """
    Quasi-Newton method for unconstrained optimization.

    Parameters
    ----------
    problem : Problem or ProblemLite
        Object containing the problem to be solved.
    recording : bool, default=False
        If ``True``, record all outputs from the optimization.
        This needs to be enabled for hot-starting the same problem later,
        if the optimization is interrupted.
    hot_start_from : str, optional
        The record file from which to hot-start the optimization.
    hot_start_atol : float, default=0.
        The absolute tolerance check for the inputs
        when reusing outputs from the hot-start record.
    hot_start_rtol : float, default=0.
        The relative tolerance check for the inputs
        when reusing outputs from the hot-start record.
    visualize : list, default=[]
        The list of scalar variables to visualize during the optimization.
    turn_off_outputs : bool, default=False
        If ``True``, prevents modOpt from generating any output files.

    maxiter : int, default=500
        Maximum number of iterations.
    opt_tol : float, default=1e-6
        Optimality tolerance.
        Certifies convergence when the 2-norm of the gradient is less than this value.
    ls_type : {None, 'backtracking-armijo', 'derivative-based-strong-wolfe', default='derivative-based-strong-wolfe'}
        Type of line search to use.
    ls_min_step : float, default=1e-14
        Minimum step size for the line search.
    ls_max_step : float, default=1.
        Maximum step size for the line search.
    ls_maxiter : int, default=10
        Maximum number of iterations for the line search.
    ls_alpha_tol : float, default=1e-14
        Relative tolerance for an acceptable step in the line search.
    ls_gamma_c : float, default=0.3
        Step length contraction factor when backtracking.
    ls_eta_a : float, default=1e-4
        Armijo (sufficient decrease condition) parameter for the line search.
    ls_eta_w : float, default=0.9
        Wolfe (curvature condition) parameter for the line search.

    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'itr', 'obj', 'x', 'opt', 'time', 'nfev', 'ngev', 'step'.
    """
    def initialize(self):
        self.solver_name = 'bfgs'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

        self.options.declare('maxiter', default=500, types=int)
        self.options.declare('opt_tol', types=float, default=1e-6)

        self.options.declare('ls_type',
                             default='derivative-based-strong-wolfe',
                             values=[None,
                                     'backtracking-armijo',
                                     'derivative-based-strong-wolfe'])
        self.options.declare('ls_min_step', default=1e-14, types=float)
        self.options.declare('ls_max_step', default=1.0, types=float)
        self.options.declare('ls_maxiter', default=10, types=int)
        self.options.declare('ls_alpha_tol', default=1e-14, types=float)
        self.options.declare('ls_gamma_c', default=0.3, types=float)
        self.options.declare('ls_eta_a', default=1e-4, types=float)
        self.options.declare('ls_eta_w', default=0.9, types=float)

        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
            'nfev': int,
            'ngev': int,
            'step': float,
        }

    def setup(self):
        # self.LS = ScipyLS(f=self.obj, g=self.grad)
        # self.LS = Minpack2LS(f=self.obj, g=self.grad)

        if self.options['ls_type'] == 'backtracking-armijo':
            self.LS = BacktrackingArmijo(f=self.obj,
                                         g=self.grad,
                                         eta_a=self.options['ls_eta_a'],
                                         gamma_c=self.options['ls_gamma_c'],
                                         max_step=self.options['ls_max_step'],
                                         maxiter=self.options['ls_maxiter'])
        elif self.options['ls_type'] == 'derivative-based-strong-wolfe':
            self.LS = Minpack2LS(f=self.obj,
                                 g=self.grad,
                                 min_step=self.options['ls_min_step'],
                                 max_step=self.options['ls_max_step'],
                                 maxiter=self.options['ls_maxiter'],
                                 alpha_tol=self.options['ls_alpha_tol'],
                                 eta_a=self.options['ls_eta_a'],
                                 eta_w=self.options['ls_eta_w'])

        self.QN = BFGS(nx=self.problem.nx,
                       exception_strategy='damp_update')

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        x0 = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']

        obj = self.obj
        grad = self.grad

        start_time = time.time()

        # Set initial values for current iterates
        x_k = x0 * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)

        # Iteration counter
        itr = 0

        opt = np.linalg.norm(g_k)
        nfev = 1
        ngev = 1

        # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time,
                            nfev=nfev,
                            ngev=ngev,
                            step=0.)

        while (opt > opt_tol and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # Hessian approximation
            B_k = self.QN.B_k

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate
            p_k = np.linalg.solve(B_k, -g_k)

            # Compute the step length along the search direction via a line search
            if self.options['ls_type'] == 'backtracking-armijo':
                alpha, f_k, new_f_evals, new_g_evals, converged = self.LS.search(
                    x=x_k, p=p_k, f0=f_k, g0=g_k)

                g_new = grad(x_k + alpha * p_k)
                new_g_evals += 1

            elif self.options['ls_type'] == 'derivative-based-strong-wolfe':
                alpha, f_k, g_new, slope_new, new_f_evals, new_g_evals, converged = self.LS.search(
                    x=x_k, p=p_k, f0=f_k, g0=g_k)

            else:
                alpha, f_k, g_new, new_f_evals, new_g_evals, converged = (
                    1., obj(x_k + p_k), grad(x_k + p_k), 1, 1, True
                    )

            # Update the number of function and gradient evaluations
            nfev += new_f_evals
            ngev += new_g_evals

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                alpha = 1e-4
                d_k = p_k * alpha

                x_k += d_k
                f_k = obj(x_k)

                g_new = grad(x_k)
                w_k = g_new - g_k
                g_k = g_new

            else:
                d_k = alpha * p_k
                x_k += d_k

                if not isinstance(g_new, np.ndarray):
                    if g_new == 'Unavailable':
                        g_new = grad(x_k)
                w_k = g_new - g_k
                g_k = g_new

            opt = np.linalg.norm(g_k)

            # Update the Hessian approximation
            self.QN.update(d_k, w_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                opt=opt,
                                time=time.time() - start_time,
                                nfev=nfev,
                                ngev=ngev,
                                step=alpha)

        self.total_time = time.time() - start_time
        converged = opt <= opt_tol

        self.results = {
            'x': x_k, 
            'objective': f_k, 
            'optimality': opt, 
            'nfev': nfev, 
            'ngev': ngev,
            'niter': itr, 
            'time': self.total_time,
            'converged': converged,
            }
        
        # Run post-processing for the Optimizer() base class
        self.run_post_processing()
        
        return self.results