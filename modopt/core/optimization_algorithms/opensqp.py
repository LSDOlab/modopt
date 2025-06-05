import numpy as np
import time
import warnings

from modopt import Optimizer
from modopt.line_search_algorithms import Minpack2LS
from modopt.merit_functions import AugmentedLagrangian
from modopt.approximate_hessians import BFGSScipy


class OpenSQP(Optimizer):
    """
    A reconfigurable open-source Sequential Quadratic Programming (SQP) optimizer
    developed in modOpt for constrained nonlinear optimization.

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
    keep_viz_open : bool, default=False
        If ``True``, keeps the visualization window open after the optimization is complete.
    turn_off_outputs : bool, default=False
        If ``True``, prevent modOpt from generating any output files.

    maxiter : int, default=1000
        Maximum number of major iterations.
    opt_tol : float, default=1e-7
        Optimality tolerance.
    feas_tol : float, default=1e-7
        Feasibility tolerance.
        Certifies convergence when the "scaled" maximum constraint violation 
        is less than this value.
    aqp_primal_feas_tol : float, default=1e-8
        Tolerance for the primal feasibility of the augmented QP subproblem.
    aqp_dual_feas_tol : float, default=1e-8
        Tolerance for the dual feasibility of the augmented QP subproblem.
    aqp_time_limit : float, default=5.0
        Time limit for the augmented QP solution in seconds.
    ls_min_step : float, default=1e-12
        Minimum step size for the line search.
    ls_max_step : float, default=1.
        Maximum step size for the line search.
    ls_maxiter : int, default=10
        Maximum number of iterations for the line search.
    ls_eta_a : float, default=1e-4
        Armijo (sufficient decrease condition) parameter for the line search.
    ls_eta_w : float, default=0.9
        Wolfe (curvature condition) parameter for the line search.
    ls_alpha_tol : float, default=1e-14
        Relative tolerance for an acceptable step in the line search.

    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'major', 'obj', 'x', 'lag_mult', 'slacks', 'constraints', 'opt',
        'feas', 'sum_viol', 'max_viol', 'time', 'nfev', 'ngev', 'step', 'rho', 
        'merit', 'elastic', 'gamma', 'low_curvature'.
    """
    # qp_tol : float, default=1e-4
    #     Tolerance for the QP subproblem.
    # qp_maxiter : int, default=5000
    #     Maximum number of iterations for the QP subproblem.
    def initialize(self):
        self.solver_name = 'opensqp'

        self.nx = self.problem.nx

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.active_callbacks = ['obj', 'grad']
        if self.problem.constrained:
            self.con_in = self.problem._compute_constraints
            self.jac_in = self.problem._compute_constraint_jacobian
            self.active_callbacks += ['con', 'jac']

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-7, types=float)
        self.options.declare('feas_tol', default=1e-7, types=float)
        self.options.declare('readable_outputs', types=list, default=[])
        self.options.declare('aqp_primal_feas_tol', default=1e-8, types=float)
        self.options.declare('aqp_dual_feas_tol', default=1e-8, types=float)
        self.options.declare('aqp_time_limit', default=5.0, types=float)
        # self.options.declare('qp_maxiter', default=5000, types=int)

        self.options.declare('ls_min_step', default=1e-12, types=float)
        self.options.declare('ls_max_step', default=1.0, types=float)
        self.options.declare('ls_maxiter', default=10, types=int)
        self.options.declare('ls_eta_a', default=1e-4, types=float)
        self.options.declare('ls_eta_w', default=0.95, types=float)
        self.options.declare('ls_alpha_tol', default=1e-14, types=float)

        self.available_outputs = {
            'major': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),

            # Note that the number of constraints will
            # be updated after constraints are setup
            'lag_mult': (float, (self.problem.nc, )),
            'slacks': (float, (self.problem.nc, )),
            'constraints': (float, (self.problem.nc, )),
            'opt': float,
            'feas': float,
            'sum_viol': float,
            'max_viol': float,
            'time': float,
            'nfev': int,
            'ngev': int,
            'step': float,
            'rho': float,
            'merit': float,
            'elastic': int,
            'gamma': float,
            'low_curvature': int,
        }

    def setup(self):
        self.setup_constraints()
        nx   = self.nx
        nc   = self.nc
        nc_e = self.nc_e
        self.available_outputs['lag_mult'] = (float, (nc, ))
        self.available_outputs['slacks'] = (float, (nc, ))
        self.available_outputs['constraints'] = (float, (nc, ))
        # Damping parameters
        self.delta_rho = 1.
        self.num_rho_changes = 0
        self.num_rho_increases = 0
        self.num_rho_decreases = 0

        self.successive_undefined_iterations = 0
        
        self.QN = BFGSScipy(nx=nx,
                            exception_strategy='damp_update',
                            init_scale=1.0)
            
        self.MF = AugmentedLagrangian(nx=nx,
                                      nc=nc,
                                      nc_e=nc_e,
                                      f=self.obj,
                                      c=self.con,
                                      g=self.grad,
                                      j=self.jac,
                                      non_bound_indices=self.non_bound_indices)

        self.LSS = Minpack2LS(f=self.MF.compute_function,
                              g=self.MF.compute_gradient,
                              min_step=self.options['ls_min_step'],
                              max_step=self.options['ls_max_step'],
                              maxiter=self.options['ls_maxiter'],
                              eta_a=self.options['ls_eta_a'],
                              eta_w=self.options['ls_eta_w'],
                              alpha_tol=self.options['ls_alpha_tol'],
                              )

    # Adapt constraints to C_e(x)=0, and C_i(x) >= 0
    def setup_constraints(self, ):
        xl = self.problem.x_lower
        xu = self.problem.x_upper

        ebi = self.eq_bound_indices    = np.array([])
        lbi = self.lower_bound_indices = np.where((xl != -np.inf))[0]
        ubi = self.upper_bound_indices = np.where((xu !=  np.inf))[0]

        self.eq_bounded    = True if len(ebi) > 0 else False
        self.lower_bounded = True if len(lbi) > 0 else False
        self.upper_bounded = True if len(ubi) > 0 else False   

        if self.problem.constrained:
            cl = self.problem.c_lower
            cu = self.problem.c_upper

            eci = self.eq_constraint_indices    = np.where(cl == cu)[0]
            lci = self.lower_constraint_indices = np.where((cl != -np.inf) & (cl != cu))[0]
            uci = self.upper_constraint_indices = np.where((cu !=  np.inf) & (cl != cu))[0]

        else:
            eci = np.array([])
            lci = np.array([])
            uci = np.array([])

        self.eq_constrained    = True if len(eci) > 0 else False
        self.lower_constrained = True if len(lci) > 0 else False
        self.upper_constrained = True if len(uci) > 0 else False

        self.nc_e = len(ebi) + len(eci)
        self.nc_b = len(lbi) + len(ubi)
        self.nc_i = len(lbi) + len(ubi) + len(lci) + len(uci)
        self.nc   = self.nc_e + self.nc_i

        self.non_bound_indices = np.concatenate([np.arange(self.nc_e), np.arange(self.nc_e+self.nc_b, self.nc)])

    def con(self, x):
        ebi = self.eq_bound_indices
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices
        
        if self.problem.constrained:
            eci = self.eq_constraint_indices
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Compute problem constraints
            c_in = self.con_in(x)

        c_out = np.array([])

        if self.eq_bounded:
            c_out = np.append(c_out, x[ebi] - self.problem.x_lower[ebi])
        if self.eq_constrained:
            c_out = np.append(c_out, c_in[eci] - self.problem.c_lower[eci])

        if self.lower_bounded:
            c_out = np.append(c_out, x[lbi] - self.problem.x_lower[lbi])
        if self.upper_bounded:
            c_out = np.append(c_out, self.problem.x_upper[ubi] - x[ubi])
        if self.lower_constrained:
            c_out = np.append(c_out, c_in[lci] - self.problem.c_lower[lci])
        if self.upper_constrained:
            c_out = np.append(c_out, self.problem.c_upper[uci] - c_in[uci])

        return c_out

    def jac(self, x):
        nx = self.nx
        # ebi = self.eq_bound_indices
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices

        if self.problem.constrained:
            eci = self.eq_constraint_indices
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Compute problem constraint Jacobian
            j_in = self.jac_in(x)

        j_out = np.empty((1, nx), dtype=float)

        # if self.eq_bounded:
        #     j_out = np.append(j_out, np.identity(nx)[ebi], axis=0)
        if self.eq_constrained:
            j_out = np.append(j_out, j_in[eci], axis=0)

        if self.lower_bounded:
            j_out = np.append(j_out, np.identity(nx)[lbi], axis=0)
        if self.upper_bounded:
            j_out = np.append(j_out, -np.identity(nx)[ubi], axis=0)
        if self.lower_constrained:
            j_out = np.append(j_out, j_in[lci], axis=0)
        if self.upper_constrained:
            j_out = np.append(j_out, -j_in[uci], axis=0)

        return j_out[1:]

    # Minimize A.L. w.r.t. slacks s.t. s >= 0
    def reset_slacks(self, c, pi, rho):
        rho_zero_indices    = np.where(rho == 0)[0]
        rho_nonzero_indices = np.where(rho  > 0)[0]
        s = np.zeros((len(rho), ))

        if len(rho_zero_indices) > 0:
            # TODO: test over a range of problems to see which one is better
            s[rho_zero_indices] = 0.
            # s[rho_zero_indices] = np.maximum(0, c[rho_zero_indices])

        if len(rho_nonzero_indices) > 0:
            c_rnz   = c[rho_nonzero_indices]
            pi_rnz  = pi[rho_nonzero_indices]
            rho_rnz = rho[rho_nonzero_indices]
            s[rho_nonzero_indices] = np.maximum(0, (c_rnz - pi_rnz / rho_rnz))

        return s

    def update_scalar_rho(self, rho_k, dir_deriv_al, pTHp, p_pi, c_k, s_k):
        # Damping for scalar rho as in SNOPT
        nc = self.nc
        delta_rho = self.delta_rho
        nbi = self.non_bound_indices

        rho_ref = rho_k[0] * 1.

        # note: scalar rho_min here
        if np.linalg.norm((c_k - s_k)) == 0:
            rho_min = 0. # See lemma 4.3 in the paper
        else:
            rho_min = 2 * np.linalg.norm(p_pi) / np.linalg.norm((c_k - s_k))
            
        rho_computed = rho_min * 1.0

        # Damping for rho (Note: vector rho is still not updated)
        if rho_k[0] < 4 * (rho_computed + delta_rho):
            rho_damped = rho_k[0]
        else:
            rho_damped = np.sqrt(rho_k[0] * (rho_computed + delta_rho))

        rho_new = max(rho_computed, rho_damped)
        rho_k[:]   = rho_new

        # Increasing rho
        if rho_k[0] > rho_ref:
            if self.num_rho_changes >= 0:
                self.num_rho_changes += 1
            else:
                self.delta_rho *= 2.
                self.num_rho_changes = 0

        # Decreasing rho
        elif rho_k[0] < rho_ref:
            if self.num_rho_changes <= 0:
                self.num_rho_changes -= 1
            else:
                self.delta_rho *= 2.
                self.num_rho_changes = 0

        return rho_k

    def update_vector_rho(self, rho_k, dir_deriv_al, pTHp, p_pi, c_k, s_k):
        delta_rho = self.delta_rho
        num_rho_changes = self.num_rho_changes
        nbi = self.non_bound_indices

        rho_ref = np.linalg.norm(rho_k)

        a = (c_k - s_k) ** 2
        b = 0.5 * pTHp + dir_deriv_al + np.inner(a, rho_k)

        if b > 0:
            a_norm = np.linalg.norm(a)
            if a_norm > 0: # Just to be extra cautious (elastic iterations)
                rho_computed = (b / (a_norm ** 2)) * a
            else:
                rho_computed = np.zeros((len(rho_k), ))
        else:
            rho_computed = np.zeros((len(rho_k), ))

        # Damping for rho (Note: vector rho is still not updated)
        rho_damped = np.where(
            rho_k < 4 * (rho_computed + delta_rho), rho_k,
            np.sqrt(rho_k * (rho_computed + delta_rho)))

        rho_new = np.maximum(rho_computed, rho_damped)
        rho_k[:] = rho_new

        rho_norm = np.linalg.norm(rho_k)

        # Increasing rho
        if rho_norm > rho_ref:
            if self.num_rho_decreases >= 2:
                self.delta_rho *= 2.
            self.num_rho_increases += 1
            self.num_rho_decreases = 0
        # Decreasing rho
        elif rho_norm < rho_ref:
            if self.num_rho_increases >= 2:
                self.delta_rho *= 2.
            self.num_rho_decreases += 1
            self.num_rho_increases = 0
        # Constant rho
        else:
            self.num_rho_increases = 0
            self.num_rho_decreases = 0

        return rho_k

    def l1_penalty_line_search(self, x_k, eta, x_qp, p_k, rho_k_Powell, f_k, c_k, g_k):
        nc_e = self.nc_e
        nx   = self.nx

        new_f_evals = 0
        converged = False
        mu = rho_k_Powell[self.non_bound_indices]
        c_viol = c_k[self.non_bound_indices]
        c_viol[:nc_e] = np.abs(c_viol[:nc_e])
        c_viol[nc_e:] = np.maximum(0, -c_viol[nc_e:])
        # Penalty merit function value at alpha = alpha
        mf0 = f_k + np.dot(mu, c_viol)

        exred = g_k.T @ x_qp - np.dot(mu, c_viol) * (1-eta)

        ls_iter = 0
        step = 1.0
        alpha = 1.0
        alpha_min = 1e-1
        d = p_k[:nx] * 1.0
        while True:
            ls_iter += 1
            exred = alpha*exred # expected reduction (<0) in the merit function
            # Note that d needs to recursively scaled down: d != alpha * x_qp 
            d  = alpha * d
            x = x_k + d
            self.MF.update_functions_in_cache(['f', 'c'], x)
            f = self.MF.cache['f'][1]
            c = self.MF.cache['c'][1]
            new_f_evals += 1

            c_viol = c[self.non_bound_indices]
            c_viol[:nc_e] = np.abs(c_viol[:nc_e])
            c_viol[nc_e:] = np.maximum(0, -c_viol[nc_e:])

            # Penalty merit function value at alpha = alpha
            mf = f + np.dot(mu, c_viol)
            # Actual reduction in the merit function
            acred = mf - mf0

            # Break out of line search if the change (-ve) in the merit function 
            # is at least one-tenth of the expected change exred (always negative)
            # i.e., the obtained change in the merit function must be more negative 
            # than one-tenth of the expected reduction
            if (acred<=exred/10.0 or ls_iter > 10):
                new_g_evals = 0
                alpha = step * 1.0
                if acred <= exred:
                    converged = True
                break

            # Otherwise,
            alpha = np.maximum(exred/(2*(exred-acred)), alpha_min)
            step *= alpha

        return new_f_evals, converged, step

    def opt_check(self, pi, c, g, j):
        opt_tol = self.options['opt_tol']

        if self.nc == 0:
            opt_tol_factor = 1.
            nonneg_violation = 0.
            compl_violation = 0.
            stat_violation = np.linalg.norm(g, np.inf)
            
        else:
            opt_tol_factor = (1 + np.linalg.norm(pi, np.inf))
            if pi[self.nc_e:].size == 0:
                nonneg_violation = 0.
            else:
                nonneg_violation = max(0., -np.min(pi[self.nc_e:]))
            # Note: compl_violation can be negative(Question on SNOPT);
            #       Other 2 violations are always nonnegative
            compl_violation = np.max(np.abs(c * pi))
            stat_violation  = np.linalg.norm(g - j.T @ pi, np.inf)

        
        scaled_opt_tol = opt_tol * opt_tol_factor
        opt_check1 = (nonneg_violation <= scaled_opt_tol)
        opt_check2 = (compl_violation <= scaled_opt_tol)
        opt_check3 = (stat_violation <= scaled_opt_tol)

        # opt is always nonnegative
        print('opt_tol_factor:', opt_tol_factor)
        print('nonneg_violation:', nonneg_violation)
        print('compl_violation:', compl_violation)
        print('stat_violation:', stat_violation)
        opt = max(nonneg_violation, compl_violation,
                  stat_violation) / opt_tol_factor
        opt_satisfied = (opt_check1 and opt_check2 and opt_check3)

        return opt_satisfied, opt

    def feas_check(self, x, c):
        feas_tol = self.options['feas_tol']

        feas_tol_factor = (1 + np.linalg.norm(x, np.inf))
        scaled_feas_tol = feas_tol * feas_tol_factor

        # Violation is positive
        max_con_violation_eq   = np.max(np.abs(c[:self.nc_e]))   if self.nc_e > 0 else 0.
        max_con_violation_ineq = max(0., -np.min(c[self.nc_e:])) if self.nc_i > 0 else 0.
        max_con_violation      = max(max_con_violation_eq, max_con_violation_ineq)
        feas_check = (max_con_violation <= scaled_feas_tol)

        sum_con_violation_eq   = np.sum(np.abs(c[:self.nc_e]))
        sum_con_violation_ineq = np.sum(np.maximum(0., -c[self.nc_e:]))
        sum_con_violation      = sum_con_violation_eq + sum_con_violation_ineq

        feas = max_con_violation / feas_tol_factor
        feas_satisfied = feas_check
        return feas_satisfied, feas, sum_con_violation, max_con_violation

    def get_results_dict(self, x_k, f_k, c_k, pi_k, opt, feas, nfev, ngev, niter, time, success):
        results = {'x': x_k,
                   'objective': f_k,
                   'c': c_k,
                   'pi': pi_k,
                   'optimality': opt,
                   'feasibility': feas,
                   'nfev': nfev,
                   'ngev': ngev,
                   'niter': niter,
                   'time': time,
                   'success': success}
        return results

    def solve(self):
        try:
            import qpsolvers
        except ImportError:
            raise ImportError("qpsolvers cannot be imported for the QP solver.  Install it with 'pip install qpsolvers'.")
        
        try:
            from quadprog import solve_qp
        except ImportError:
            raise ImportError("quadprog cannot be imported for the QP solver.  Install it with 'pip install quadprog'.")
        
        try:
            import highspy
        except ImportError:
            raise ImportError("HiGHS cannot be imported for the QP solver.  Install it with 'pip install highspy'.")

        # Assign shorter names to variables and methods
        nx = self.nx
        nc = self.nc
        nc_e = self.nc_e
        nc_i = self.nc_i

        x0 = self.problem.x0
        maxiter = self.options['maxiter']
        aqp_primal_tol  = self.options['aqp_primal_feas_tol']
        aqp_dual_tol    = self.options['aqp_dual_feas_tol']
        aqp_time_limit  = self.options['aqp_time_limit']
        # qp_maxiter = self.options['qp_maxiter']

        LSS = self.LSS
        QN = self.QN
        MF = self.MF

        eps = 2.22e-16

        start_time = time.time()

        # Proximal point initialization for a feasible start wrt bounds
        x_k = np.clip(x0, self.problem.x_lower, self.problem.x_upper)
        
        self.MF.update_functions_in_cache(['f', 'c', 'g', 'j'], x_k)
        f_k = self.MF.cache['f'][1]
        g_k = self.MF.cache['g'][1]
        c_k = self.MF.cache['c'][1]
        J_k = self.MF.cache['j'][1]
        undefined_proximal_point = False

        if np.isnan(f_k) or np.isinf(f_k):
            warnings.warn('Objective value at the computed proximal initial point is NaN or Inf. Trying given initial point ...')
            undefined_proximal_point = True
        elif np.any(np.isnan(c_k)) or np.any(np.isinf(c_k)):
            warnings.warn('Constraint values at the computed proximal initial point contains NaN or Inf. Trying given initial point ...')
            undefined_proximal_point = True
        elif np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
            warnings.warn('Objective gradient at the computed proximal initial point contains NaN or Inf. Trying given initial point ...')
            undefined_proximal_point = True
        elif np.any(np.isnan(J_k)) or np.any(np.isinf(J_k)):
            warnings.warn('Constraint Jacobian at the computed proximal initial point contains NaN or Inf. Trying given initial point ...')
            undefined_proximal_point = True
        else:
            print('Proximal point initialization is well-defined and good to go.')
        
        if undefined_proximal_point:
            if np.all(x0 == x_k):
                print('Initial point provided and proximal point computed were the same and is undefined. Exiting ...')
                return self.get_results_dict(x_k, f_k, c_k, None, None, None, 1, 1, 0, time.time() - start_time, False)
            
            x_k = x0 * 1.

            self.MF.update_functions_in_cache(['f', 'c', 'g', 'j'], x_k)
            f_k = self.MF.cache['f'][1]
            g_k = self.MF.cache['g'][1]
            c_k = self.MF.cache['c'][1]
            J_k = self.MF.cache['j'][1]

            if np.isnan(f_k) or np.isinf(f_k):
                print('Objective value at given initial point and computed proximal point is NaN or Inf. Exiting ...')
                return self.get_results_dict(x_k, f_k, c_k, None, None, None, 2, 2, 0, time.time() - start_time, False)
            elif np.any(np.isnan(c_k)) or np.any(np.isinf(c_k)):
                print('Constraint values at given initial point and computed proximal point contains NaN or Inf. Exiting ...')
                return self.get_results_dict(x_k, f_k, c_k, None, None, None, 2, 2, 0, time.time() - start_time, False)
            elif np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
                print('Gradient at given initial point and computed proximal point contains NaN or Inf. Exiting ...')
                return self.get_results_dict(x_k, f_k, c_k, None, None, None, 2, 2, 0, time.time() - start_time, False)
            elif np.any(np.isnan(J_k)) or np.any(np.isinf(J_k)):
                print('Constraint Jacobian at given initial point and computed proximal point contains NaN or Inf. Exiting ...')
                return self.get_results_dict(x_k, f_k, c_k, None, None, None, 2, 2, 0, time.time() - start_time, False)

        # Set initial values for multipliers and slacks
        # pi_k = np.full((nc, ), 1.)
        pi_k = np.full((nc, ), 0.)

        nfev = 1
        ngev = 1

        # Vector of penalty parameters
        rho_k = np.full((nc, ), 0.)
        rho_k_Powell = np.full((nc, ), 0.)

        # Compute slacks by minimizing A.L. s.t. s_k >= 0
        s_k = self.reset_slacks(c_k[nc_e:], pi_k[nc_e:], rho_k[nc_e:])
        # s_k = np.maximum(0, c_k) # Use this if rho_k = 0. at start

        # Vector of design vars., lag. mults., and slack vars.
        v_k  = np.concatenate((x_k, pi_k, s_k))
        x_k  = v_k[:nx]
        pi_k = v_k[nx:(nx + nc)]
        s_k  = v_k[(nx + nc):]

        p_k  = np.zeros((len(v_k), ))
        p_x  = p_k[:nx]
        p_pi = p_k[nx:(nx + nc)]
        p_s  = p_k[(nx + nc):]

        # Iteration counter
        itr = 0

        opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
        if self.nc == 0:
            feas_satisfied, feas, sviol, mviol = True, 0., 0., 0.
        else:
            feas_satisfied, feas, sviol, mviol = self.feas_check(x_k, c_k)
        
        tol_satisfied = (opt_satisfied and feas_satisfied)

        # Evaluate merit function value
        MF.set_rho(rho_k)
        mf_k  = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
        mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)

        # Initializing declared outputs
        self.update_outputs(major=0,
                            x=x_k,
                            lag_mult=pi_k,
                            slacks=s_k,
                            obj=f_k,
                            constraints=c_k,
                            opt=opt,
                            feas=feas,
                            sum_viol= sviol,
                            max_viol=mviol,
                            time=time.time() - start_time,
                            nfev=nfev,
                            ngev=ngev,
                            step=0.,
                            # rho=rho_k[0],
                            rho=np.linalg.norm(rho_k),
                            merit=mf_k,
                            elastic=0,
                            gamma=0.,
                            low_curvature=0)
                            # merit=penalty_k)

        elastic_itr = 0
        gamma_0 = 1e6
        max_gamma = 1e12
        gamma = gamma_0 * 1.
        el_wt_check_freq = 25
        el_feas_arr = np.zeros((el_wt_check_freq, ))

        while itr < maxiter:
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate

            # Solve a strictly convex quadratic program
            # Minimize     1/2 x^T G x - a^T x
            # Subject to   C.T x >= b

            # def solve_qp(double[:, :] G, double[:] a, double[:, :] C=None, double[:] b=None, int meq=0, factorized=False):
            # First meq constraints are treated as equality constraints

            try:
                if c_k.size == 0:
                    x_qp, f_qp, xu_qp, iter_qp, lag_qp, iact_qp = solve_qp(QN.B_k, -g_k)
                    lag_qp = np.array([])
                else:
                    x_qp, f_qp, xu_qp, iter_qp, lag_qp, iact_qp = solve_qp(QN.B_k, -g_k, C=J_k.T, b=-c_k, meq=nc_e)
                    gamma = gamma_0 * 1.

                p_k[:nx] = x_qp
                p_k[nx:(nx + nc)] = lag_qp - v_k[nx:nx + nc]
                elastic_itr = 0         # reset elastic_itr if QP is successful
                elastic_mode = False    # reset elastic_mode if QP is successful
                eta = 0.

            except Exception as e:
                print('QP solver failed at major iteration:', itr)
                print(e)

                if "matrix G is not positive definite" in str(e):
                    print('Matrix G is not positive definite. Resetting Hessian.')
                    # Reset Hessian
                    wTw = np.dot(w_k, w_k)
                    wTd = np.dot(w_k, d_k[:nx])

                    init_scale = wTw / (wTd+1e-16) if wTd > 0 else 1.
                    self.QN = QN = BFGSScipy(nx=nx,
                                            exception_strategy='damp_update',
                                            init_scale=init_scale)
                                            # init_scale=np.linalg.norm(np.diag(QN.B_k)))
                    
                    continue # Skip the rest of the iteration and solve QP with new reset Hessian

                # break
                if 'constraints are inconsistent' in str(e) or \
                "unsupported operand type(s) for -: 'NoneType' and 'float'" in str(e) or \
                "bad operand type for unary -: 'NoneType'" in str(e) or \
                "x_qp is zero" in str(e) or \
                "Rank(A) < p or Rank([P; A; G]) < n" in str(e):
                    print('Infeasible QP problem. Entering elastic mode.')
                    elastic_mode = True
                    while True:
                        el_feas_arr[elastic_itr % el_wt_check_freq] = feas
                        elastic_itr += 1
                        print('Elastic programmming with weight:', gamma)

                        # METHOD: Modified Powell's treatment of infeasible QP problems - eq ->ineq
                        # TODO: only for violation of inequality constraints
                        # minimize 1/2 x^T B_k x + g_k^T x + gamma/2 * \eta^2
                        # subject to J_k^T x + c_k(1-\eta)  = 0
                        #            J_k^T x + c_k(1-\eta) >= 0 # only for violated constraints
                        G_elastic = np.block([[QN.B_k, np.zeros((nx, 1))],
                                              [np.zeros((1, nx)),   gamma]])
                        a_elastic = -np.concatenate((g_k, [0.]))

                        el_c_k = np.concatenate((c_k, -c_k[:nc_e]))
                        el_c_k = -np.maximum(0, -el_c_k)
                        J_elastic = np.block([[J_k,         -el_c_k[:nc].reshape(nc, 1)],
                                              [-J_k[:nc_e], -el_c_k[nc:].reshape(nc_e, 1)],
                                              [np.zeros((2, nx)), np.array([[1.],[-1.]])]])

                        b_elastic = np.concatenate((-c_k, c_k[:nc_e], [0., -1.]))

                        if elastic_itr > el_wt_check_freq:
                                if gamma < max_gamma:
                                    gamma *= 10.
                                    
                                elastic_itr = 0

                        qp_prob = qpsolvers.Problem(G_elastic, -a_elastic, 
                                                    -J_elastic, -b_elastic)
                        qp_solver = 'highs'
                        qp_initvals = np.zeros((nx+1, ))
                        qp_initvals[-1] = 1.
                        print('Using elastic QP solver', qp_solver, 'with weight', gamma)
                        qp_sol = qpsolvers.solve_problem(qp_prob,
                                                         qp_solver,
                                                        #  initvals=qp_initvals,
                                                        #  verbose=True,
                                                        #  highs
                                                         dual_feasibility_tolerance=aqp_dual_tol,
                                                         primal_feasibility_tolerance=aqp_primal_tol,
                                                         time_limit=aqp_time_limit,
                                                         )
                        x_qp = qp_sol.x[:nx] * 1.
                        lag_qp = qp_sol.z[:nc] * 1.

                        if nc_e > 0:
                            lag_qp[:nc_e] -= qp_sol.z[nc:nc+nc_e]

                        p_k[:nx] = x_qp[:nx]
                        p_k[nx:(nx + nc)] = lag_qp - v_k[nx:nx + nc]
                        eta = qp_sol.x[nx] * 1.
                        break

                if np.linalg.norm(qp_sol.x[:nx]) == 0 and qp_sol.x[nx] == 1.:
                    print('Augmented QP is also infeasible. Terminating ...')
                    return self.get_results_dict(x_k, f_k, c_k, pi_k, opt, feas, nfev, ngev, itr, time.time() - start_time, False)

            # Clip the step length such that the design variables remain within bounds
            p_k[:nx] = np.clip(p_x, self.problem.x_lower - x_k, self.problem.x_upper - x_k)

            # Search direction for s_k :
            # (c_k + J_k @ p_x) is the new estimate for s for iniequality constraints
            p_k[(nx + nc):] = c_k[nc_e:] + J_k[nc_e:] @ p_k[:nx] - v_k[nx + nc:]

            print('Major iteration:', itr)
            print("=====================================")

            if self.nc > 0:
                dir_deriv_al = np.dot(mfg_k, p_k)
                pTHp = p_x.T @ (QN.B_k @ p_x)

                # Update penalty parameters
                rho_k_Powell = np.maximum(np.abs(lag_qp), 0.5*(rho_k_Powell + np.abs(lag_qp)))
                rho_k = self.update_vector_rho(rho_k, dir_deriv_al, pTHp,
                                               p_pi, c_k, np.concatenate([np.zeros((nc_e,)), s_k]))

            MF.set_rho(rho_k)
            mf_k  = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)
            dir_deriv_al_0 = np.dot(mfg_k, p_k)

            # Compute the step length along the search direction via a line search
            p_k_temp = p_k * 1.
            v_k_temp = v_k * 1.

            if dir_deriv_al_0 > -2.22e-16 or np.linalg.norm(p_k[:nx]) <= 2.22e-16:
                alpha = 1.0
                mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = mf_k, mfg_k, dir_deriv_al_0, 1, 1, True

            else:
                alpha, mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = LSS.search(
                    x=v_k_temp, p=p_k_temp, f0=mf_k, g0=mfg_k)

            undefined_direction = False
            if np.isnan(mf_new) or np.isinf(mf_new) or np.isnan(mfg_new).any() or np.isinf(mfg_new).any():
                undefined_direction = True
                print('Merit function or its gradient at the new point has NaN or inf.')
            else:
                nfev += new_f_evals
                ngev += new_g_evals

            alpha_golden = 0.061803398875
            if (not converged) and (not undefined_direction):
                # Use an inexact LS with l1-penalty and only function evaluations
                new_f_evals, converged, alpha = self.l1_penalty_line_search(x_k, eta, x_qp, p_k, rho_k_Powell, f_k, c_k, g_k)

                nfev += new_f_evals
                
            if not undefined_direction:
                print("###### SUCCESSFUL LINE SEARCH #######")
                d_k = alpha * p_k
                d_k[nx:(nx + nc)] = d_k[nx:(nx + nc)] / np.sqrt(alpha)
                d_k_temp = d_k * 1.


            g_old = g_k * 1.
            J_old = J_k * 1.

            # If the line search failed with nans at alpha=1 earlier, find a smaller alpha where the function is well-defined
            undefined_new_point = False
            if undefined_direction:

                x_k_new = x_k + p_k[:nx]

                self.MF.update_functions_in_cache(['f', 'g', 'c', 'j'], x_k_new)
                f_new = self.MF.cache['f'][1]
                g_new = self.MF.cache['g'][1]
                c_new = self.MF.cache['c'][1]
                J_new = self.MF.cache['j'][1]
                alpha = 1.0
                new_f_evals = 0
                new_g_evals = 0

                while np.isnan(f_new) or np.isinf(f_new):
                    print('Objective function is NaN or Inf. Stepping back x_k_new.')
                    alpha *= 0.1
                    d_k_temp = alpha * p_k
                    x_k_new = x_k + d_k_temp[:nx]
                    self.MF.update_functions_in_cache(['f'], x_k_new)
                    f_new = self.MF.cache['f'][1]
                    new_f_evals += 1
                    if alpha < 1e-12:
                        undefined_new_point = True
                        break
                
                if not undefined_new_point:
                    while np.any(np.isnan(c_new)) or np.any(np.isinf(c_new)):
                        print('Constraint values contain NaN or Inf. Stepping back x_k_new.')
                        alpha *= 0.1
                        d_k_temp = alpha * p_k
                        x_k_new = x_k + d_k_temp[:nx]
                        self.MF.update_functions_in_cache(['c'], x_k_new)
                        c_new = self.MF.cache['c'][1]
                        new_f_evals += 1
                        if alpha < 1e-12:
                            undefined_new_point = True
                            break
                
                if not undefined_new_point:
                    while np.any(np.isnan(g_new)) or np.any(np.isinf(g_new)):
                        print('Objective gradient contains NaN or Inf. Stepping back x_k_new.')
                        alpha *= 0.1
                        d_k_temp = alpha * p_k
                        x_k_new = x_k + d_k_temp[:nx]
                        self.MF.update_functions_in_cache(['g'], x_k_new)
                        g_new = self.MF.cache['g'][1]
                        new_g_evals += 1
                        if alpha < 1e-12:
                            undefined_new_point = True
                            break

                if not undefined_new_point:
                    while np.any(np.isnan(J_new)) or np.any(np.isinf(J_new)):
                        print('Constraint Jacobian contains NaN or Inf. Stepping back x_k_new.')
                        alpha *= 0.1
                        d_k_temp = alpha * p_k
                        x_k_new = x_k + d_k_temp[:nx]
                        self.MF.update_functions_in_cache(['j'], x_k_new)
                        J_new = self.MF.cache['j'][1]
                        new_g_evals += 1
                        if alpha < 1e-12:
                            undefined_new_point = True
                            break

                if self.problem.constrained:
                    nfev += int(new_f_evals/2)
                    ngev += int(new_g_evals/2)
                else:
                    nfev += new_f_evals
                    ngev += new_g_evals
            
            if undefined_new_point:
                self.successive_undefined_iterations += 1
                if self.successive_undefined_iterations == 1:
                    print('No points along the search direction is well-defined. Resetting Hessian.')
                    self.QN = QN = BFGSScipy(nx=nx,
                                             exception_strategy='damp_update',
                                             init_scale=1.)
                    continue

                if self.successive_undefined_iterations == 2:
                    print('Two successive iterations with unsuccessful search along predicted direction for well-defined points. Terminating ...')
                    return self.get_results_dict(x_k, f_k, c_k, pi_k, opt, feas, nfev, ngev, itr, time.time() - start_time, False)
     
            elif undefined_direction:
                self.successive_undefined_iterations = 0

                print('alpha successful without nan:', alpha)

                # If the line search failed with nans at alpha=1 earlier, reperform line search with a smaller alpha found above
                alpha_new, mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = LSS.search(
                    x=v_k_temp, p=alpha*p_k, f0=mf_k, g0=mfg_k)
                
                nfev += new_f_evals
                ngev += new_g_evals
                
                # If the line search failed to find a point satisfying the Wolfe conditions, try backtracking line search
                if not converged: 
                    print('Inside SLSQP line search: Reperforming backtracking line search with a smaller alpha found above.')
                    
                    new_f_evals, converged, alpha_new = self.l1_penalty_line_search(x_k, eta, x_qp, alpha*p_k, rho_k_Powell, f_k, c_k, g_k)
                    nfev += new_f_evals

                alpha *= alpha_new
                d_k = alpha * p_k
                d_k[nx:(nx + nc)] = d_k[nx:(nx + nc)] / np.sqrt(alpha) 
                d_k_temp = d_k * 1.

            v_k += d_k_temp

            x_k  = v_k[:nx]
            pi_k = v_k[nx:(nx + nc)]

            g_old = g_k * 1.
            J_old = J_k * 1.

            self.MF.update_functions_in_cache(['f', 'c', 'g', 'j'], x_k)
            f_k = self.MF.cache['f'][1]
            c_k = self.MF.cache['c'][1]
            g_k = self.MF.cache['g'][1]
            J_k = self.MF.cache['j'][1]

            v_k[(nx + nc):] = self.reset_slacks(c_k[nc_e:], pi_k[nc_e:], rho_k[nc_e:])
            s_k = v_k[(nx + nc):]

            # Note: MF changes (decreases) after slack reset
            mf_k  = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)

            w_k = (g_k - g_old) - (J_k - J_old).T @ pi_k


            dTd = np.dot(d_k[:nx], d_k[:nx])
            wTw = np.dot(w_k, w_k)
            wTd = np.dot(w_k, d_k[:nx])
            B_scaler = wTw / (wTd + 2.22e-16)
            dBd = np.dot(d_k[:nx], QN.B_k @ d_k[:nx])

            low_curvature = 1 if (wTd > 0.2*dBd) else 0

            QN_d_k = d_k[:nx]

            QN.update(QN_d_k, w_k)

            # # <<<<<<<<<<<<<<<<<<<
            # # ALGORITHM ENDS HERE

            opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
            if self.nc == 0:
                feas_satisfied, feas, sviol, mviol = True, 0., 0., 0.
            else:
                feas_satisfied, feas, sviol, mviol = self.feas_check(x_k, c_k)
            tol_satisfied = (opt_satisfied and feas_satisfied)

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(
                major=itr,
                x=x_k,
                lag_mult=pi_k,
                slacks=s_k,
                obj=f_k,
                constraints=c_k,
                opt=opt,
                feas=feas,
                sum_viol=sviol,
                max_viol=mviol,
                time=time.time() - start_time,
                nfev=nfev,
                ngev=ngev,
                # rho=rho_k[0],
                rho=np.linalg.norm(rho_k),
                step=alpha,
                # step=0.5**ls_count,
                merit=mf_k,
                elastic=(elastic_itr>0),
                gamma=gamma,
                low_curvature=low_curvature,)
                # merit=penalty_k)
            if tol_satisfied:
                print('Convergence achieved!')
                break

        self.total_time = time.time() - start_time

        self.results = {
            'x': x_k,
            'objective': f_k,
            'c': c_k,
            'pi': pi_k,
            'optimality': opt,
            'feasibility': feas,
            'nfev': nfev,
            'ngev': ngev,
            'niter': itr,
            'time': self.total_time,
            'success': tol_satisfied
        }

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results