import numpy as np
import scipy as sp
import time

from modopt import Optimizer
from modopt.line_search_algorithms import BacktrackingArmijo
from modopt.merit_functions import L1Eq

from modopt.approximate_hessians import BFGSScipy as BFGS

epsmch = np.finfo(np.float64).eps  # 2.22e-16

# This optimizer considers equality and inequality constraints separately
class InteriorPoint(Optimizer):
    """
    An Interior Point algorithm for nonlinearly constrained optimization problems.
    
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
    opt_tol : float, default=1e-6
        Optimality tolerance.
    feas_tol : float, default=1e-6
        Feasibility tolerance.
    fraction_to_boundary : float, default=0.99
        Fraction to the boundary for step size calculation.
    init_barrier_param : float, default=0.50
        Initial value of the barrier parameter.
    barrier_reduction_factor : float, default=0.20
        Factor by which the barrier parameter is reduced
        when convergence tolerances are met for the current barrier problem.
    barrier_reduction_power : float, default=1.50
        Power to which the barrier parameter is raised
        when convergence tolerances are met for the current barrier problem.
    barrier_problem_tolerance_factor : float, default=10.0
        Convergence tolerance for each barrier problem set as
        a factor of the current barrier parameter.

    ls_maxiter : int, default=10
        Maximum number of line search iterations.
    ls_eta_a : float, default=1e-4
        Armijo parameter for the line search.
    ls_gamma_c : float, default=0.5
        Step length contraction factor for backtracking line search.

    bfgs_reset_frequency : int, default=100
        Iteration frequency to reset the BFGS approximate Hessian.
    
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'major', 'obj', 'x', 'lag_mult', 'slacks', 'constraints', 'opt',
        'feas', 'time', 'nfev', 'ngev', 'step', 'rho', 'merit'.
    """
    def initialize(self):
        self.solver_name = 'interior_point'

        self.nx = self.problem.nx

        self.obj  = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.active_callbacks = ['obj', 'grad']
        if self.problem.constrained:
            self.con_in = self.problem._compute_constraints
            self.jac_in = self.problem._compute_constraint_jacobian

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-6, types=float)
        self.options.declare('feas_tol', default=1e-6, types=float)
        self.options.declare('fraction_to_boundary', default=0.99, types=float)
        self.options.declare('initial_barrier_parameter', default=0.50, types=float)
        self.options.declare('barrier_reduction_factor',  default=0.20, types=float)
        self.options.declare('barrier_reduction_power',   default=1.50, types=float)
        self.options.declare('barrier_problem_tolerance_factor', default=10.0, types=float)
        self.options.declare('readable_outputs', types=list, default=[])

        self.options.declare('ls_maxiter', default=10, types=int)
        self.options.declare('ls_eta_a', default=1e-4, types=float)
        self.options.declare('ls_gamma_c', default=0.5, types=float)

        self.options.declare('bfgs_reset_frequency', default=100, types=int)

        self.available_outputs = {
            'major': int,
            'obj': float,
            # For array outputs, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            # Number of constraints will be updated after setup_constraints()
            'lag_mult': (float, (self.problem.nc, )),
            'slacks': (float, (self.problem.nc, )),
            'constraints': (float, (self.problem.nc, )),
            'mu': float,
            'opt': float,
            'feas': float,
            'time': float,
            'nfev': int,
            'ngev': int,
            'step': float,
            'rho': float,
            'merit': float,
        }

        self.successive_kkt_fails = 0
        self.successive_undefined_steps = 0
        self.successive_undefined_directions = 0
        self.successive_skipped_BFGS_updates = 0
 
    def setup(self):
        self.setup_constraints()
        nx   = self.nx
        nc_i = self.nc_i
        nc   = self.nc
        self.mu = mu = self.options['initial_barrier_parameter'] if nc_i > 0 else 0.
        self.available_outputs['lag_mult'] = (float, (nc, ))
        self.available_outputs['rho'] = (float, (nc, ))
        self.available_outputs['slacks'] = (float, (nc, ))
        self.available_outputs['constraints'] = (float, (nc, ))

        self.QN = BFGS(nx=nx,
                       init_scale=1.0,
                       min_curvature=0.25,
                       exception_strategy='damp_update')
    
    # Adapt constraints to c_e(x)=0, and c_i(x) >= 0
    def setup_constraints(self, ):
        xl = self.problem.x_lower
        xu = self.problem.x_upper

        ebi = self.eq_bound_indices    = np.array([]) # Equality bounds on x is disabled
        lbi = self.lower_bound_indices = np.where(xl != -np.inf)[0]
        ubi = self.upper_bound_indices = np.where(xu !=  np.inf)[0]

        self.eq_bounded    = True if len(ebi) > 0 else False # Always False
        self.lower_bounded = True if len(lbi) > 0 else False
        self.upper_bounded = True if len(ubi) > 0 else False

        if self.problem.constrained:
            cl = self.problem.c_lower
            cu = self.problem.c_upper

            eci = self.eq_constraint_indices    = np.where(cl == cu)[0]
            lci = self.lower_constraint_indices = np.where((cl != -np.inf) & (cl != cu))[0]
            uci = self.upper_constraint_indices = np.where((cu !=  np.inf) & (cl != cu))[0]
        else:
            eci = np.array([], dtype=int)
            lci = np.array([], dtype=int)
            uci = np.array([], dtype=int)
        
        self.eq_constrained    = True if len(eci) > 0 else False
        self.lower_constrained = True if len(lci) > 0 else False
        self.upper_constrained = True if len(uci) > 0 else False

        self.nc_e = len(ebi) + len(eci)
        self.nc_b = len(lbi) + len(ubi)
        self.nc_i = len(lbi) + len(ubi) + len(lci) + len(uci)
        self.nc   = self.nc_e + self.nc_i

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

        c_e = np.array([], dtype=float)
        if self.eq_bounded:
            c_e = np.append(c_e, x[ebi] - self.problem.x_lower[ebi])
        if self.eq_constrained:
            c_e = np.append(c_e, c_in[eci] - self.problem.c_lower[eci])

        c_i = np.array([], dtype=float)
        if self.lower_bounded:
            c_i = np.append(c_i, x[lbi] - self.problem.x_lower[lbi])
        if self.upper_bounded:
            c_i = np.append(c_i, self.problem.x_upper[ubi] - x[ubi])
        if self.lower_constrained:
            c_i = np.append(c_i, c_in[lci] - self.problem.c_lower[lci])
        if self.upper_constrained:
            c_i = np.append(c_i, self.problem.c_upper[uci] - c_in[uci])

        return np.concatenate((c_e, c_i))

    def jac(self, x):
        nx  = self.nx
        # ebi = self.eq_bound_indices
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices

        if self.problem.constrained:
            eci = self.eq_constraint_indices
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Compute problem constraint Jacobian
            j_in = self.jac_in(x)

        j_e = np.empty((0, nx), dtype=float)
        # if self.eq_bounded:
        #     j_e = np.append(j_e, np.identity(nx)[ebi], axis=0)
        if self.eq_constrained:
            j_e = np.append(j_e, j_in[eci], axis=0)

        j_i = np.empty((0, nx), dtype=float)
        if self.lower_bounded:
            j_i = np.append(j_i,  np.identity(nx)[lbi], axis=0)
        if self.upper_bounded:
            j_i = np.append(j_i, -np.identity(nx)[ubi], axis=0)
        if self.lower_constrained:
            j_i = np.append(j_i,  j_in[lci], axis=0)
        if self.upper_constrained:
            j_i = np.append(j_i, -j_in[uci], axis=0)

        return np.vstack((j_e, j_i))
    
    def backtrack_to_avoid_nans_infs(self, x_k, p_x):
        empty = np.array([])
        funcs = {'f': [self.obj,  empty, 0, False],
                 'g': [self.grad, empty, 0, False],
                 'c': [self.con, empty, 0, False],
                 'j': [self.jac, empty.reshape(0,0), 0, False]}
        alpha = 1.0
        for key, func in funcs.items():
            while alpha > 1e-12-2.22e-15:
                func[1]  = func[0](x_k+alpha*p_x)
                func[2] += 1
                if np.isnan(func[1]).any() or np.isinf(func[1]).any():
                    alpha *= 1e-1
                else:
                    func[3] = True
                    break
        
        new_f_evals = funcs['f'][2] + funcs['c'][2] - 1
        new_g_evals = funcs['g'][2] + funcs['j'][2] - 1
        success     = all([funcs[key][3] for key in funcs])
        max_step    = alpha * 1.0  

        return success, max_step, new_f_evals, new_g_evals, funcs['f'][1], funcs['g'][1], funcs['c'][1], funcs['j'][1]

    def get_results_dict(self, x_k, f_k, c_k, pi_k, s_k, opt, feas, nfev, ngev, niter, time, success):
        results = {'x': x_k,
                   'objective': f_k,
                   'c': c_k,
                   'pi': pi_k,
                   's': s_k,
                   'optimality': opt,
                   'feasibility': feas,
                   'nfev': nfev,
                   'ngev': ngev,
                   'niter': niter,
                   'time': time,
                   'success': success}
        return results

    def solve(self):
        # Assign shorter names to variables and methods
        nx  = self.nx
        nce = self.nc_e
        ncb = self.nc_b
        nci = self.nc_i
        nc  = self.nc

        x0 = self.problem.x0
        maxiter = self.options['maxiter']
        tau_min = self.options['fraction_to_boundary']
        mu_red_factor = self.options['barrier_reduction_factor']
        mu_red_power  = self.options['barrier_reduction_power']
        bp_tol_factor = self.options['barrier_problem_tolerance_factor']

        # Barrier parameter
        mu = self.mu

        obj  = self.obj
        grad = self.grad
        con_temp = self.con
        jac_temp = self.jac

        QN = self.QN
        start_time = time.time()

        # Initialize the starting point
        move_by = 1e-1 * (self.problem.x_upper - self.problem.x_lower)
        move_by = np.minimum(move_by, 1e-1)
        x_k = np.clip(x0, self.problem.x_lower+move_by, self.problem.x_upper-move_by)
        pi_k = np.full(( nc,), 1.)
        s_k  = np.full((nci,), 1.)

        # Vector of design vars., lag. mults., and slack vars.
        v_k   = np.concatenate((x_k, pi_k, s_k))
        x_k   = v_k[:nx]
        pi_k  = v_k[nx:(nx + nc)]
        lam_k = v_k[nx:(nx + nce)]        # Equality constraint multipliers
        sig_k = v_k[(nx + nce):(nx + nc)] # Inequality constraint multipliers
        s_k   = v_k[(nx + nc):]

        p_k   = np.zeros((len(v_k), ))
        p_x   = p_k[:nx]
        p_pi  = p_k[nx:(nx + nc)]
        p_lam = p_k[nx:(nx + nce)]
        p_sig = p_k[(nx + nce):(nx + nc)]
        p_s   = p_k[(nx + nc):]

        f_k = obj(x_k)
        g_k = grad(x_k)
        c_k = con_temp(x_k)
        J_k = jac_temp(x_k)

        nfev = 1
        ngev = 1

        # Reset initial point if proximal point functions are undefined
        if (x_k != x0).any():
            if (np.isnan(f_k) or np.isinf(f_k) or
                np.isnan(g_k).any() or np.isinf(g_k).any() or
                np.isnan(c_k).any() or np.isinf(c_k).any() or
                np.isnan(J_k).any() or np.isinf(J_k).any()):

                x_k = x0 * 1.
                f_k = obj(x_k)
                g_k = grad(x_k)
                c_k = con_temp(x_k)
                J_k = jac_temp(x_k)

                nfev += 1
                ngev += 1

        if (np.isnan(f_k) or np.isinf(f_k) or
            np.isnan(g_k).any() or np.isinf(g_k).any() or
            np.isnan(c_k).any() or np.isinf(c_k).any() or
            np.isnan(J_k).any() or np.isinf(J_k).any()):
            print('Objective or constraint functions undefined at initial and proximal points. Exiting.')

            return self.get_results_dict(x_k, f_k, c_k, None, None, None, None, 2, 2, 0, time.time() - start_time, False)

        # Scale objective and constraint functions to improve performance
        o_scaler = min(1., 1./max(np.linalg.norm(g_k, ord=np.inf), 1e-12))
        o_scaler = max(o_scaler, 1e-10)
        
        f_k *= o_scaler
        g_k *= o_scaler
        obj  = self.obj  = lambda x: o_scaler * self.problem._compute_objective(x)
        grad = self.grad = lambda x: o_scaler * self.problem._compute_objective_gradient(x)

        if nc > ncb:
            nbce = non_bound_con_indices = np.concatenate((np.arange(nce, dtype=int), np.arange(nce+ncb, nc, dtype=int)))
            c_scaler = np.array([min(1., 1./max(np.linalg.norm(J_k[i], ord=np.inf), 1e-12)) for i in nbce])
            c_scaler = np.maximum(c_scaler, 1e-10)
            c_scaler = np.concatenate((c_scaler[:nce], np.ones((ncb, )), c_scaler[nce:]))
            c_k *= c_scaler
            J_k *= c_scaler[:, None]
    
            con = self.con = lambda x: c_scaler * con_temp(x)
            jac = self.jac = lambda x: jac_temp(x) * c_scaler[:, None]

        else:
            c_scaler = np.ones((nc, ))
            con = self.con
            jac = self.jac

        self.bsp_obj  = lambda xs: self.obj(xs[:nx]) - self.mu * np.sum(np.log(xs[nx:]))
        self.bsp_grad = lambda xs: np.concatenate((self.grad(xs[:nx]),
                                                  -self.mu / xs[nx:]))
        self.bsp_con  = lambda xs: self.con(xs[:nx]) - \
                                  np.concatenate((np.zeros((nce, )), xs[nx:]))
        self.bsp_jac  = lambda xs: np.hstack((self.jac(xs[:nx]),
                                             np.vstack((np.zeros((nce, nci)), 
                                                        -np.eye(nci)))))
        
        MF = self.MF = L1Eq(nx=nx+nci,
                       nc=nc,
                       f=self.bsp_obj,
                       c=self.bsp_con,
                       g=self.bsp_grad,
                       j=self.bsp_jac)
        
        LSB = self.LSB = BacktrackingArmijo(f=self.MF.compute_function,
                                      g=self.MF.compute_gradient,
                                      maxiter=self.options['ls_maxiter'],
                                      gamma_c=self.options['ls_gamma_c'],
                                      eta_a=self.options['ls_eta_a'])

        # Update initial slacks
        s_k[:] = np.maximum(1e-1, c_k[nce:]+1e-1)

        # Form the barrier subproblem variables
        bsp_xk = np.concatenate((x_k, s_k))
        bsp_fk = f_k - mu * np.sum(np.log(s_k))
        bsp_ck = c_k - np.concatenate((np.zeros((nce,)), s_k))
        bsp_gk = np.concatenate((g_k, -mu / s_k))
        bsp_Jk = np.hstack((J_k, np.vstack((np.zeros((nce, nci)), -np.eye(nci)))))

        # Vector of penalty parameters
        rho_k = np.full((nc,), 0.0)

        # Iteration counter
        itr = 0

        # Evaluate merit function value
        MF.set_rho(rho_k)
        mf_k  = MF.evaluate_function(bsp_xk, bsp_fk, bsp_ck)
        mfg_k = MF.evaluate_gradient(bsp_xk, bsp_fk, bsp_ck, bsp_gk, bsp_Jk)

        # Compute optimality and feasibility measures
        lag_grad = g_k - J_k.T @ pi_k
        compl    = s_k * sig_k if nci > 0 else np.array([0.0])
        opt      = max(np.linalg.norm(lag_grad, ord=np.inf),
                       np.linalg.norm(compl, ord=np.inf))
        c_viol = np.concatenate((np.abs(c_k[:nce]), np.maximum(0., -c_k[nce:]))) if nc > 0 else np.array([0.0])
        tol_satisfied = False

        # Initializing declared outputs
        self.update_outputs(major=0,
                            x=x_k,
                            lag_mult=pi_k*c_scaler/o_scaler,
                            mu=mu,
                            slacks=s_k/c_scaler[nce:],
                            obj=f_k/o_scaler,
                            constraints=c_k/c_scaler,
                            opt=opt,
                            feas=np.linalg.norm(c_viol/c_scaler, np.inf) if nc > 0 else 0.,
                            time=time.time() - start_time,
                            nfev=nfev,
                            ngev=ngev,
                            step=0.,
                            rho=rho_k,
                            merit=mf_k)
        
        while (not (tol_satisfied) and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate
            ce_k = c_k[:nce]
            ci_k = c_k[nce:]
            r_i  = ci_k - s_k

            Je_k = J_k[:nce]
            Ji_k = J_k[nce:]
            W = sig_k / s_k

            # Form the reduced KKT system for the barrier iteration
            KKT = np.block([[QN.B_k + Ji_k.T@(W[:, None]*Ji_k),   Je_k.T],
                            [Je_k,                  np.zeros((nce, nce))]])
            
            rhs = -np.concatenate((g_k + Ji_k.T@(W*r_i - mu/s_k), c_k[:nce]))

            # Solve the KKT system
            solved = False
            try:
                kkt_sol = np.linalg.solve(KKT, rhs)
                solved = True
            except np.linalg.LinAlgError:
                print (f'KKT system solve failed. Applying regularization.')
                # KKT[      :nx,      :nx] += 1e-6 * np.eye(nx)
                KKT[nx:nx+nce,nx:nx+nce] -= 1e-6 * np.eye(nce)
                try:
                    kkt_sol = np.linalg.solve(KKT, rhs)
                    solved = True
                except np.linalg.LinAlgError:
                    print('KKT system solve failed even after regularization.')

            if not solved or np.isnan(kkt_sol).any() or np.isinf(kkt_sol).any():
                # KKT[nx:nx+nce,nx:nx+nce] += 1e-6 * np.eye(nce)
                try:
                    kkt_sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
                    print('KKT system solve failed. Using least-squares solution.')

                except np.linalg.LinAlgError:
                    self.successive_kkt_fails += 1
                    if self.successive_kkt_fails >= 2:
                        print('KKT system least-squares solve also failed in 2 successive iterations. Stopping optimization.')
                        break
                    else:
                        print('KKT system least-squares solve also failed. Resetting Hessian and continuing to next iteration.')
                        # Reset Hessian
                        self.QN = QN = BFGS(nx=nx,
                                            exception_strategy='damp_update',
                                            init_scale=1.0)
                        continue
            
            self.successive_kkt_fails = 0

            # Search direction for x_k:
            p_k[:nx] = kkt_sol[:nx]

            # Search direction for lam_k:
            p_k[nx:(nx + nce)] = -kkt_sol[nx:] - lam_k

            # Search direction for s_k :
            p_k[(nx + nc):] = r_i + Ji_k @ p_x

            # Append -sigma to kkt_sol for remaining calculations
            kkt_sol = np.concatenate((kkt_sol, -(mu / s_k - W * p_s)))

            # Search direction for sig_k:
            p_k[nx+nce:nx+nc] = -kkt_sol[nx+nce:] - sig_k

            if np.isnan(p_k).any() or np.isinf(p_k).any():
                self.successive_undefined_directions += 1
                if self.successive_undefined_directions >= 2:
                    print('Search direction contains NaN or Inf in 2 successive iterations. Stopping optimization.')
                    break
                else:
                    print('Search direction contains NaN or Inf. Resetting Hessian and continuing to next iteration.')
                    # Reset Hessian
                    self.QN = QN = BFGS(nx=nx,
                                        exception_strategy='damp_update',
                                        init_scale=1.0)
                    continue
            
            self.successive_undefined_directions = 0

            # Compute the maximum step length along the search direction
            tau = tau_min * 1.0
            if nci > 0:
                idx = p_s < 0
                alpha_s_max   = np.min(np.append(-tau * (s_k[idx]/p_s[idx]), 1.0))
                idx = p_sig < 0
                alpha_sig_max = np.min(np.append(-tau * (sig_k[idx]/p_sig[idx]), 1.0))
                alpha_max_min = min(alpha_s_max, alpha_sig_max)

                p_k[:] = np.concatenate((alpha_s_max * p_x,
                                        #  alpha_sig_max * p_pi,
                                         alpha_max_min * p_pi,
                                         alpha_s_max * p_s))
                
            # Form the barrier subproblem variables
            bsp_xk = np.concatenate((x_k, s_k))
            bsp_fk = f_k - mu * np.sum(np.log(s_k))
            bsp_ck = c_k - np.concatenate((np.zeros((nce,)), s_k))
            bsp_gk = np.concatenate((g_k, -mu / s_k))
            bsp_Jk = np.hstack((J_k, np.vstack((np.zeros((nce, nci)), -np.eye(nci)))))

            g_old = g_k * 1.
            J_old = J_k * 1.

            # Backtracking line search to avoid NaNs and Infs
            success, max_step, new_f_evals, new_g_evals, f_, g_, c_, J_ = self.backtrack_to_avoid_nans_infs(x_k, p_x)
            # TODO: Correct when caching is implemented in MF
            nfev = nfev + new_f_evals - 1
            ngev = ngev + new_g_evals - 1

            if not success:
                self.successive_undefined_steps += 1
                if self.successive_undefined_steps >= 2:
                    print('Could not find a step that avoids NaNs or Infs in 2 successive iterations. Stopping optimization.')
                    break
                else:
                    print('Step contains NaNs or Infs. Resetting Hessian and continuing to next iteration.')
                    # Reset Hessian
                    self.QN = QN = BFGS(nx=nx,
                                        exception_strategy='damp_update',
                                        init_scale=1.0)
                    continue
                
            self.successive_undefined_steps = 0
            p_k *= max_step

            bsp_pk = np.concatenate((p_x, p_s))

            # Update penalty parameters
            if nc > 0:
                rho_k = np.maximum(np.abs(-kkt_sol[nx:])+2e-6, 0.1*rho_k + 0.9*np.abs(-kkt_sol[nx:]))

            MF.set_rho(rho_k)
            mf_k  = MF.evaluate_function(bsp_xk, bsp_fk, bsp_ck)
            mfg_k = MF.evaluate_gradient(bsp_xk, bsp_fk, bsp_ck, bsp_gk, bsp_Jk)

            # Perform line search to enforce sufficient decrease in merit function
            alpha, mf_new, new_f_evals, new_g_evals, converged = LSB.search(
                x=bsp_xk, p=bsp_pk, f0=mf_k, g0=mfg_k)
            
            # TODO: Correct when caching is implemented in MF
            new_g_evals = 1

            nfev += new_f_evals
            ngev += new_g_evals
            alpha = alpha if converged else 0.9995 # Take nearly full step if LS fails
            d_k = p_k * alpha

            # Accept full step if tiny step size from line search
            idx = np.abs(d_k)<2.22045*1e-15
            d_k[idx] = p_k[idx]

            v_k += d_k

            f_k = obj(x_k)
            g_k = grad(x_k)
            c_k = con(x_k)
            J_k = jac(x_k)

            # TODO: Correct when caching is implemented in MF
            # nfev += 1
            # ngev += 1

            if not converged:
                print("Major Iteration:", itr)
                print("  Line search did not converge. Step size used:", alpha)

            # Update the Hessian approximation using the BFGS formula
            w_k = (g_k - g_old) - (J_k - J_old).T @ pi_k
            QN.update(d_k[:nx], w_k)

            if itr % self.options['bfgs_reset_frequency'] == 0:
                QN = self.QN = BFGS(nx=nx,
                                    exception_strategy='damp_update',
                                    min_curvature=0.2,
                                    init_scale='auto')
                # if not (w_k==0).all():
                QN.update(d_k[:nx], w_k)
            
            # Update the barrier parameter, mu
            opt_scaler   = max(1., np.linalg.norm( pi_k, ord=1)/ nc)/1. if nc  > 0 else 1.0
            compl_scaler = max(1., np.linalg.norm(sig_k, ord=1)/nci)/1. if nci > 0 else 1.0
            lag_grad = g_k
            feas = compl = np.array([0.0])
            if nci > 0:
                compl    = s_k * sig_k - mu
            if nc > 0:
                lag_grad = g_k - J_k.T @ pi_k
                feas     = np.concatenate((c_k[:nce], c_k[nce:] - s_k))
            opt = np.max([np.linalg.norm(lag_grad, np.inf)/opt_scaler,
                          np.linalg.norm(compl,    np.inf)/compl_scaler,
                          np.linalg.norm(feas,     np.inf)])
            
            while opt < (mu*bp_tol_factor):
                self.mu = mu = min(mu*mu_red_factor, mu**mu_red_power)

            # Prevent mu from going below a threshold
            if nci > 0:
                self.mu = mu = max(2.22045e-14, mu)

            opt_scaler   = max(100., np.linalg.norm( pi_k, ord=1)/ nc)/100. if nc  > 0 else 1.0
            compl_scaler = max(100., np.linalg.norm(sig_k, ord=1)/nci)/100. if nci > 0 else 1.0

            compl = s_k * sig_k if nci > 0 else np.array([0.0])
            opt_satisfied  = (np.linalg.norm(lag_grad, np.inf)/opt_scaler <= self.options['opt_tol']) \
                         and (np.linalg.norm(compl, np.inf)/compl_scaler  <= self.options['opt_tol'])
            c_viol = np.concatenate((np.abs(c_k[:nce]), np.maximum(0., -c_k[nce:]))) if nc > 0 else np.array([0.0])
            feas_satisfied = (np.linalg.norm(c_viol, np.inf) <= self.options['feas_tol'])
            tol_satisfied  = (opt_satisfied and feas_satisfied)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(
                major=itr,
                x=x_k,
                mu=mu,
                lag_mult=pi_k*c_scaler/o_scaler,
                slacks=s_k/c_scaler[nce:],
                obj=f_k/o_scaler,
                constraints=c_k/c_scaler,
                opt=opt,
                feas=np.linalg.norm(c_viol/c_scaler, np.inf) if nc > 0 else 0.,
                time=time.time() - start_time,
                nfev=nfev,
                ngev=ngev,
                rho=rho_k,
                step=alpha,
                merit=mf_k)
            
        if tol_satisfied:
            print('Optimization converged successfully!')
        elif itr >= maxiter:
            print(f'Maximum number of iterations ({maxiter}) reached.')

        self.total_time = time.time() - start_time

        self.results = {
            'x': x_k,
            'objective': f_k/o_scaler,
            'c': c_k/c_scaler,
            'pi': pi_k*c_scaler/o_scaler,
            's': s_k/c_scaler[nce:],
            'optimality': opt,
            'feasibility': np.linalg.norm(c_viol/c_scaler, np.inf) if nc > 0 else 0.,
            'nfev': nfev,
            'ngev': ngev,
            'niter': itr,
            'time': self.total_time,
            'success': tol_satisfied
        }

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results