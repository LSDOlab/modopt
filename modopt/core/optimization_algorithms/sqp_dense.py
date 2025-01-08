import numpy as np
from scipy.sparse.csc import csc_matrix

import scipy.sparse as sp
import time

from modopt import Optimizer
from modopt.line_search_algorithms import ScipyLS, BacktrackingArmijo, Minpack2LS
from modopt.merit_functions import AugmentedLagrangianIneq
from modopt.approximate_hessians import BFGSScipy as BFGS

# This optimizer takes constraints in all-inequality form, C(x) >= 0
class SQP(Optimizer):
    def initialize(self):
        self.solver_name = 'sqp'

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
        self.options.declare('qp_tol', default=1e-4, types=float)
        self.options.declare('readable_outputs', types=list, default=[])

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
            'time': float,
            'nfev': int,
            'ngev': int,
            'step': float,
            'rho': float,
            'merit': float,
        }

    def setup(self):
        self.setup_constraints()
        nx = self.nx
        nc = self.nc
        self.available_outputs['lag_mult'] = (float, (nc, ))
        self.available_outputs['rho'] = (float, (nc, ))
        self.available_outputs['slacks'] = (float, (nc, ))
        self.available_outputs['constraints'] = (float, (nc, ))
        # Damping parameters
        self.delta_rho = 1.
        self.num_rho_changes = 0
        self.num_consecutive_ls_fails = 0

        self.QN = BFGS(nx=nx,
                       exception_strategy='damp_update')
        self.MF = AugmentedLagrangianIneq(nx=nx,
                                          nc=nc,
                                          f=self.obj,
                                          c=self.con,
                                          g=self.grad,
                                          j=self.jac)
        # self.LSS = ScipyLS(f=self.MF.compute_function,
        #                    g=self.MF.compute_gradient,
        #                    max_step=1.,
        #                    maxiter=8,
        #                    eta_w=0.9)
        
        self.LSS = Minpack2LS(f=self.MF.compute_function,
                              g=self.MF.compute_gradient,
                              min_step=1e-14,
                              max_step=1.,
                              maxiter=10,
                              alpha_tol=1e-14,
                              eta_w=0.9)
        
        self.LSB = BacktrackingArmijo(f=self.MF.compute_function,
                                      g=self.MF.compute_gradient,
                                      gamma_c=0.3,
                                      max_step=1.0,
                                      maxiter=25)

    # Adapt constraints to C(x) >= 0
    def setup_constraints(self, ):

        xl = self.problem.x_lower
        xu = self.problem.x_upper

        # Adapt bounds as ineq constraints C(x) >= 0
        # Remove bounds with -np.inf as lower bound
        lbi = self.lower_bound_indices = np.where(xl != -np.inf)[0]
        # Remove bounds with np.inf as upper bound
        ubi = self.upper_bound_indices = np.where(xu !=  np.inf)[0]

        self.lower_bounded = True if len(lbi) > 0 else False
        self.upper_bounded = True if len(ubi) > 0 else False
        
        # Adapt eq/ineq constraints as constraints >= 0
        # Remove constraints with -np.inf as lower bound
        if self.problem.constrained:
            cl = self.problem.c_lower
            cu = self.problem.c_upper
            lci = self.lower_constraint_indices = np.where(cl != -np.inf)[0]
            # Remove constraints with np.inf as upper bound
            uci = self.upper_constraint_indices = np.where(cu !=  np.inf)[0]
        else:
            lci = np.array([])
            uci = np.array([])

        self.lower_constrained = True if len(lci) > 0 else False
        self.upper_constrained = True if len(uci) > 0 else False

        self.nc = len(lbi) + len(ubi) + len(lci) + len(uci)

    def con(self, x):
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices
        if self.problem.constrained:
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Compute problem constraints
            c_in = self.con_in(x)

        c_out = np.array([])

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
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices
        if self.problem.constrained:
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Compute problem constraint Jacobian
            j_in = self.jac_in(x)

        j_out = np.empty((1, nx), dtype=float)

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
            # s[rho_zero_indices] = 0.
            s[rho_zero_indices] = np.maximum(0, c[rho_zero_indices])

        if len(rho_nonzero_indices) > 0:
            c_rnz   = c[rho_nonzero_indices]
            pi_rnz  = pi[rho_nonzero_indices]
            rho_rnz = rho[rho_nonzero_indices]
            s[rho_nonzero_indices] = np.maximum(0, (c_rnz - pi_rnz / rho_rnz))

        return s

    def update_scalar_rho(self, rho_k, dir_deriv_al, pTHp, p_pi, c_k, s_k):
        # Damping for scalar rho as in SNOPT
        delta_rho = self.delta_rho
        rho_ref   = rho_k[0] * 1.

        if dir_deriv_al <= -0.5 * pTHp:
            rho_computed = rho_k[0]
        else:
            # note: scalar rho_min here
            rho_min = 2 * np.linalg.norm(p_pi) / np.linalg.norm(c_k - s_k)
            rho_computed = max(rho_min, 2 * rho_k[0])

        # Damping for rho (Note: vector rho is still not updated)
        if rho_k[0] < 4 * (rho_computed + delta_rho):
            rho_damped = rho_k[0]
        else:
            rho_damped = np.sqrt(rho_k[0] * (rho_computed + delta_rho))

        rho_new = max(rho_computed, rho_damped)
        rho_k[:] = rho_new

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

        rho_ref = np.linalg.norm(rho_k)

        # ck_sk =
        a = -np.power(c_k - s_k, 2)
        b = -0.5 * pTHp - dir_deriv_al + np.inner(a, rho_k)

        well_defined_a = True
        # print('a*sgn(b)', a * np.sign(b))

        if b > 0:
            a_plus = np.maximum(a, 0)
            a_plus_norm = np.linalg.norm(a_plus)
            if a_plus_norm == 0:
                well_defined_a = False
            else:
                rho_computed = (b / (a_plus_norm**2)) * a_plus
        elif b < 0:
            a_minus = -np.minimum(a, 0)
            a_minus_norm = np.linalg.norm(a_minus)
            if a_minus_norm == 0:
                well_defined_a = False
            else:
                rho_computed = -(b / (a_minus_norm**2)) * a_minus

        # TODO: check the condition below
        else:  # When b = 0
            rho_computed = np.zeros((len(rho_k), ))

        if not (well_defined_a):  # Perform a scalar rho update if a = 0
            if dir_deriv_al <= -0.5 * pTHp:
                rho_computed = rho_k * 1.
            else:
                # note: scalar rho_min here
                rho_min = 2 * np.linalg.norm(p_pi) / np.linalg.norm(c_k - s_k)
                rho_computed = np.maximum(2 * rho_k, rho_min)

        # Damping for rho (Note: vector rho is still not updated)
        rho_damped = np.where(
            rho_k < 4 * (rho_computed + delta_rho), rho_k,
            np.sqrt(rho_k * (rho_computed + delta_rho)))

        rho_new = np.maximum(rho_computed, rho_damped)
        rho_k[:] = rho_new

        rho_norm = np.linalg.norm(rho_k)
        # Increasing rho
        if rho_norm > rho_ref:
            if self.num_rho_changes >= 0:
                self.num_rho_changes += 1
            else:
                self.delta_rho *= 2.
                self.num_rho_changes = 0

        # Decreasing rho
        elif rho_norm < rho_ref:
            if self.num_rho_changes <= 0:
                self.num_rho_changes -= 1
            else:
                self.delta_rho *= 2.
                self.num_rho_changes = 0

        return rho_k

    def opt_check(self, pi, c, g, j):
        opt_tol = self.options['opt_tol']

        if self.nc == 0:
            opt_tol_factor = 1.
            nonneg_violation = 0.
            compl_violation = 0.
            stat_violation = np.linalg.norm(g, np.inf)

        else:
            opt_tol_factor = (1 + np.linalg.norm(pi, np.inf))
            nonneg_violation = max(0., -np.min(pi))
            # Note: compl_violation can be negative(Question: SNOPT paper);
            #       Other 2 violations are always nonnegative
            compl_violation = np.max(c * pi)
            stat_violation = np.linalg.norm(g - j.T @ pi, np.inf)

        scaled_opt_tol = opt_tol * opt_tol_factor
        opt_check1 = (nonneg_violation <= scaled_opt_tol)
        opt_check2 = (compl_violation <= scaled_opt_tol)
        opt_check3 = (stat_violation <= scaled_opt_tol)

        # opt is always nonnegative
        opt = max(nonneg_violation, compl_violation,
                  stat_violation) / opt_tol_factor
        opt_satisfied = (opt_check1 and opt_check2 and opt_check3)

        return opt_satisfied, opt

    def feas_check(self, x, c):
        feas_tol = self.options['feas_tol']

        feas_tol_factor = (1 + np.linalg.norm(x, np.inf))
        scaled_feas_tol = feas_tol * feas_tol_factor

        # Violation is positive
        max_con_violation = max(0., -np.min(c))
        feas_check = (max_con_violation <= scaled_feas_tol)

        feas = max_con_violation / feas_tol_factor
        feas_satisfied = feas_check
        return feas_satisfied, feas

    def solve(self):
        try:
            import osqp
        except ImportError:
            raise ImportError("OSQP cannot be imported for the SQP solver. Install it with 'pip install osqp'.")
        
        # Assign shorter names to variables and methods
        nx = self.nx
        nc = self.nc

        x0 = self.problem.x0
        maxiter = self.options['maxiter']
        qp_tol  = self.options['qp_tol']

        obj  = self.obj
        grad = self.grad
        con  = self.con
        jac  = self.jac

        LSS = self.LSS
        LSB = self.LSB
        QN = self.QN
        MF = self.MF

        start_time = time.time()

        # Set initial values for current iterates
        x_k  = x0 * 1.              # Initial optimization variables
        pi_k = np.full((nc, ), 1.)  # Initial Lagrange multipliers

        f_k = obj(x_k)
        g_k = grad(x_k)
        c_k = con(x_k)
        J_k = jac(x_k)

        nfev = 1
        ngev = 1

        B_k   = np.identity(nx)     # Initial Hessian approximation
        rho_k = np.full((nc, ), 0.) # Initial penalty parameter vector

        # Compute slacks by minimizing A.L. s.t. s_k >= 0
        s_k = self.reset_slacks(c_k, pi_k, rho_k)
        # s_k = np.maximum(0, c_k) # Use this if rho_k = 0. at start

        # Vector of design vars., lag. mults., and slack vars.
        v_k = np.concatenate((x_k, pi_k, s_k))
        x_k = v_k[:nx]
        pi_k = v_k[nx:(nx + nc)]
        s_k = v_k[(nx + nc):]

        p_k = np.zeros((len(v_k), ))
        p_x = p_k[:nx]
        p_pi = p_k[nx:(nx + nc)]
        p_s = p_k[(nx + nc):]

        # QP parameters
        l_k = -c_k
        u_k = np.full((nc, ), np.inf)
        A_k = J_k

        # Iteration counter
        itr = 0

        opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
        if self.nc == 0:
            feas_satisfied, feas = True, 0.
        else:
            feas_satisfied, feas = self.feas_check(x_k, c_k)
        
        tol_satisfied = (opt_satisfied and feas_satisfied)

        # Evaluate merit function value
        MF.set_rho(rho_k)
        mf_k = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
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
                            time=time.time() - start_time,
                            nfev=nfev,
                            ngev=ngev,
                            step=0.,
                            rho=rho_k,
                            merit=mf_k)

        # Create scipy csc_matrix for Hk
        Bk_rows = np.triu(np.outer(np.arange(1, nx + 1),
                                   np.ones(nx, dtype='int'))).flatten('f')
        Bk_rows = Bk_rows[Bk_rows != 0] - 1
        Bk_cols = np.repeat(np.arange(nx), np.arange(1, nx + 1))
        Bk_ind_ptr = np.insert(np.cumsum(np.arange(1, nx + 1)), 0, 0)
        Bk_data = B_k[Bk_rows, Bk_cols]

        csc_Bk = sp.csc_matrix((Bk_data, Bk_rows, Bk_ind_ptr), shape=(nx, nx))

        # QP problem setup
        qp_prob = osqp.OSQP()

        if isinstance(A_k, np.ndarray):
            # Setting up QP problem with fully dense dummy_A ensures that
            # change in density of A_k will not affect the 'update()'s
            dummy_A = np.full(A_k.shape, 1.)
            qp_prob.setup(
                csc_Bk,
                g_k,
                sp.csc_matrix(dummy_A),
                l_k,
                u_k,
                max_iter=5000,
                verbose=False,
                # verbose=True,
                # warm_start=False,
                # polish=False,
                # polish=True,
                # polish_refine_iter=3,
                # linsys_solver='qdldl',
                # eps_prim_inf=1e-4,
                # eps_dual_inf=1e-4,
                eps_abs=qp_tol,   # default=1e-3
                eps_rel=qp_tol,   # default=1e-3
                # eps_abs=max(min(opt_tol, feas_tol) * 1e-2, 1e-8),
                # eps_rel=max(min(opt_tol, feas_tol) * 1e-2, 1e-8),
            )
            qp_prob.update(Ax=A_k.flatten('F'))

            dummy_A = None
            del dummy_A
        elif isinstance(A_k, sp.csc_matrix):
            # We assume that the sparsity structure of A_k will not
            # change during iterations if a csc_matrix is given as input
            qp_prob.setup(
                csc_Bk,
                g_k,
                A_k,
                l_k,
                u_k,
                #   warm_start=True,
                verbose=False,
                polish=True)
        else:
            raise TypeError(
                'A_k must be a numpy matrix or scipy csc_matrix')

        while (not (tol_satisfied) and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate

            # Solve QP problem
            qp_sol = qp_prob.solve()

            # Search direction for x_k:
            # (qp_sol.x) is the direction toward the next iterate for x
            p_k[:nx] = qp_sol.x

            # Search direction for pi_k:
            # (-qp_sol.y) is the new estimate for pi
            # p_k[nx:(nx + nc)] = (-qp_sol.y) - pi_k
            p_k[nx:(nx + nc)] = (-qp_sol.y) - v_k[nx:nx + nc]

            # Search direction for s_k :
            # (c_k + J_k @ p_x) is the new estimate for s
            # p_k[(nx + nc):] = c_k + J_k @ p_x - s_k
            p_k[(nx + nc):] = c_k + J_k @ p_x - v_k[nx + nc:]

            # Update penalty parameters
            if self.nc > 0:
                dir_deriv_al = np.dot(mfg_k, p_k)
                pTHp = p_x.T @ (B_k @ p_x)
                rho_k = self.update_scalar_rho(rho_k, dir_deriv_al, pTHp, p_pi, c_k, s_k)
                # rho_k = self.update_vector_rho(rho_k, dir_deriv_al, pTHp, p_pi, c_k, s_k)

                MF.set_rho(rho_k)
                mf_k  = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
                mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)

            else:
                mf_k  = f_k
                mfg_k = g_k

            # Compute the step length along the search direction via a line search
            alpha, mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = LSS.search(
                x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            nfev += new_f_evals
            ngev += new_g_evals

            if not converged:  # Fallback: Backtracking LS
                print(f"Wolfe line search failed at major iteration {itr}. Falling back to backtracking...")
                alpha, mf_new, new_f_evals, new_g_evals, converged = LSB.search(
                    x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

                nfev += new_f_evals
                ngev += new_g_evals

            # Reset Hessian if both line searches fail
            if not converged:
                self.num_consecutive_ls_fails += 1 
                if self.num_consecutive_ls_fails >= 2:
                    print('Both line searches failed again after resetting Hessian. Exiting...')
                    break
                else:
                    print(f"Both line searches failed at major iteration {itr}. Resetting Hessian...")
                    self.QN = BFGS(nx=nx, exception_strategy='damp_update')
                    continue

            else:
                self.num_consecutive_ls_fails = 0
                d_k = alpha * p_k

            g_old = g_k * 1.
            J_old = J_k * 1.

            v_k += d_k

            f_k = obj(x_k)
            g_k = grad(x_k)
            c_k = con(x_k)
            J_k = jac(x_k)

            # Slack reset for scalar rho
            # if rho_k[0] == 0:
            #     # v_k[nx + nc:] = 0.
            #     v_k[nx + nc:] = np.maximum(0, c_k)
            # # When rho_k[0] > 0
            # else:
            #     v_k[nx + nc:] = np.maximum(0, c_k - v_k[nx:nx + nc] / rho_k)

            v_k[(nx + nc):] = self.reset_slacks(c_k, pi_k, rho_k)

            # Note: MF changes (decreases) after slack reset
            mf_k  = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)

            # Update QP parameters
            l_k = -c_k
            A_k =  J_k

            w_k = g_k - (J_k - J_old).T @ pi_k - g_old 

            # if g_new == 'Unavailable':
            #     g_new = grad(x_k)

            # Update the Hessian approximation
            QN.update(d_k[:nx], w_k)

            # New Hessian approximation
            B_k = QN.B_k
            Bk_data[:] = B_k[Bk_rows, Bk_cols]

            # Update the QP problem
            if isinstance(A_k, np.ndarray):
                qp_prob.update(q=g_k,
                               l=l_k,
                               Px=Bk_data,
                               Ax=A_k.flatten('F'))
            else:
                qp_prob.update(q=g_k, l=l_k, Px=Bk_data, Ax=A_k.data)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
            if self.nc == 0:
                feas_satisfied, feas = True, 0.
            else:
                feas_satisfied, feas = self.feas_check(x_k, c_k)
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
                time=time.time() - start_time,
                nfev=nfev,
                ngev=ngev,
                rho=rho_k,
                step=alpha,
                # merit=mf_new)
                merit=mf_k)

        self.total_time = time.time() - start_time
        converged = tol_satisfied

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
            'converged': converged
        }

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results