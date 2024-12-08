import numpy as np
import time

from modopt import Optimizer
from modopt.line_search_algorithms import ScipyLS, BacktrackingArmijo, Minpack2LS
from modopt.merit_functions import AugmentedLagrangian
from modopt.approximate_hessians import BFGSDamped as BFGS

# This optimizer takes constraints in all-inequality form, C(x) >= 0
class BSQP(Optimizer):
    '''
    An implementation of the Sequential Quadratic Programming (SQP) algorithm
    for general constrained optimization problems that uses a very basic QP solver.
    '''
    def initialize(self):
        self.solver_name = 'bsqp'

        self.nx = self.problem.nx

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        # self.active_callbacks = ['obj', 'grad']
        if self.problem.constrained:
            self.con_in = self.problem._compute_constraints
            self.jac_in = self.problem._compute_constraint_jacobian
            # self.lag_hess = self.problem._compute_lagrangian_hessian
            # self.active_callbacks += ['con', 'jac']

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-7, types=float)
        self.options.declare('feas_tol', default=1e-7, types=float)
        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'major': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),

            # Note that the number of constraints will
            # be updated after constraints are setup
            'lag_mult': (float, (self.problem.nc, )),
            # 'slacks': (float, (self.problem.nc, )),
            'constraints': (float, (self.problem.nc, )),
            # 'opt': float,
            # 'feas': float,
            'time': float,
            'nfev': int,
            'ngev': int,
            'step': float,
            # 'rho': float,
            'merit': float,
        }

    def setup(self):
        self.setup_constraints()
        nx   = self.nx
        nc   = self.nc
        nc_e = self.nc_e
        self.available_outputs['lag_mult'] = (float, (nc, ))
        # self.available_outputs['rho'] = (float, (nc, ))
        # self.available_outputs['slacks'] = (float, (nc, ))
        self.available_outputs['constraints'] = (float, (nc, ))
        # Damping parameters
        self.delta_rho = 1.
        self.num_rho_changes = 0

        # self.QN = BFGS(nx=nx)
        self.QN = BFGS(nx=nx,
                       exception_strategy='skip_update')
                    #    exception_strategy='damp_update')
        self.MF = AugmentedLagrangian(nx=nx,
                                      nc=nc,
                                      nc_e=nc_e,
                                      f=self.obj,
                                      c=self.con,
                                      g=self.grad,
                                      j=self.jac)

        self.LSS = ScipyLS(f=self.MF.compute_function,
                           g=self.MF.compute_gradient,
                           max_step=2.0)
        # self.LS = Minpack2LS(f=self.MF.compute_function,
        #                      g=self.MF.compute_gradient)
        self.LSB = BacktrackingArmijo(f=self.MF.compute_function,
                                      g=self.MF.compute_gradient)

        # For u_k of QP solver
        self.u_k = np.full((nc, ), np.inf)

    # Adapt constraints to C_e(x)=0, and C_i(x) >= 0
    def setup_constraints(self, ):
        xl = self.problem.x_lower
        xu = self.problem.x_upper

        ebi = self.eq_bound_indices    = np.where(xl == xu)[0]
        lbi = self.lower_bound_indices = np.where((xl != -np.inf) & (xl != xu))[0]
        ubi = self.upper_bound_indices = np.where((xu !=  np.inf) & (xl != xu))[0]

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
        self.nc_i = len(lbi) + len(ubi) + len(lci) + len(uci)
        self.nc   = len(lbi) + len(ubi) + len(ebi) + len(lci) + len(uci) + len(eci)

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
        ebi = self.eq_bound_indices
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices

        if self.problem.constrained:
            eci = self.eq_constraint_indices
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Compute problem constraint Jacobian
            j_in = self.jac_in(x)

        j_out = np.empty((1, nx), dtype=float)

        if self.eq_bounded:
            j_out = np.append(j_out, np.identity(nx)[ebi], axis=0)
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
        rho_zero_indices = np.where(rho == 0)[0]
        rho_nonzero_indices = np.where(rho > 0)[0]
        s = np.zeros((len(rho), ))

        if len(rho_zero_indices) > 0:
            s[rho_zero_indices] = np.maximum(0, c[rho_zero_indices])

        if len(rho_nonzero_indices) > 0:
            c_rnz = c[rho_nonzero_indices]
            pi_rnz = pi[rho_nonzero_indices]
            rho_rnz = rho[rho_nonzero_indices]
            s[rho_nonzero_indices] = np.maximum(
                0, (c_rnz - pi_rnz / rho_rnz))

        return s

    def update_scalar_rho(self, rho_k, dir_deriv_al, pTHp, p_pi, c_k,
                          s_k):
        nc = self.nc
        delta_rho = self.delta_rho

        # rho_ref = np.linalg.norm(rho_k)
        rho_ref = rho_k[0] * 1.

        if dir_deriv_al <= -0.5 * pTHp:
            rho_computed = rho_k[0]
        else:
            # note: scalar rho_min here
            rho_min = 2 * np.linalg.norm(p_pi) / np.linalg.norm(c_k -
                                                                s_k)
            rho_computed = max(rho_min, 2 * rho_k[0])
            # rho[:] = np.max(rho_min, 2 * rho[0]) * np.ones((m,))

        # Damping for rho (Note: vector rho is still not updated)
        if rho_k[0] < 4 * (rho_computed + delta_rho):
            rho_damped = rho_k[0]
        else:
            rho_damped = np.sqrt(rho_k[0] * (rho_computed + delta_rho))

        rho_new = max(rho_computed, rho_damped)
        rho_k[:] = rho_new * np.ones((nc, ))

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

    def update_vector_rho(self, rho_k, dir_deriv_al, pTHp, p_pi, c_k,
                          s_k):
        delta_rho = self.delta_rho

        rho_ref = np.linalg.norm(rho_k)
        # rho_ref = rho_k[0] * 1.

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
                rho_min = 2 * np.linalg.norm(p_pi) / np.linalg.norm(
                    c_k - s_k)
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
            # Note: compl_violation can be negative(Question on SNOPT);
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
            raise ImportError("OSQP cannot be imported for the SQP solver.  Install it with 'pip install osqp'.")
        try:
            import qpsolvers
        except ImportError:
            raise ImportError("qpsolvers cannot be imported for the SQP solver.  Install it with 'pip install qpsolvers'.")

        
        # Assign shorter names to variables and methods
        nx = self.nx
        nc = self.nc
        nc_e = self.nc_e
        nc_i = self.nc_i

        x0 = self.problem.x0
        maxiter = self.options['maxiter']

        obj = self.obj
        grad = self.grad
        con = self.con
        jac = self.jac

        LSS = self.LSS
        LSB = self.LSB
        QN = self.QN
        MF = self.MF

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1.
        # pi_k = np.full((nc, ), 1.)
        pi_k = np.full((nc, ), 0.)

        f_k = obj(x_k)
        g_k = grad(x_k)
        c_k = con(x_k)
        J_k = jac(x_k)

        if nc > 0:
            print('dim of J_k:', J_k.shape)
            print('rank of J_k:', np.linalg.matrix_rank(J_k))

        eq_con_viol_indices   = np.where(c_k[:nc_e] != 0)[0]
        ineq_con_viol_indices = nc_e + np.where(c_k[nc_e:]  < 0)[0]
        con_viol_indices = np.concatenate((eq_con_viol_indices, ineq_con_viol_indices))
        penalty_k = f_k + np.dot(np.absolute(pi_k[con_viol_indices]), np.absolute(c_k[con_viol_indices]))

        # # A constraint (eq/ineq) is considered active if its absolute value is less than 1e-12
        # iact = np.where(np.abs(c_k) < 1e-12)[0]
        
        # Active set represented as a boolean vector
        iact = np.zeros((nc, ), dtype=bool)
        # Activate only equality constraints initially
        iact[:nc_e] = 1

        nfev = 1
        ngev = 1

        B_k = np.identity(nx)

        # Vector of penalty parameters
        # rho_k = np.full((nc, ), 1.)
        rho_k = np.full((nc, ), 0.)

        # Compute slacks by minimizing A.L. s.t. s_k >= 0
        # s_k = np.full((nc, ), 1.)
        s_k = self.reset_slacks(c_k, pi_k[nc_e:], rho_k[nc_e:])

        # s_k = np.maximum(0, c_k) # Use this if rho_k = 0. at start

        # Vector of design vars., lag. mults., and slack vars.
        v_k = np.concatenate((x_k, pi_k, s_k))
        # x_k = v_k[:nx]
        # pi_k = v_k[nx:(nx + nc)]
        # s_k = v_k[(nx + nc):]

        # p_k = np.zeros((len(v_k), ))
        # p_x = p_k[:nx]
        # p_pi = p_k[nx:(nx + nc)]
        # p_s = p_k[(nx + nc):]

        # # QP parameters
        # l_k = -c_k
        # u_k = self.u_k
        # A_k = J_k

        # Iteration counter
        itr = 0

        # opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
        # if self.nc > 0:
        #     feas_satisfied, feas = self.feas_check(x_k, c_k)
        # else:
        #     feas_satisfied, feas = True, 0.
        
        # tol_satisfied = (opt_satisfied and feas_satisfied)

        # Evaluate merit function value
        MF.set_rho(rho_k)
        mf_k = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
        mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)

        # Initializing declared outputs
        self.update_outputs(major=0,
                            x=x_k,
                            lag_mult=pi_k,
                            # slacks=s_k,
                            obj=f_k,
                            constraints=c_k,
                            # opt=opt,
                            # feas=feas,
                            time=time.time() - start_time,
                            nfev=nfev,
                            ngev=ngev,
                            step=0.,
                            # rho=rho_k,
                            merit=mf_k)
                            # merit=penalty_k)

        # if self.nc > 0:
        #     qp_prob = qpsolvers.Problem(B_k, g_k, np.vstack([A_k, -A_k]), np.vstack([u_k, l_k]))
        # else:
        #     qp_prob = qpsolvers.Problem(B_k, g_k)

        # while (not (tol_satisfied) and itr < maxiter):
        while itr < maxiter:
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate

            # Solve QP problem
            # qp_sol = qp_prob.solve()

            # qp_sol = qpsolvers.solve_problem(qp_prob, 'quadprog')
            # p_k[:nx] = qp_sol.x
            # p_k[nx:(nx + nc)] = (-qp_sol.z) - v_k[nx:nx + nc]

            eps = 1e-16
            for j in range(self.nc_i + 1): # + 1 is needed to ensure that one execution of the loop is done for unconstrained or equality constrained cases
                J_act = J_k[iact]
                c_act = c_k[iact]
                nc_act = len(c_act)
                kkt_mtx = np.block([[B_k, J_act.T], [J_act, np.zeros((nc_act, nc_act))]])
                kkt_rhs = -np.concatenate([g_k, c_act])
                kkt_sol = np.linalg.solve(kkt_mtx, kkt_rhs)

                p_x    =  kkt_sol[:nx]
                pi_new = np.zeros((nc, ))
                pi_new[iact] = -kkt_sol[nx:]
                p_pi  = pi_new - pi_k
                c_new  = c_k + J_k @ p_x

                # ineq_qp_con_viol_indices = np.where(c_new[nc_e:] < 0)[0]
                # iact[nc_e:] = c_new[nc_e:] < 0
                iact[nc_e:] = np.where(c_new[nc_e:] <  -eps, 1, iact[nc_e:])
                i_satisfied = np.where(c_new[nc_e:] >= -eps)[0]
                iact[nc_e:][i_satisfied] = np.where(pi_new[nc_e:][i_satisfied] < -eps, 0, iact[nc_e:][i_satisfied])
                # pi_new      = np.where(pi_new[nc_e:] < 0, 0, pi_new[nc_e:])

                ########## NEW ADDITION ##########


                ########## NEW ADDITION ##########
                
                if all(c_new[nc_e:] >= -eps) and all(pi_new[nc_e:] >= -eps):
                    print('All constraints satisfied with no negative multipliers '\
                          'at major iteration:', itr, 'with minor iterations:', j)
                    break

                # pi_new = np.where(kkt_sol[nx:] < 0, 0, kkt_sol[nx:])
                # p_x = np.linalg.solve(B_k, -g_k) if any(kkt_sol[nx:] < 0) else p_x
                # # p_x = np.linalg.solve(B_k, -g_k - J_k.T @ pi_new)
                # # p_pi = pi_new - pi_k

            else:
                print(f'Maximum number of minor iterations ({j}) reached at major iteration:', itr)

            # x_new  = x_k + p_x
            # f_new = obj(x_new)
            # g_new = grad(x_new)
            # c_new = con(x_new)
            # J_new = jac(x_new)

            # eq_con_viol_indices   = np.where(c_new[:nc_e] != 0)[0]
            # ineq_con_viol_indices = nc_e + np.where(c_new[nc_e:]  < 0)[0]
            # con_viol_indices = np.concatenate((eq_con_viol_indices, ineq_con_viol_indices))
            # penalty_new = f_new + np.dot(np.absolute(pi_new[con_viol_indices]) + 1, np.absolute(c_new[con_viol_indices]))

            # ls_count = 0
            # while penalty_new > penalty_k:
            #     ls_count += 1
            #     p_x = 0.5 * p_x
            #     x_new  = x_k + p_x
            #     f_new = obj(x_new)
            #     # g_new = grad(x_new)
            #     c_new = con(x_new)
            #     # J_new = jac(x_new)

            #     eq_con_viol_indices   = np.where(c_new[:nc_e] != 0)[0]
            #     ineq_con_viol_indices = nc_e + np.where(c_new[nc_e:]  < 0)[0]
            #     con_viol_indices = np.concatenate((eq_con_viol_indices, ineq_con_viol_indices))
            #     penalty_new = f_new + np.dot(np.absolute(pi_new[con_viol_indices]) + 1, np.absolute(c_new[con_viol_indices]))

            #     if ls_count > 10:
            #         break
            
            # print('step:', .5 ** ls_count)
            # print('merit:', penalty_new)
            # nfev += ls_count

            # Search direction for s_k :
            # (c_k + J_k @ p_x) is the new estimate for s
            # p_k[(nx + nc):] = c_k + J_k @ p_x - s_k
            p_s = c_k[nc_e:] + J_k[nc_e:] @ p_x - v_k[nx + nc:]

            p_k = np.concatenate([p_x, p_pi, p_s])

            dir_deriv_al = np.dot(mfg_k, p_k)
            pTHp = p_x.T @ (B_k @ p_x)

            # Update penalty parameters
            if self.nc > 0:
                rho_k = self.update_scalar_rho(rho_k, dir_deriv_al, pTHp,
                                               p_pi, c_k, np.concatenate([np.zeros((nc_e,)), s_k]))
                # rho_k = self.update_vector_rho(rho_k, dir_deriv_al, pTHp,
                #                                p_pi, c_k, s_k)

            MF.set_rho(rho_k)
            mf_k  = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)

            # Compute the step length along the search direction via a line search
            alpha, mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = LSS.search(
                x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            # alpha, mf_new, new_f_evals, new_g_evals, converged = LSB.search(
            #     x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            nfev += new_f_evals
            ngev += new_g_evals

            # if not converged:  # Backup: Backtracking LS
            #     alpha, mf_new, new_f_evals, new_g_evals, converged = LSB.search(
            #         x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            #     nfev += new_f_evals
            #     ngev += new_g_evals

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                "Compute this factor heuristically"
                alpha = 0.99
                d_k = p_k * 0.1
                print("###### FAILED LINE SEARCH #######")
                nfev += 1  # Commented because of bug in Scipy line search: calls f twice with amax before failure
                ngev += 1

            else:
                print("###### SUCCESSFUL LINE SEARCH #######")
                d_k = alpha * p_k

            v_k += d_k

            x_k = v_k[:nx]
            pi_k = v_k[nx:(nx + nc)]
            f_k = obj(x_k)
            g_old = g_k * 1.
            g_k = grad(x_k)
            c_k = con(x_k)
            J_old = J_k * 1.
            J_k = jac(x_k)

            v_k[(nx + nc):] = self.reset_slacks(c_k[nc_e:], pi_k[nc_e:], rho_k[nc_e:])
            s_k = v_k[(nx + nc):]

            # Note: MF changes (decreases) after slack reset
            mf_k = MF.evaluate_function(x_k, pi_k, s_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(x_k, pi_k, s_k, f_k, c_k, g_k, J_k)
            
            # x_k = x_new
            # f_k = f_new
            # g_k = grad(x_k)
            # c_k = c_new
            # J_k = jac(x_k)
            # penalty_k = penalty_new
            # ngev += 1

            w_k = (g_k - g_old) - (J_k - J_old).T @ pi_k
            p_x = d_k[:nx]
            QN.update(p_x, w_k)

            B_k = QN.B_k

            # print('########### MAJOR ITERATION ENDS ############')

            # # <<<<<<<<<<<<<<<<<<<
            # # ALGORITHM ENDS HERE

            # opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
            # if self.nc > 0:
            #     feas_satisfied, feas = self.feas_check(x_k, c_k)
            # else:
            #     feas_satisfied, feas = True, 0.
            # tol_satisfied = (opt_satisfied and feas_satisfied)

            # Update arrays inside outputs dict with new values from the current iteration

            self.update_outputs(
                major=itr,
                x=x_k,
                lag_mult=pi_k,
                # slacks=s_k,
                obj=f_k,
                constraints=c_k,
                # opt=opt,
                # feas=feas,
                time=time.time() - start_time,
                nfev=nfev,
                ngev=ngev,
                # rho=rho_k,
                step=alpha,
                # step=0.5**ls_count,
                merit=mf_k)
                # merit=penalty_k)

            if np.linalg.norm(p_x) < 1e-6:
                break

        self.total_time = time.time() - start_time
        # converged = tol_satisfied

        self.results = {
            'x': x_k,
            'objective': f_k,
            'c': c_k,
            'pi': pi_k,
            # 'optimality': opt,
            # 'feasibility': feas,
            'nfev': nfev,
            'ngev': ngev,
            'niter': itr,
            'time': self.total_time,
            # 'converged': converged
        }

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results