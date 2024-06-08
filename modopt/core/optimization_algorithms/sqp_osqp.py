import osqp
import numpy as np
import scipy.sparse as sp
import scipy.optimize as scipy_opt

import time
# import pandas

import sys

sys.path.append("..")

from modopt.core.optimizer import Optimizer
# from lsdo_optimizer.core.line_search.line_search import wolfe_line_search
from modopt.core.problem import Problem


class AL(object):
    def __init__(self, n, m, func, grad, con, jac):
        self.n = n
        self.m = m
        self.func = func
        self.grad = grad
        self.con = con
        self.jac = jac

        self.rho = np.full((m, ), 0.)

    def al(self, v):
        n = self.n
        m = self.m
        rho = self.rho
        d_L = self.con(v[:n], surf_line_search=True) - v[n + m:]
        return self.func(v[:n], surf_line_search=True) - np.dot(
            v[n:n + m], d_L) + 0.5 * np.dot(rho, (d_L**2))

    def d_al(self, v):
        n = self.n
        m = self.m
        rho = self.rho
        d_L = self.con(v[:n], surf_line_search=True) - v[n + m:]

        upp = self.grad(v[:n]) - self.jac(v[:n]).T @ (v[n:n + m] -
                                                      (rho * d_L))
        mid = -d_L
        low = v[n:n + m] - rho * d_L

        return np.concatenate((upp, mid, low), axis=None)

        # def lag(x, v_k): # modified Lagrangian
        #     d_L = con(x) - con(v_k[:n]) - jac(v_k[:n]) @ (x - v_k[:n])
        #     return func(v[:n]) + np.dot(v[n:n+m], d_L)

        #     # With slacks
        #     d_L = con(v[:n]) - v[n+m:]
        #     return func(v[:n]) + np.dot(v[n:n+m], d_L)

        # def d_lag(v):
        #     d_L = con(v[:n]) - v[n+m:]

        #     upp = grad(v[:n]) - np.matmul(jac(v[:n]).T, v[n:n+m])
        #     mid = -d_L

        #     return np.concatenate((upp, mid), axis=None)


class SQP_OSQP(Optimizer):
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem, **kwargs)

        self.solver = 'sqp'

        self.nx = self.options['nx']
        self.ny = self.options['ny']
        self.nc = self.options['nc']

        self.n = self.nx + self.ny
        self.m = 2 * self.ny + self.nc

    def initialize(self):
        self.options.declare('opt_tol', default=1e-7, types=float)
        self.options.declare('feas_tol', default=1e-7, types=float)

    def setup(self):
        pass

    def run(self):
        nx = self.options['nx']
        ny = self.options['ny']
        # nr = self.options['nr']

        # ne = self.options['ne']
        # ni = self.options['ni']
        nc = self.options['nc']

        # x0 = self.options['x0']
        opt_tol = self.options['opt_tol']
        feas_tol = self.options['feas_tol']
        func = self.obj
        grad = self.grad

        n = nx + ny
        # m = 2 * (ny + nc) + ni

        m = 2 * ny + nc

        # y0 = np.ones((m,), dtype=float)
        # s0 = np.ones((m,), dtype=float)

        v0 = np.full(((n + 2 * m), ), 1.)

        v0[:n] = self.options['x0']
        self.options['x0'] = x0 = v0[:n]

        # ''' ADD BACK
        if self.formulation == 'surf':
            print('surf')
            update_states = self.update_states
            p_state = update_states(x0) - v0[nx:n]
            v0[nx:n] += p_state

            adj = self.adj
            psi = adj(x0)
            v0[n + 1 + 2 * nx:n + m] = np.append(psi, psi)

        # '''

        y0 = v0[n:n + m]
        s0 = v0[n + m:]

        con = self.con
        '''def con(x):
            # ce = self.ce(x)
            # ci = self.ci(x)
            # return np.concatenate((ce, -ce, ci), axis=None)
            c = self.c(x)
            r = self.r(x)
            return np.concatenate((r, -r, c), axis=None)'''

        jac = self.jac
        '''def jac(x):
            # Je = self.Je(x)
            # Ji = self.Ji(x)        
            # return np.concatenate((Je, -Je, Ji), axis=0)
            Jc = self.Jc(x)
            Jr = self.Jr(x)
            return sp.vstack((Jr, -Jr, Jc))'''
        def qp_con(x, p):
            return con(x) + jac(x) @ p

        # Vector of penalty parameters
        rho = np.full((m, ), 0.)
        # rho = 1. is better for scalar rho
        # rho = np.full((m,), 1.)

        # Damping parameter
        delta_rho = 1.
        num_rho_changes = 0

        al = AL(n, m, func, grad, con, jac)

        start_time = time.time()
        func1 = func(x0)

        grad1 = grad(x0)
        grad2 = grad1 * 1.

        jac1 = jac(x0)
        jac2 = jac1 * 1.

        Hk = np.identity(n)

        # For csc format of sparse H_k
        rows = np.triu(
            np.outer(np.arange(1, n + 1),
                     np.ones(n, dtype='int'))).flatten('f')
        rows = rows[rows != 0] - 1
        cols = np.repeat(np.arange(n), np.arange(1, n + 1))

        ind_ptr = np.insert(np.cumsum(np.arange(1, n + 1)), 0, 0)

        # flat_indices=np.ravel_multi_index([rows, cols], (n,n))

        itr_counter = 0
        fg_array = np.array([1])
        x_array = 1 * x0.reshape(1, n)
        func_array = func1 * 1

        # need to correct this later
        # d_lag = np.full((n,), 1.)
        # d_lag = grad1 * 1.
        # norm_array = np.linalg.norm(d_lag)
        opt_array = 1.
        feas_array = 1.
        rho_array = 0.
        step_array = 0.
        merit_array = 0.

        time_array = np.array([0.])

        c = con(x0)
        A = jac1 * 1.

        # tolerance relaxing the constraint violation
        # l = -c - 1e-3
        l = -c

        # ''' ADD BACK
        # if self.formulation == 'surf':
        #     print('surf')
        #     print(c[0], c[1:nx+1], c[nx+1:2*nx+1])
        # '''

        # l = A @ x0 - c
        u = np.full((m, ), np.inf)
        # u = np.full((m,), np.inf)

        data = Hk[rows, cols]

        csc_Hk = sp.csc_matrix((data, rows, ind_ptr), shape=(n, n))

        qp_prob = osqp.OSQP()
        # qp_prob.setup(csc_Hk, grad1, A, l, u, alpha=1.0)
        print('before-setup')

        if isinstance(A, np.ndarray):
            dummy_A = np.full(A.shape, 1.)
            qp_prob.setup(csc_Hk, grad1, sp.csc_matrix(dummy_A), l, u)
            qp_prob.update(Ax=A.flatten('F'))
            dummy_A = None
            del dummy_A
        else:
            qp_prob.setup(csc_Hk, grad1, A, l, u, warm_start=True)

        print('after-setup')

        tol_unsatisfied = True

        while (tol_unsatisfied and itr_counter < 1e+3):
            # while (tol_unsatisfied and itr_counter < 3):
            # while (np.linalg.norm(d_lag) > tol and itr_counter < 1e+3):
            print(c[0])
            print(func1)
            start = time.time()
            itr_counter += 1

            grad1[:] = grad2 * 1
            jac1 = jac2 * 1.

            # Setup workspace and change alpha parameter

            # csc_A = sp.csc_matrix((data, rows, ind_ptr), shape=(n, n))

            # qp_prob.setup(sp.csc_matrix(Hk), grad1, A, l, u, alpha=1.0, warm_start=True, linsys_solver='qdldl')

            # Solve QP problem
            qp_sol = qp_prob.solve()

            # p_s = qp_con(x0, qp_sol.x)

            p_x = qp_sol.x

            ## CULPRIT all the time
            p_y = (-qp_sol.y) - v0[n:n + m]
            ## CULPRIT all the time

            p_s = c + jac1 @ p_x - v0[n + m:]
            # v0[n+m:] = qp_con(x0, res.x)

            p_v = np.concatenate((p_x, p_y, p_s), axis=None)

            pTHp = p_x.T @ (Hk @ p_x)

            # Update penalty parameters
            rho_ref = np.linalg.norm(rho)
            # rho_ref = rho[0] * 1.

            al1 = al.al(v0)
            d_al1 = al.d_al(v0)

            dir_deriv_al = np.dot(d_al1, p_v)
            '''
            aT = ((c - v0[n+m:]) * (jac1 @ p_x - p_s)).reshape(1, m)
            b = -0.5 * pTHp - (dir_deriv_al - aT @ rho)
            # print(aT @ rho)
            # print('b' , b)
            # b = -0.5 * pTHp - np.dot(grad1, p_x) + np.dot(v0[n:n+m], jac1 @ p_x - p_s) + np.dot(c - v0[n+m:], p_y)
            
            l1 = np.append(np.full((m,), 0.), [b])
            # l1 = np.append(np.full((m,), 0.), [-np.inf])
            u1 = np.append(np.full((m,), np.inf), [b])
            # print(l1)
            # print(u1)
            
            g1 = np.full((m,), 0.)
            H1 = sp.csc_matrix(2. * np.identity(m))
            A1 = sp.csc_matrix((np.concatenate((np.identity(m), aT), axis=0)))
            # A1 = sp.csc_matrix((np.concatenate((np.identity(m), aT, -aT), axis=0)))

            rho_min_prob = osqp.OSQP()
            rho_min_prob.setup(P=H1, q=g1, A=A1, l=l1, u=u1, polish=True)

            rho_sol = rho_min_prob.solve()

            '''

            # '''
            if dir_deriv_al <= -0.5 * pTHp:
                rho_computed = rho[0]
            else:
                # note: scalar rho_min here
                rho_min = 2 * np.linalg.norm(
                    p_v[n:n + m]) / np.linalg.norm(c - v0[n + m:])
                rho_computed = max(rho_min, 2 * rho[0])
                # rho[:] = np.max(rho_min, 2 * rho[0]) * np.ones((m,))
            # '''
            '''
            rho_computed = rho_sol.x
            # print(rho_computed)
            # print(-0.5 * pTHp - (dir_deriv_al - aT @ rho) - aT @ rho_computed)
            '''

            # '''
            # Damping for rho (Note: vector rho is still not updated)
            if rho[0] < 4 * (rho_computed + delta_rho):
                rho_damped = rho[0]
            else:
                rho_damped = np.sqrt(rho[0] *
                                     (rho_computed + delta_rho))

            rho_new = max(rho_computed, rho_damped)
            rho[:] = rho_new * np.ones((m, ))
            # '''
            '''
            rho_damped = np.where(rho < 4*(rho_computed + delta_rho), rho, np.sqrt(rho*(rho_computed + delta_rho)))

            rho_new = np.maximum(rho_damped, rho)
            rho[:] = rho_new
            '''

            # '''
            # Increasing rho
            if rho[0] > rho_ref:
                if num_rho_changes >= 0:
                    num_rho_changes += 1
                else:
                    delta_rho *= 2.
                    num_rho_changes = 0

            # Decreasing rho
            elif rho[0] < rho_ref:
                if num_rho_changes <= 0:
                    num_rho_changes -= 1
                else:
                    delta_rho *= 2.
                    num_rho_changes = 0
            # '''
            '''
            # Increasing rho
            if np.linalg.norm(rho) > rho_ref:
                if num_rho_changes >= 0:
                    num_rho_changes += 1
                else:
                    delta_rho *= 2.
                    num_rho_changes = 0

            # Decreasing rho
            elif np.linalg.norm(rho) < rho_ref:
                if num_rho_changes <= 0:
                    num_rho_changes -= 1
                else:
                    delta_rho *= 2.
                    num_rho_changes = 0

            '''

            al.rho = rho

            alpha, new_fg, new_g, new_merit, old_merit, new_slope = scipy_opt.line_search(
                al.al, al.d_al, v0, p_v)

            # alpha, new_merit, new_fg = wolfe_line_search(v0, al1, d_al1, p_v, al.al, al.d_al, 'quasi_newton')

            # alpha = 1
            # new_fg = 1
            # new_merit = al.al(v0 + p_v)

            if alpha is None:
                dk = p_v[:n]
                func1 = func(x0 + dk, surf_line_search=False)
                v0 += p_v
                new_merit = al.al(v0)

            else:
                dk = alpha * p_v[:n]
                # func1 = new_func * 1.
                func1 = func(x0 + dk, surf_line_search=False)
                v0 += alpha * p_v

            c = con(v0[:n], surf_line_search=False)
            # Slack reset
            if rho[0] == 0:
                v0[n + m:] = np.maximum(0, c)
            # When rho[0] > 0
            else:
                v0[n + m:] = np.maximum(0, c - v0[n:n + m] / rho)

            grad2[:] = grad(v0[:n])
            jac2 = jac(v0[:n])

            # ''' ADD BACK
            if self.formulation == 'surf':
                print(
                    'updating psi inside surf optimizer.................'
                )
                # Updating the multipliers
                psi[:] = adj(v0[:n])
                v0[n + 1 + 2 * nx:n + m] = np.append(psi, psi)

                # Updating the step for states
                print(
                    'updating states inside surf optimizer.................'
                )
                p_state[:] = update_states(v0[:n]) - v0[nx:n]
                v0[nx:n] += p_state
                dk[nx:n] += p_state
            # '''

            d_lag = grad2 - (jac2 - jac1).T @ v0[n:n + m]

            wk = d_lag - grad1
            # wk = (grad2 - grad1) + (jac2 - jac1).T @ v0[n:n+m]

            Hd = Hk.dot(dk)

            tol1 = 1e-14

            wTd = np.dot(wk, dk)
            sign = 1. if wTd >= 0. else -1.
            if abs(wTd) > tol1:
                Hk += np.outer(wk, wk) / wTd
            else:
                Hk += np.outer(wk, wk) / sign / tol1

            dTHd = np.dot(dk, Hd)
            sign = 1. if dTHd >= 0. else -1.
            if abs(dTHd) > tol1:
                Hk -= np.outer(Hd, Hd) / dTHd
            else:
                Hk -= np.outer(Hd, Hd) / sign / tol1

            A = jac2

            # relaxing constraint violation tolerance
            # l = -c - 1e-3
            l = -c
            # l = A @ x0 - c

            data[:] = Hk[rows, cols]

            print('before-update')
            if isinstance(A, np.ndarray):
                qp_prob.update(q=grad2, l=l, Px=data, Ax=A.flatten('F'))
            else:
                qp_prob.update(q=grad2, l=l, Px=data, Ax=A.data)
            print('after-update')

            # csc_Hk = sp.csc_matrix((data, rows, ind_ptr), shape=(n, n))

            opt_tol_factor = (1 + np.linalg.norm(v0[n:n + m], np.inf))
            feas_tol_factor = (1 + np.linalg.norm(v0[:n], np.inf))

            opt_tol_check_val = opt_tol * opt_tol_factor
            feas_tol_check_val = feas_tol * feas_tol_factor

            print(opt_tol_check_val)
            print(feas_tol_check_val)

            opt1 = np.linalg.norm(
                np.where(v0[n:n + m] < 0, v0[n:n + m], 0.), np.inf)
            opt_tol_check1 = (opt1 <= opt_tol_check_val)

            # Note: opt2 can be negative; opt1 and opt3 are always nonnegative
            opt2 = np.amax(c * v0[n:n + m])
            opt_tol_check2 = (opt2 <= opt_tol_check_val)

            opt3 = np.linalg.norm(grad2 - jac2.T @ v0[n:n + m], np.inf)
            opt_tol_check3 = (opt3 <= opt_tol_check_val)

            # opt is always nonnegative
            opt = max(opt1, opt2, opt3) / opt_tol_factor

            print(opt_tol_check1)
            print(opt1)
            # print(v0[n:n+m])
            print(opt_tol_check2)
            print(opt2)
            # print(c * v0[n:n+m])
            print(opt_tol_check3)
            print(opt3)

            # feas is always nonnegative
            feas1 = np.linalg.norm(np.where(c < 0, c, 0.), np.inf)
            feas_tol_check = (feas1 <= feas_tol_check_val)

            feas = feas1 / feas_tol_factor

            print(feas_tol_check)
            print(feas1)

            print(opt, feas)

            tol_unsatisfied = not (opt_tol_check1 and opt_tol_check2 and
                                   opt_tol_check3 and feas_tol_check)

            x_array = np.append(x_array, x0.reshape(1, n), axis=0)
            func_array = np.append(func_array, func1)
            # norm_array = np.append(norm_array, np.linalg.norm(d_lag))

            opt_array = np.append(opt_array, opt)
            feas_array = np.append(feas_array, feas)
            rho_array = np.append(rho_array, rho[0])
            step_array = np.append(step_array, alpha)
            merit_array = np.append(merit_array, new_merit)

            end = time.time()
            time_array = np.append(time_array,
                                   [time_array[-1] + end - start])
            fg_array = np.append(fg_array, fg_array[-1] + new_fg)

            iter_array = np.arange(itr_counter + 1)

            # pandas.set_option('display.float_format', '{:.2E}'.format)

            # table = pandas.DataFrame({
            #     "Major": iter_array,
            #     "fg evals": fg_array,
            #     "Obj": func_array,
            #     "Opt": opt_array,
            #     "Feas": feas_array,
            #     "Penalty": rho_array,
            #     "Step": step_array,
            #     "Merit": merit_array,
            #     "Time": time_array
            # })

            name = self.problem_name
            with open(name + '_print.out', 'w') as f:
                f.writelines(table.to_string(index=False))

        iter_array = np.arange(itr_counter + 1)
        end_time = time.time()
        total_time = end_time - start_time

        print(.5 + c[0])
        print(c[1:])

        self.iter_array = iter_array
        self.fg_array = fg_array
        self.x_array = x_array

        self.obj_array = func_array
        self.norm_array = norm_array
        self.opt_array = opt_array
        self.feas_array = feas_array
        self.rho_array = rho_array
        self.step_array = step_array
        self.merit_array = merit_array

        self.total_time = total_time
        self.time_array = time_array

    def check_partials(self, x):
        func = self.obj
        grad = self.grad
        con = self.con
        jac = self.jac

        n = self.n
        m = self.m
        h = 1e-9
        # h = np.full((n,), 1e-9)

        grad_fd = np.full((n, ), func(x))

        jac_fd = np.outer(con(x), np.ones((n, ), dtype=float))

        for i in range(n):
            e = np.zeros((n, ), dtype=float)
            e[i] = h

            grad_fd[i] -= func(x + e)

            jac_fd[:, i] -= con(x + e)

        grad_fd /= -h
        jac_fd /= -h

        grad_exact = grad(x)
        jac_exact = jac(x)

        EPSILON = 1e-10

        grad_abs_error = np.absolute(grad_fd - grad_exact)
        grad_rel_error = grad_abs_error / (np.absolute(grad_fd) +
                                           EPSILON)
        # grad_rel_error = grad_abs_error / (np.absolute(grad_exact) + EPSILON)

        jac_abs_error = np.absolute(jac_fd - jac_exact)
        jac_rel_error = jac_abs_error / np.linalg.norm(jac_fd, 'fro')
        # jac_rel_error = jac_abs_error / (np.absolute(jac_fd) + EPSILON)
        print(jac_rel_error.shape)
        # jac_rel_error = jac_abs_error / (np.absolute(jac_exact.toarray()) + EPSILON)

        print('grad_fd norm', np.linalg.norm(grad_fd))
        print('grad_exact norm', np.linalg.norm(grad_exact))

        print('jac_fd norm', np.linalg.norm(jac_fd, 'fro'))
        print('jac_exact norm', sp.linalg.norm(jac_exact, 'fro'))

        print('grad_abs_error_norm', np.linalg.norm(grad_abs_error))
        print('jac_abs_error_norm',
              np.linalg.norm(jac_abs_error, 'fro'))

        print('grad_rel_error_norm', np.linalg.norm(grad_rel_error))
        print('jac_rel_error_norm',
              np.linalg.norm(jac_rel_error, 'fro'))


# def sqp(objective, gradient, constraints, jacobian, hvp):

# x0 =
# Finding a feasible constraint

# BFGS hessian
# Hx = np.identity()

# KKT matrix
# KKT = np.array([[Hx, A.T], [A, 0]])

# rhs =

# quadratic approximation

# linearized constraints

# qp problem

# prob = osqp.OSQP()

# Setup workspace and change alpha parameter

# prob.setup(P, q, A, l, u, alpha=1.0)

# Solve problem
# res = prob.solve()

# line search with x, lm, and slacks s

# Augmented Lagrangian

# check for optimality and feasiblity
