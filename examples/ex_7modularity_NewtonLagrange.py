'''Method of Newton Lagrange'''

import numpy as np
import time

from modopt import Optimizer
from modopt.line_search_algorithms import ScipyLS, BacktrackingArmijo, Minpack2LS
from modopt.merit_functions import AugmentedLagrangianEq, LagrangianEq
# from modopt.approximate_hessians import BFGS, BFGSScipy, BFGSM1, Broyden, DFP, PSB, SR1
from modopt.approximate_hessians import BFGSM1 as BFGS


class NewtonLagrange(Optimizer):
    def initialize(self):
        self.solver_name = 'newton_lagrange'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.con = self.problem._compute_constraints
        self.jac = self.problem._compute_constraint_jacobian

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-8, types=float)
        self.options.declare('feas_tol', default=1e-8, types=float)
        self.options.declare('outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'mult': (float, (self.problem.nc, )),
            'con': (float, (self.problem.nc, )),
            'opt': float,
            'feas': float,
            'time': float,
            'num_f_evals': int,
            'num_g_evals': int,
            'step': float,
            'merit': float,
        }

    def setup(self):
        self.setup_outputs()
        nx = self.problem.nx
        nc = self.problem.nc
        self.QN = BFGS(nx=self.problem.nx)
        self.MF = AugmentedLagrangianEq(nx=nx,
                                        nc=nc,
                                        f=self.obj,
                                        c=self.con,
                                        g=self.grad,
                                        j=self.jac)

        self.LS = BacktrackingArmijo(f=self.MF.compute_function,
                                     g=self.MF.compute_gradient)
        # self.LS = ScipyLS(f=self.MF.compute_function,
        #                   g=self.MF.compute_gradient)

        self.OF = LagrangianEq(nx=nx,
                               nc=nc,
                               f=self.obj,
                               c=self.con,
                               g=self.grad,
                               j=self.jac)

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        nc = self.problem.nc
        x0 = self.problem.x0
        opt_tol = self.options['opt_tol']
        feas_tol = self.options['feas_tol']
        maxiter = self.options['maxiter']
        rho = 1.

        obj = self.obj
        grad = self.grad
        con = self.con
        jac = self.jac

        LS = self.LS
        QN = self.QN
        MF = self.MF
        OF = self.OF

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)
        c_k = con(x_k)
        J_k = jac(x_k)
        pi_k = np.full((nc, ), 0.)
        v_k = np.concatenate((x_k, pi_k))
        x_k = v_k[:nx]
        pi_k = v_k[nx:]

        # L_k = OF.evaluate_function(x_k, pi_k, f_k, c_k)
        gL_k = OF.evaluate_gradient(x_k, pi_k, f_k, c_k, g_k, J_k)

        MF.set_rho(rho)
        mf_k = MF.evaluate_function(x_k, pi_k, f_k, c_k)
        mfg_k = MF.evaluate_gradient(x_k, pi_k, f_k, c_k, g_k, J_k)

        # Iteration counter
        itr = 0

        opt = np.linalg.norm(gL_k[:nx])
        feas = np.linalg.norm(c_k)
        num_f_evals = 1
        num_g_evals = 1

        # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            mult=pi_k,
                            obj=f_k,
                            con=c_k,
                            opt=opt,
                            feas=feas,
                            time=time.time() - start_time,
                            num_f_evals=num_f_evals,
                            num_g_evals=num_g_evals,
                            step=0.,
                            merit=mf_k)

        while ((opt > opt_tol or feas > feas_tol) and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # Hessian approximation
            B_k = QN.B_k

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Assemble KKT system
            A = np.block([[B_k, -J_k.T], [-J_k, np.zeros((nc, nc))]])
            b = -gL_k

            # Compute the SEARCH DIRECTION toward the next iterate
            p_k = np.linalg.solve(A, b)

            # Compute the STEP LENGTH along the search direction via a LINE SEARCH
            alpha, mf_k, new_f_evals, new_g_evals, converged = LS.search(
                x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            num_f_evals += new_f_evals
            num_g_evals += new_g_evals

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                alpha = None
                d_k = p_k * 1.

            else:
                d_k = alpha * p_k

            v_k += d_k

            # Update previous Lag. gradient with new lag. mult.s
            # before updating the previous values
            gL_k[:nx] += np.dot(J_k.T, (v_k[nx:] - pi_k))

            f_k = obj(x_k)
            g_k = grad(x_k)
            c_k = con(x_k)
            J_k = jac(x_k)

            gL_k_new = OF.evaluate_gradient(x_k, pi_k, f_k, c_k, g_k, J_k)
            w_k = (gL_k_new - gL_k)
            gL_k = gL_k_new
            mf_k = MF.evaluate_function(x_k, pi_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(x_k, pi_k, f_k, c_k, g_k, J_k)

            opt = np.linalg.norm(gL_k[:nx])
            feas = np.linalg.norm(c_k)

            # Update the Hessian approximation
            QN.update(d_k[:nx], w_k[:nx])

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                mult=pi_k,
                                obj=f_k,
                                con=c_k,
                                opt=opt,
                                feas=feas,
                                time=time.time() - start_time,
                                num_f_evals=num_f_evals,
                                num_g_evals=num_g_evals,
                                step=alpha,
                                merit=mf_k)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time