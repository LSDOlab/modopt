import numpy as np
import warnings
import time
from modopt import Optimizer
from modopt.utils.options_dictionary import OptionsDictionary

try:
    import cvxopt as co
except:
    warnings.warn("'cvxopt' could not be imported. Install 'cvxopt' using `pip install cvxopt` for using CVXOPT optimizer.")

class CVXOPT(Optimizer):
    '''
    Class that interfaces modOpt with the CVXOPT package to solve
    Nonlinear Convex Optimization problems.
    Note that second derivatives (Hessians) must be provided for these problems,
    EQUALITY constraints must be LINEAR.
    Lagrangian Hessian must be provided for constrained problems, and
    objective Hessian must be provided for unconstrained/bounded problems.
    '''
    def initialize(self, ):
        '''
        Initialize the Optimizer() instance for CVXOPT.
        '''
        self.solver_name = 'cvxopt-cp'
        self.options.declare('solver_options', types=dict, default={})

        self.default_solver_options = {
            'show_progress': (bool, True),
            'maxiters': (int, 100),
            'abstol': (float, 1e-7),
            'reltol': (float, 1e-6),
            'feastol': (float, 1e-7),
            'refinement': (int, 1),
        }
        # Used for verifying the keys and value-types of user-provided solver_options, 
        # and generating an updated pure Python dictionary to update cvxopt.solvers.options
        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[0], default=value[1])

        # Declare outputs
        self.available_outputs = {}
        self.options.declare('readable_outputs', values=([],), default=[])

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.obj_hess = self.problem._compute_objective_hessian
        self.active_callbacks = ['obj', 'grad', 'obj_hess']
        if self.problem.constrained:
            self.con_in = self.problem._compute_constraints
            self.jac_in = self.problem._compute_constraint_jacobian
            self.lag_hess = self.problem._compute_lagrangian_hessian
            self.active_callbacks.remove('obj_hess')
            self.active_callbacks += ['con_in', 'jac_in', 'lag_hess']

    def setup(self):
        '''
        Setup the constraints and initial guess.
        Check if objective/Lagrangian Hessian is defined.
        '''
        # Check if gradient/Jacobian/Hessian are declared and raise error/warning for Problem/ProblemLite
        if not self.problem.constrained:
            self.check_if_callbacks_are_declared('grad', 'Objective gradient', 'CVXOPT')
            self.check_if_callbacks_are_declared('obj_hess', 'Objective Hessian', 'CVXOPT')
        else:
            self.check_if_callbacks_are_declared('jac', 'Constraint Jacobian', 'CVXOPT')
            self.check_if_callbacks_are_declared('lag_hess', 'Lagrangian Hessian', 'CVXOPT')

        # Check if user-provided solver_options have valid keys and value-types
        self.solver_options.update(self.options['solver_options'])

        self.x0 = self.problem.x0 * 1.
        self.nx = self.problem.nx * 1
        self.setup_constraints()

    def setup_constraints(self, ):
        '''
        Adapt the problem constraints and variable bounds for the cvxopt.cp solver in the form of
        f_k(x) <= 0, and Ax = b, where k = 1, 2, ..., m.
        f_0(x) will be the objective function.
        All equality constraints must be linear and are merged into Ax = b.
        In modOpt, we combine all the inequality constraints (linear and nonlinear) 
        and variable bounds into f_k(x) <= 0.
        '''
        xl = self.problem.x_lower
        xu = self.problem.x_upper

        lbi = self.lower_bound_indices = np.where(xl != -np.inf)[0]
        ubi = self.upper_bound_indices = np.where(xu !=  np.inf)[0]     
        self.lower_bounded = True if len(lbi) > 0 else False
        self.upper_bounded = True if len(ubi) > 0 else False    

        if self.problem.constrained:
            cl = self.problem.c_lower
            cu = self.problem.c_upper
            eqi = self.eq_constraint_indices    = np.where(cl == cu)[0]
            lci = self.lower_constraint_indices = np.where((cl != -np.inf) & (cl != cu))[0]
            uci = self.upper_constraint_indices = np.where((cu !=  np.inf) & (cl != cu))[0]
        else:
            eqi = self.eq_constraint_indices    = np.array([])
            lci = self.lower_constraint_indices = np.array([])
            uci = self.upper_constraint_indices = np.array([])
        
        self.eq_constrained    = True if len(eqi) > 0 else False
        self.lower_constrained = True if len(lci) > 0 else False
        self.upper_constrained = True if len(uci) > 0 else False

        self.nc_b = len(lbi) + len(ubi)
        self.nc_e = len(eqi)
        self.nc_i = self.nc_b + len(lci) + len(uci)
        
        if self.eq_constrained:
            c0_eq = self.con_in(self.x0)[eqi]
            j0_eq = self.jac_in(self.x0)[eqi]
            k     = c0_eq - j0_eq @ self.x0
            self.A = co.matrix(j0_eq)
            self.b = co.matrix(cu[eqi] - k)
        else:
            self.A = None
            self.b = None

    def con(self, x):
        '''
        Compute problem inequality constraints and variable bounds as f_k(x)<=0.
        '''
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices
        lci = self.lower_constraint_indices
        uci = self.upper_constraint_indices
        xl  = self.problem.x_lower
        xu  = self.problem.x_upper
        cl  = self.problem.c_lower
        cu  = self.problem.c_upper

        c_out = np.array([])

        if self.lower_bounded:
            c_out = np.append(c_out, xl[lbi] - x[lbi])
        if self.upper_bounded:
            c_out = np.append(c_out, x[ubi] - xu[ubi])

        if self.problem.constrained:
            c_in = self.con_in(x)
            if self.lower_constrained:
                c_out = np.append(c_out, cl[lci] - c_in[lci])
            if self.upper_constrained:
                c_out = np.append(c_out, c_in[uci] - cu[uci])

        return c_out

    def jac(self, x):
        '''
        Compute problem Jacobian for inequality constraints and variable bounds as f_k(x)<=0.
        '''
        nx = self.nx
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices
        lci = self.lower_constraint_indices
        uci = self.upper_constraint_indices

        j_out = np.empty((0, nx), dtype=float)

        if self.lower_bounded:
            j_out = np.append(j_out, -np.identity(nx)[lbi], axis=0)
        if self.upper_bounded:
            j_out = np.append(j_out,  np.identity(nx)[ubi], axis=0)
        
        if self.problem.constrained:
            j_in = self.jac_in(x)
            if self.lower_constrained:
                j_out = np.append(j_out, -j_in[lci], axis=0)
            if self.upper_constrained:
                j_out = np.append(j_out,  j_in[uci], axis=0)

        return j_out

    def F(self, x=None, z=None):
        '''
        Computes and returns the objective, constraints, and their first and second derivatives.

        Parameters
        ----------
        x : np.ndarray
            A dense real matrix of size (n,1).
        z : np.ndarray
            A positive dense real matrix of size (m+1,1).

        Returns
        -------
        (m, x0) : tuple(int, np.ndarray)
            F() returns a tuple (m, x0) where m is the number of nonlinear constraints
            and x0 is a point in the domain of f.
            x0 is a dense real matrix of size (n,1).
        (f, Df) : tuple(np.ndarray, np.ndarray)
            F(x) returns a tuple (f, Df) where f is a dense real matrix of size (m+1, 1), 
            with f[0] equal to the objective and f[1:m] equal to the constraints. 
            (If m is zero, f can also be returned as a number.) 
            Df is a dense or sparse real matrix of size (m + 1, n) with Df[k,:] equal to the 
            transpose of the gradient \nabla f_k(x). If x is not in the domain of f, 
            F(x) returns None or a tuple (None, None).
        (f, Df, H) : tuple(np.ndarray, np.ndarray, np.ndarray)
            F(z) returns a tuple (f, Df, H) where f and Df are defined as above.
            H is a square dense or sparse real matrix of size (n, n), 
            whose lower triangular part contains the lower triangular part of

            z_0 \nabla^2f_0(x) + z_1 \nabla^2f_1(x) + \cdots + z_m \nabla^2f_m(x).

            If F is called with two arguments, it can be assumed that x is in the domain of f.

        '''
        if x is None: return self.nc_i, co.matrix(self.x0)
        x_np = np.asarray(x).flatten()
        if self.nc_i > 0:   # If bounds or inequality constraints are present
            f  = co.matrix(np.concatenate(([self.obj(x_np)],  self.con(x_np))))
            Df = co.matrix(np.concatenate(([self.grad(x_np)], self.jac(x_np))))
        else:
            f  = co.matrix(self.obj(x_np))
            Df = co.matrix(self.grad(x_np)).T
        if z is None: return f, Df
        z_np = np.asarray(z).flatten()
        if self.problem.constrained:
            z_prob = np.zeros(self.problem.nc)
            # Hessians of equality constraints and lower/upper bounds are zero
            # Hessians of lower inequality constraints are negative in f_k(x) <= 0
            z_prob[self.lower_constraint_indices] -= z_np[1+self.nc_b:1+self.nc_b+len(self.lower_constraint_indices)]
            z_prob[self.upper_constraint_indices] += z_np[1+self.nc_b+len(self.lower_constraint_indices):]
            H = z[0] * co.matrix(self.lag_hess(x_np, z_prob/z[0]))
        else:
            H = z[0] * co.matrix(self.obj_hess(x_np))
        return f, Df, H

    def solve(self, ):
        '''
        Solve the nonlinear convex problem by calling cvxopt.solvers.cp with given options.
        '''
        co.solvers.options.update(self.solver_options.get_pure_dict())

        start_time = time.time()
        solution = co.solvers.cp(self.F, A=self.A, b=self.b)
        self.total_time = time.time() - start_time

        # cp() returns a dictionary with keys []'status', 'x', 'snl', 'sl',
        # 'znl', 'zl', 'y', 'primal objective', 'dual objective', 'gap',
        # 'relative gap', 'primal infeasibility', 'dual infeasibility',
        # 'primal slack', 'dual slack'].

        # The 'status' field has values 'optimal' or 'unknown'.
        # If status is 'optimal', x, snl, sl, y, znl, zl  are approximate 
        # solutions of the primal and dual optimality conditions

        #     f(x)[1:] + snl = 0,  G*x + sl = h,  A*x = b 
        #     Df(x)'*[1; znl] + G'*zl + A'*y + c = 0 
        #     snl >= 0,  znl >= 0,  sl >= 0,  zl >= 0
        #     snl'*znl + sl'* zl = 0.

        # If status is 'unknown', x, snl, sl, y, znl, zl are the last
        # iterates before termination.  They satisfy snl > 0, znl > 0, 
        # sl > 0, zl > 0, but are not necessarily feasible.

        # cp solves the problem by applying cpl to the epigraph form problem.
        # The values of the other fields describe the accuracy of the solution and 
        # are the values returned by cpl() applied to the epigraph form problem

        #     minimize   t 
        #     subjec to  f0(x) <= t
        #                fk(x) <= 0, k = 1, ..., mnl
        #                G*x <= h
        #                A*x = b.


        # Termination with status 'unknown' indicates that the algorithm 
        # failed to find a solution that satisfies the specified tolerances.
        # In some cases, the returned solution may be fairly accurate.  If
        # the primal and dual infeasibilities, the gap, and the relative gap
        # are small, then x, y, snl, sl, znl, zl are close to optimal.

        self.results = solution
        self.results['time'] = self.total_time
        self.results['x']   = np.asarray(self.results['x']).flatten()   # Optimal design variables
        self.results['y']   = np.asarray(self.results['y']).flatten()   # Optimal dual variables for linear equality constraints
        self.results['zl']  = np.asarray(self.results['zl']).flatten()  # Optimal dual variables for linear inequality constraints (not supported in modOpt)
        self.results['znl'] = np.asarray(self.results['znl']).flatten() # Optimal dual variables for nonlinear inequality constraints
        self.results['sl']  = np.asarray(self.results['sl']).flatten()  # Optimal slack variables for linear inequality constraints (not supported in modOpt)
        self.results['snl'] = np.asarray(self.results['snl']).flatten() # Optimal slack variables for nonlinear inequality constraints
        self.results['objective'] = self.obj(self.results['x'])
        if self.problem.constrained:
            self.results['constraints'] = self.con_in(self.results['x'])
        else:
            self.results['constraints'] = []

        self.run_post_processing()

        return self.results

    def print_results(self, 
                      optimal_variables=False,
                      optimal_constraints=False,
                      optimal_dual_variables=False,
                      optimal_slack_variables=False,
                      all=False):

        output  = "\n\tSolution from CVXOPT:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':40}: {self.problem_name}"
        output += f"\n\t{'Solver':40}: {self.solver_name}"
        output += f"\n\t{'Status':40}: {self.results['status']}"
        output += f"\n\t{'Objective':40}: {self.results['objective']}"
        output += f"\n\t{'Total time':40}: {self.results['time']}"
        output += self.get_callback_counts_string(40)
        
        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':40}: {self.results['x']}"
        if optimal_constraints or all:
            output += f"\n\t{'Optimal constraints':40}: {self.results['constraints']}"        
        if optimal_dual_variables or all:
            output += f"\n\t{'Optimal dual variables (linear eq)':40}: {self.results['y']}"       
            output += f"\n\t{'Optimal dual variables (linear ineq)':40}: {self.results['zl']} (merged with nonlinear ineq since specifying linear ineq constraints is not supported in modOpt)"       
            output += f"\n\t{'Optimal dual variables (nonlinear ineq)':40}: {self.results['znl']}"     
        if optimal_slack_variables or all:
            output += f"\n\t{'Optimal slack variables (linear ineq)':40}: {self.results['sl']} (merged with nonlinear ineq since specifying linear ineq constraints is not supported in modOpt)"       
            output += f"\n\t{'Optimal slack variables (nonlinear ineq)':40}: {self.results['snl']}"

        output += "\n\n\t\tOutputs of the epigraph form problem 'cpl()':"
        output += "\n\t\t" + "-" * (100-8)
        output += f"\n\t\t{'Primal objective':40}: {self.results['primal objective']}"
        output += f"\n\t\t{'Dual objective':40}: {self.results['dual objective']}"
        output += f"\n\t\t{'Gap':40}: {self.results['gap']}"
        output += f"\n\t\t{'Relative gap':40}: {self.results['relative gap']}"
        output += f"\n\t\t{'Primal infeasibility':40}: {self.results['primal infeasibility']}"
        output += f"\n\t\t{'Dual infeasibility':40}: {self.results['dual infeasibility']}"
        output += f"\n\t\t{'Primal slack':40}: {self.results['primal slack']}"
        output += f"\n\t\t{'Dual slack':40}: {self.results['dual slack']}" 

        output += '\n\t' + '-'*100

        print(output)