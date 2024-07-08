import numpy as np
import warnings
import time
from modopt import Optimizer, Problem, ProblemLite

try:
    import qpsolvers
except:
    warnings.warn("'qpsolvers' could not be imported."\
                  "Install 'qpsolvers' using `pip install qpsolvers[wheels_only] quadprog osqp` "\
                  "to install open-source QP solvers with pre-compiled binaries.")

class ConvexQPSolvers(Optimizer):
    '''
    Class that interfaces modOpt with the qpsolvers package which provides 
    a unified interface to various Quadratic Programming (QP) solvers available in Python.
    By default, the solver used is 'quadprog'.

    Attributes
    ----------
    solver_name : str
        The name of the solver to be used.
    available_solvers : list
        The list of available solvers in qpsolvers.
        A subset of all the suppported solvers that are installed.
    supported_solvers : list
        The list of supported solvers in qpsolvers.
    qp_problem : qpsolvers.Problem
        The QP problem to be solved.

        Attributes
        ----------
            P : np.ndarray
                The symmetric cost matrix. Always positive semi-definite for convex QP.
                Some solvers also require it to be positive definite.
            q : np.ndarray
                The cost vector.
            G : np.ndarray
                The linear inequality constraint matrix.
            h : np.ndarray
                The linear inequality constraint vector.
            A : np.ndarray
                The linear equality constraint matrix.
            b : np.ndarray
                The linear equality constraint vector.
            lb : np.ndarray
                The lower bounds for the variables.
            ub : np.ndarray
                The upper bounds for the variables.
            has_sparse : bool
                Check whether the problem has sparse matrices.
            is_unconstrained : bool
                Check whether the problem has constraints.
        
        Methods
        -------
            check_constraints()
                Check if the problem constraints are correctly defined.
            cond(active_set: qpsolvers.ActiveSet)
                Compute the condition number of the symmetric problem matrix
                representing hte problem data.
            save(filename: str)
                Save the problem data to a file.
                The ".npz" extension will be appended to the filename if it is not already there.
            load(filename: str)
                Load the problem data from a file.
            unpack()
                Unpack the problem data into 
                a tuple of numpy arrays (P, q, G, h, A, b, lb, ub) and return it.
            get_cute_classification(interest: str)
                Get the CUTE classification string of the problem.
                interest can be 'A', 'M', or 'R', depending on whether your problem
                is academic, part of a modeling exercise, or has real-world applications.
    results : dict
        The results of the optimization.
    '''
    def initialize(self, ):
        '''
        Initialize the Optimizer() instance for QPSolvers.
        '''
        self.solver_name = 'convex_qpsolvers-'
        self.available_solvers = qpsolvers.available_solvers
        self.supported_solvers = ['clarabel', 'cvxopt', 'daqp', 'ecos', 
                                  'gurobi', 'highs', 'hpipm', 'mosek', 'osqp', 
                                  'piqp', 'proxqp', 'qpalm', 'qpoases', 
                                  'qpswift', 'quadprog', 'scs', 'nppro']
        self.options.declare('solver_options', types=dict, default={})

        # Defined only for checking derivatives
        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.con = self.problem._compute_constraints
        self.jac = self.problem._compute_constraint_jacobian

    def setup(self, ):
        '''
        Setup the solver name, initial guess, and QP matrices and vectors.
        Check if solver is specified in the solver_options dictionary.
        Check if the gradient/Jacobian/Hessian functions are defined.
        '''
        if 'solver' in self.options['solver_options']:
            self.solver_name += self.options['solver_options']['solver']
        else:
            raise ValueError("Please specify a 'solver' in the 'solver_options' dictionary. "\
                             f"Solvers available on your machine are: {self.available_solvers}"\
                             f"Solvers supported by 'qpsolvers' are: {self.supported_solvers}")
        if 'verbose' not in self.options['solver_options']:
            self.options['solver_options']['verbose'] = True

        self.x0 = self.problem.x0 * 1.

        # Check if gradient/Jacobian/Hessian are declared and raise error/warning for Problem/ProblemLite
        # NOTE: Objective Hessian and gradient needs to declared even if running a QP feasibility problem
        self.check_if_callbacks_are_declared('grad', 'Objective gradient', 'ConvexQPSolvers')
        self.check_if_callbacks_are_declared('obj_hess', 'Objective Hessian', 'ConvexQPSolvers')
        if self.problem.constrained:
            self.check_if_callbacks_are_declared('jac', 'Constraint Jacobian', 'ConvexQPSolvers')

        # Define the cost matrix and cost vector
        self.P = self.problem._compute_objective_hessian(self.x0)
        self.q = self.problem._compute_objective_gradient(self.x0) - self.P @ self.x0
        
        con_0 = self.problem._compute_constraints(self.x0)
        jac_0 = self.problem._compute_constraint_jacobian(self.x0)

        # Define the lower bounds for the variables: lb <= x
        if np.all(self.problem.x_lower == -np.inf):
            self.lb = None
        else:
            self.lb = self.problem.x_lower

        # Define the upper bounds for the variables: x <= ub
        if np.all(self.problem.x_upper == np.inf):
            self.ub = None
        else:
            self.ub = self.problem.x_upper

        # Identify eq constraint bounds
        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper

        # Compute the constant component vector for the linear constraints
        k = con_0 - jac_0 @ self.x0

        eqi = np.where(c_lower == c_upper)[0]

        # Define the linear equality constraint: Ax = b
        if len(eqi) > 0:
            self.A = jac_0[eqi]
            self.b = c_upper[eqi] - k[eqi]
        else:
            self.A = None
            self.b = None

        # Identify constraints with only lower bounds
        lci = np.where((c_lower != -np.inf) & (c_lower != c_upper))[0]
        # Identify constraints with only upper bounds
        uci = np.where((c_upper !=  np.inf) & (c_lower != c_upper))[0]

        # Setup the linear inequality constraint: Gx <= h
        G = np.zeros((0, self.problem.nx), dtype=float)
        h = np.array([])

        if len(uci) > 0:
            G = np.append(G, jac_0[uci], axis=0)
            h = np.append(h, c_upper[uci] - k[uci])
        if len(lci) > 0:
            G = np.append(G, -jac_0[lci], axis=0)
            h = np.append(h, k[lci] - c_lower[lci])
        
        if len(lci) + len(uci) > 0:
            self.G = G
            self.h = h
        else:
            self.G = None
            self.h = None

    def solve(self, ):
        '''
        Solve the QP problem by calling qpsolvers with the requested solver and its options.
        '''
        solver_options = self.options['solver_options']
        # x = qpsolvers.solve_qp(self.P, self.q, 
        #                        self.G, self.h, 
        #                        self.A, self.b,
        #                        self.lb, self.ub,
        #                        initvals=self.x0,
        #                        **solver_options)
        # print(x) # <<-- just the solution vector x
        
        problem = qpsolvers.Problem(self.P, self.q, 
                                    self.G, self.h, 
                                    self.A, self.b,
                                    self.lb, self.ub)
        
        # We save the QP problem as an instance of qpsolvers.Problem for debugging purposes
        # and providing users with more information about the QP problem.
        self.qp_problem = problem
        # print("P", self.qp_problem.P)
        # print("q", self.qp_problem.q)
        # print("G", self.qp_problem.G)
        # print("h", self.qp_problem.h)
        # print("A", self.qp_problem.A)
        # print("b", self.qp_problem.b)
        # print("lb", self.qp_problem.lb)
        # print("ub", self.qp_problem.ub)
        # print("check_constraints", self.qp_problem.check_constraints())
        # # print("cond", self.qp_problem.cond(active_set))
        # print("get_cute_classification", self.qp_problem.get_cute_classification('M'))
        # print("has_sparse", self.qp_problem.has_sparse)
        # print("is_unconstrained", self.qp_problem.is_unconstrained)
        # print("save", self.qp_problem.save('test_qp_file'))
        # print("load", self.qp_problem.load('test_qp_file.npz'))
        # print("unpack", self.qp_problem.unpack())
        # exit()

        start_time = time.time()
        # solution returned is an instance of qpsolvers.Solution, a subclass of dataclasses.dataclass
        solution = qpsolvers.solve_problem(problem, initvals=self.x0, **solver_options)
        self.total_time = time.time() - start_time
        
        from dataclasses import asdict
        self.results = asdict(solution)
        self.results.pop('obj')         # obj returned by qpsolvers does not include the constant term so remove it
        self.results['objective']       = self.obj(solution.x)  # compute the objective value for the problem
        self.results['constraints']     = self.con(solution.x)  # compute the constraints value for the problem
        self.results['primal_residual'] = solution.primal_residual()
        self.results['dual_residual']   = solution.dual_residual()
        self.results['duality_gap']     = solution.duality_gap()
        self.results['time']            = self.total_time

        return self.results
        
    def print_results(self,
                      optimal_variables=False,
                      optimal_constraints=False,
                      optimal_dual_variables=False,
                      extras=False):

        output  = "\n\tSolution from qpsolvers:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':40}: {self.problem_name}"
        output += f"\n\t{'Solver':40}: {self.solver_name}"
        output += f"\n\t{'Found':40}: {self.results['found']}"
        output += f"\n\t{'Objective':40}: {self.results['objective']}"
        output += f"\n\t{'Dual residual':40}: {self.results['dual_residual']}"
        output += f"\n\t{'Primal residual':40}: {self.results['primal_residual']}"
        output += f"\n\t{'Duality gap':40}: {self.results['duality_gap']}"
        output += f"\n\t{'Total time':40}: {self.results['time']}"

        if optimal_variables:
            output += f"\n\t{'Optimal variables':40}: {self.results['x']}"
        if optimal_constraints:
            output += f"\n\t{'Optimal constraints':40}: {self.results['constraints']}"
        if optimal_dual_variables:
            output += f"\n\t{'Optimal dual variables (bounds)':40}: {self.results['z_box']}"
            output += f"\n\t{'Optimal dual variables (eq constraints)':40}: {self.results['y']}"
            output += f"\n\t{'Optimal dual variables (ineq cons.)':40}: {self.results['z']}"
        if extras:
            output += f"\n\t{'Extras':40}: {self.results['extras']}"

        output += '\n\t' + '-'*100

        print(output)