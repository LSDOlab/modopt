import numpy as np
from modopt import Optimizer, CSDLProblem, CSDLAlphaProblem, OpenMDAOProblem
import warnings
import time
try:
    from casadi import *
except:
    warnings.warn("'casadi' could not be imported. Install casadi using 'pip install casadi' for using IPOPT optimizer.")

class IPOPT(Optimizer):
    '''
    Class that interfaces modOpt with the IPOPT in the CasADi package.
    IPOPT is an open-source interior point algorithm that can solve 
    nonlinear programming problems with equality and inequality constraints.
    It can make use of second order information in the form of the Hessian of 
    the objective for unconstrained problems or the Hessian of the Lagrangian for constrained 
    problems.
    '''
    def initialize(self, ):
        self.solver_name = 'ipopt'
        self.available_options = {'print_level': 5, 'linear_solver': 'ma57', 'tol': 1e-8, 'max_iter': 1000}
        # Options to be passed to IPOPT solver in CasADi, 
        # e.g., {'print_level': 5, 'linear_solver': 'ma57', 'tol': 1e-8, 'max_iter': 1000, file_print_level: 5,
        # 'hessian_approximation': 'limited-memory'}
        self.options.declare('solver_options', default={}, types=dict)

        # Declare outputs
        self.available_outputs = {}
        self.options.declare('readable_outputs', values=([],), default=[])
        
        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.obj_hess = self.problem._compute_objective_hessian
        self.active_callbacks = ['obj', 'grad']
        if self.problem.constrained:
            self.con = self.problem._compute_constraints
            self.jac = self.problem._compute_constraint_jacobian
            self.lag_hess = self.problem._compute_lagrangian_hessian
            self.active_callbacks += ['con', 'jac']

    def setup(self):
        '''
        Setup the initial guess, and matrices and vectors.
        '''
        self.x0 = self.problem.x0 * 1.
        self.nx = self.problem.nx * 1
        self.nc = self.problem.nc * 1

        # Define the IPOPT specific options that the nlp solver need to pass to the IPOPT solver
        nlp_options = {'ipopt': self.options['solver_options']}
        # By default, switch to first-order information only
        # [for using exact obj/lag Hessian, user has to set 'hessian_approximation': 'exact' in solver_options]
        if 'hessian_approximation' not in nlp_options['ipopt']:
            nlp_options['ipopt']['hessian_approximation'] = 'limited-memory'

        # Check if the user has declared the callbacks for the Hessian when using 'exact' Hessian approximation
        if nlp_options['ipopt']['hessian_approximation'] == 'exact':
            if isinstance(self.problem, (CSDLProblem, CSDLAlphaProblem, OpenMDAOProblem)):
                raise ValueError("Exact Hessians are not available with 'OpenMDAOProblem', 'CSDLProblem' or 'CSDLAlphaProblem'.")
            else:
                if not self.problem.constrained:
                    self.check_if_callbacks_are_declared('obj_hess', 'Objective Hessian', 'IPOPT')
                    self.active_callbacks += ['obj_hess']
                else:
                    self.check_if_callbacks_are_declared('lag_hess', 'Lagrangian Hessian', 'IPOPT')
                    self.active_callbacks += ['lag_hess']

        self.nlp_options = nlp_options

    def setup_constraints(self, ):
        pass

    def solve(self):
        # Define the initial guess and bounds
        x0 = self.x0
        lbx = self.problem.x_lower
        ubx = self.problem.x_upper
        options = self.nlp_options

        # Create an optimization variable
        x = MX.sym("x", self.nx)
        # Wrap the external objective function
        f = self.generate_objective_callback()
        # Define the objective expression using the callbacks (Includes objective, gradient and Hessian)
        objective_expr = f(x)

        if self.problem.constrained:
            lbg = self.problem.c_lower
            ubg = self.problem.c_upper
            # Wrap the external constraint function
            c = self.generate_constraint_callback()
            # Define the constraint expression using the callbacks (includes constraints and Jacobian)
            constraint_expr = c(x)

            # Create an NLP problem
            nlp = {'x': x, 'f': objective_expr, 'g': constraint_expr}

            # Create an empty parameter variable
            p = MX.sym("p", 0, 1)
            # Create a 1x1 Lagrange multiplier variable for the objective
            lam_f = MX.sym("lf")
            # Create a 1x1 Lagrange multiplier variable for the objective
            lam_g = MX.sym("lg", self.nc, 1)
            # Wrap the external Lagrangian Hessian function
            hess_lag = self.generate_hess_lag_callback()
            # Define the Lagrangian Hessian as a function in the options dictionary
            options['hess_lag'] = Function("H",[x, p, lam_f, lam_g],[hess_lag(x, p, lam_f, lam_g)[0]])

            # Create an NLP solver
            solver = nlpsol('solver', 'ipopt', nlp, options)

            start_time = time.time()

            # Solve the problem
            results = solver(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

        else:
            # Create an NLP problem
            nlp = {'x': x, 'f': objective_expr}

            # Create an NLP solver
            solver = nlpsol('solver', 'ipopt', nlp, options)

            start_time = time.time()

            # Solve the problem
            results = solver(x0=x0, lbx=lbx, ubx=ubx)

        self.total_time = time.time() - start_time

        self.results = {
            'x': np.array(results['x']).reshape((self.nx,)),
            'f': np.array(results['f'])[0],
            'c': np.array(results['g']).reshape((self.nc,)),
            'lam_c': np.array(results['lam_g']).reshape((self.nc,)),
            'lam_x': np.array(results['lam_x']).reshape((self.nx,)),
            'lam_p': np.array(results['lam_p']),
            'time' : self.total_time,
            }
        
        self.run_post_processing()

        return self.results
    
    def print_results(self, 
                      optimal_variables=False,
                      optimal_constraints=False,
                      optimal_multipliers=False,
                      all=False,):
        
        output  = "\n\tSolution from ipopt:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':35}: {self.problem_name}"
        output += f"\n\t{'Solver':35}: {self.solver_name}"
        output += f"\n\t{'Objective':35}: {self.results['f']}"
        output += f"\n\t{'Total time':35}: {self.results['time']}"
        
        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':35}: {self.results['x']}"
        if optimal_constraints or all:
            output += f"\n\t{'Optimal constraints':35}: {self.results['c']}"        
        if optimal_multipliers or all:
            output += f"\n\t{'Optimal multipliers (bounds)':35}: {self.results['lam_x']}"       
            output += f"\n\t{'Optimal multipliers (constr.)':35}: {self.results['lam_c']}"

        output += '\n\t' + '-'*100
        print(output)
        
    def generate_objective_callback(self,):
        nx   = self.nx
        obj  = self.obj
        grad = self.grad
        hess = self.obj_hess
        class Objective(Callback):
            def __init__(self, name, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 1
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                return Sparsity.dense(nx,1)

            def get_sparsity_out(self,i):
                return Sparsity.dense(1,1)

            # Evaluate numerically
            def eval(self, arg):
                # print('arg', arg)
                # print('arg[0]', arg[0])
                x = np.array(arg[0]).reshape((nx,))    # arg[0] is the input
                # print(x.shape)
                return [obj(x)]

            def has_jacobian(self): return True
            def get_jacobian(self,name,inames,onames,opts):
                class GradFun(Callback):
                    def __init__(self, opts={}):
                        Callback.__init__(self)
                        self.construct(name, opts)

                    def get_n_in(self): return 2
                    def get_n_out(self): return 1

                    def get_sparsity_in(self,i):
                        if i==0: # nominal input        (here, x)
                            return Sparsity.dense(nx,1)
                        elif i==1: # nominal output     (here, obj)
                            return Sparsity.dense(1,1)
                            # return Sparsity(1,1)

                    def get_sparsity_out(self,i):      # obj wrt x
                        return Sparsity.dense(1,nx)
                        # return sparsify(DM([[0,0,1,1],[1,0,1,0],[0,1,1,0]])).sparsity()

                    # Evaluate numerically
                    def eval(self, arg):
                        # print('arg', arg)
                        # print('arg[0]', arg[0])
                        x = np.array(arg[0]).reshape((nx,))    # arg[0] is the input
                        # print('x_shape', x.shape)
                        return [grad(x).reshape((1,nx))]
                    
                    def has_jacobian(self): return True
                    def get_jacobian(self,name,inames,onames,opts):
                        class HessFun(Callback):
                            def __init__(self, opts={}):
                                Callback.__init__(self)
                                self.construct(name, opts)

                            def get_n_in(self): return 3
                            def get_n_out(self): return 2

                            def get_sparsity_in(self,i):
                                if   i==0: return Sparsity.dense(nx,1)  # x
                                elif i==1: return Sparsity.dense(1,1)   # obj
                                elif i==2: return Sparsity.dense(nx,1)  # grad
                                
                            def get_sparsity_out(self,i):
                                if i==0: return Sparsity.dense(nx,nx)   # grad wrt x
                                if i==1: return Sparsity.dense(nx,1)    # grad wrt obj = 0
                            
                            def eval(self, arg):
                                x = np.array(arg[0]).reshape((nx,))
                                return [hess(x), np.zeros((nx,1))]
                            
                        self.hess_callback = HessFun()
                        return self.hess_callback

                # You are required to keep a reference alive to the returned Callback object
                self.grad_callback = GradFun()
                return self.grad_callback
            
        return Objective('f')
    
    def generate_constraint_callback(self,):
        nx = self.nx
        nc = self.nc
        con = self.con
        jac = self.jac
        class Constraints(Callback):
            def __init__(self, name, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 1
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                return Sparsity.dense(nx,1)

            def get_sparsity_out(self,i):
                return Sparsity.dense(nc,1)

            # Evaluate numerically
            def eval(self, arg):
                # print('arg', arg)
                # print('arg[0]', arg[0])
                x = np.array(arg[0]).reshape((nx,))    # arg[0] is the input
                # print(x.shape)
                return [con(x)]

            def has_jacobian(self): return True
            def get_jacobian(self,name,inames,onames,opts):
                class JacFun(Callback):
                    def __init__(self, opts={}):
                        Callback.__init__(self)
                        self.construct(name, opts)

                    def get_n_in(self): return 2
                    def get_n_out(self): return 1

                    def get_sparsity_in(self,i):
                        if i==0: # nominal input
                            return Sparsity.dense(nx,1)
                        elif i==1: # nominal output
                            return Sparsity.dense(nc,1)
                            # return Sparsity(1,1)

                    def get_sparsity_out(self,i):
                        return Sparsity.dense(nc,nx)
                        # return sparsify(DM([[0,0,1,1],[1,0,1,0],[0,1,1,0]])).sparsity()

                    # Evaluate numerically
                    def eval(self, arg):
                        # print('arg', arg)
                        # print('arg[0]', arg[0])
                        x = np.array(arg[0]).reshape((nx,))    # arg[0] is the input
                        # print('x', x)
                        # print('x[0]', x[0])
                        # print(np.array([[1, 1], [2*x[0], -1]]))
                        return [jac(x)]

                # You are required to keep a reference alive to the returned Callback object
                self.jac_callback = JacFun()
                return self.jac_callback
            
        return Constraints('c')

    def generate_hess_lag_callback(self,):
        nx = self.nx
        nc = self.nc
        lag_hess = self.problem._compute_lagrangian_hessian
        # obj_hess = self.problem._compute_objective_hessian
        class LagrangianHessian(Callback):
            def __init__(self, name, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 4
            def get_n_out(self): return 1

            def get_sparsity_in(self,i): # args = [x, p(parameters), lam_f, lam_g]
                if i==0:
                    return Sparsity.dense(nx,1)
                elif i==1:
                    return Sparsity.dense(0,0)
                elif i==2:
                    return Sparsity.dense(1,1)
                elif i==3:
                    return Sparsity.dense(nc,1)

            def get_sparsity_out(self,i):
                return Sparsity.dense(nx,nx)

            # Evaluate numerically
            def eval(self, arg):
                x = np.array(arg[0]).reshape((nx,))     # arg[0] is the decision variable
                lam_f = np.array(arg[2]).reshape((1,))  # arg[2] is the lagrange multiplier for the objective
                # if nc == 0:
                #     # hess_lag = (lag_hess(x, np.array([]))) * lam_f
                #     hess_lag = (obj_hess(x)) * lam_f
                # else:
                lam_g = np.array(arg[3]).reshape((nc,)) # arg[3] is the lagrange multiplier for the constraints
                z = lam_g / lam_f
                hess_lag = (lag_hess(x, z)) * lam_f

                return [hess_lag]
            
        return LagrangianHessian('HessLag')