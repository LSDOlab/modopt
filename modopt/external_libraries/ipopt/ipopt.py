import numpy as np
from modopt import Optimizer, CSDLProblem, CSDLAlphaProblem, OpenMDAOProblem
import time

class IPOPT(Optimizer):
    '''
    Class that interfaces modOpt with the IPOPT solver in the CasADi package.
    IPOPT is an open-source interior point algorithm that can solve 
    nonlinear programming problems with equality and inequality constraints.
    It can make use of second order information in the form of the Hessian of 
    the objective for unconstrained problems or the Hessian of the Lagrangian for constrained 
    problems.

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
        If ``True``, keep the visualization window open after the optimization is complete.
    turn_off_outputs : bool, default=False
        If ``True``, prevent modOpt from generating any output files.

    solver_options : dict, default={}
        Dictionary containing the options to be passed to the solver.
        See the IPOPT page in modOpt's documentation for more information.
    '''
    def initialize(self, ):
        self.solver_name = 'ipopt'
        self.options.declare('solver_options', default={}, types=dict)
        self.modopt_default_options = {
            'sb'                        : 'yes',
            'output_file'               : 'ipopt_output.txt',
            'hessian_approximation'     : 'limited-memory',
            'print_level'               : 5,
            'file_print_level'          : 5,
            'max_iter'                  : 1000,
            'tol'                       : 1e-8,
            'linear_solver'             : 'mumps',
            'print_timing_statistics'   : 'no',

            'derivative_test'               : 'none',
            'derivative_test_print_all'     : 'no',
            'derivative_test_perturbation'  : 1e-8,
            'derivative_test_tol'           : 1e-4,
        }

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

        ipopt_options = self.modopt_default_options
        ipopt_options.update(self.options['solver_options'])
        if hasattr(self, 'out_dir'):
            ipopt_options['output_file'] = self.out_dir + '/' + ipopt_options['output_file']

        # Define the IPOPT specific options that the nlp solver need to pass to the IPOPT solver
        nlp_options = {}
        nlp_options['ipopt'] = ipopt_options

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
        try:
            from casadi import MX, nlpsol, Function, triu
        except ImportError:
            raise ImportError("'casadi' could not be imported. Install casadi using 'pip install casadi' for using IPOPT optimizer.")
        
        # Define the initial guess and bounds
        x0 = self.x0
        bounds = {'lbx': self.problem.x_lower, 
                  'ubx': self.problem.x_upper}
        options = self.nlp_options

        # Create an optimization variable
        x = MX.sym("x", self.nx)
        # Create an empty parameter variable
        p = MX.sym("p", 0, 1)
        # Create a 1x1 Lagrange multiplier variable for the objective
        lam_f = MX.sym("lf")
        # Create a ncx1 Lagrange multiplier variable for the constraints
        lam_g = MX.sym("lg", self.nc, 1)
        # Wrap the external objective function
        f = self.generate_objective_callback()
        # Define the objective expression using the callbacks (Includes objective, gradient and Hessian)
        objective_expr = f(x)

        # Create an NLP problem
        nlp = {'x': x, 'f': objective_expr}

        # Prevent the creation of the ‘nlp_grad’ function, which looks for the gradients(/Hessians if 'exact')
        # of the nlp 'f' and 'g' functions. 
        options['no_nlp_grad'] = True
        # This is required since we are providing the grad_f, jac_g and hess_lag functions explicitly
        # to avoid CasADi redundantly calling the obj/con functions when computing the grad/jac
        # and obj+grad/con+jac functions when computing the obj/lag Hessians.
        
        grad_f = self.generate_grad_f_callback()
        options['grad_f'] = Function("G",[x, p],[0, grad_f(x, p)])
        # Proper way in CasADi is shown below but this redundantly calls the objective function 
        # everytime the gradient is computed. Hence, we are using the above method.
        # options['grad_f'] = Function("G",[x, p],[objective_expr, grad_f(x, p)[0]])

        # Wrap the external Lagrangian Hessian function
        hess_lag = self.generate_hess_lag_callback()
        # Define the Lagrangian Hessian as a function in the options dictionary
        options['hess_lag'] = Function("H",[x, p, lam_f, lam_g], [triu(hess_lag(x, p, lam_f, lam_g))])

        if self.problem.constrained:
            bounds.update({'lbg': self.problem.c_lower,
                           'ubg': self.problem.c_upper})
            # Wrap the external constraint function
            c = self.generate_constraint_callback()
            # Define the constraint expression using the callbacks (includes constraints and Jacobian)
            constraint_expr = c(x)

            # Update the NLP problem
            nlp['g'] = constraint_expr

            jac_g = self.generate_jac_g_callback()
            options['jac_g'] = Function("J",[x, p],[0, jac_g(x, p)])
            # Proper way in CasADi is shown below but this redundantly calls the constraint function 
            # everytime the jacobian is computed. Hence, we are using the above method.
            # options['jac_g'] = Function("J",[x, p],[constraint_expr, jac_g(x, p)[0]])


        # Create an NLP solver
        solver = nlpsol('solver', 'ipopt', nlp, options)

        # Solve the problem
        start_time = time.time()
        results = solver(x0=x0, **bounds)
        stats = solver.stats()
        iterations  = stats.pop('iterations')
        self.total_time = time.time() - start_time

        self.results = {
            'x': np.array(results['x']).reshape((self.nx,)),
            'f': np.array(results['f'])[0],
            'c': np.array(results['g']).reshape((self.nc,)),
            'lam_c': np.array(results['lam_g']).reshape((self.nc,)),
            'lam_x': np.array(results['lam_x']).reshape((self.nx,)),
            'lam_p': np.array(results['lam_p']),
            'time' : self.total_time,
            'success': stats['return_status'] == 'Solve_Succeeded',
            # 'success': stats['return_status'] in ['Solve_Succeeded', 'Solved_To_Acceptable_Level'],
            'return_status': stats['return_status'],
            'iterations': iterations,
            'stats': stats,
            }
        
        self.run_post_processing()

        return self.results
    
    def print_results(self, 
                      optimal_variables=False,
                      optimal_constraints=False,
                      optimal_multipliers=False,
                      all=False,):
        '''
        Print the optimization results to the console.

        Parameters
        ----------
        optimal_variables : bool, default=False
            If ``True``, print the optimal variables.
        optimal_constraints : bool, default=False
            If ``True``, print the optimal constraints.
        optimal_multipliers : bool, default=False
            If ``True``, print the optimal multipliers.
        all : bool, default=False
            If ``True``, print all available information.
        '''
        output  = "\n\tSolution from ipopt:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':35}: {self.problem_name}"
        output += f"\n\t{'Solver':35}: {self.solver_name}"
        output += f"\n\t{'Success':35}: {self.results['success']}"
        output += f"\n\t{'Return status':35}: {self.results['return_status']}"
        output += f"\n\t{'Objective':35}: {self.results['f']}"
        output += f"\n\t{'Total time':35}: {self.results['time']}"
        output += self.get_callback_counts_string(35)
        
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
        from casadi import Callback, Sparsity

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

            # def has_jacobian(self): return True
            # def get_jacobian(self,name,inames,onames,opts):
            #     class GradFun(Callback):
            #         def __init__(self, opts={}):
            #             Callback.__init__(self)
            #             self.construct(name, opts)

            #         def get_n_in(self): return 2
            #         def get_n_out(self): return 1

            #         def get_sparsity_in(self,i):
            #             if i==0: # nominal input        (here, x)
            #                 return Sparsity.dense(nx,1)
            #             elif i==1: # nominal output     (here, obj)
            #                 return Sparsity.dense(1,1)
            #                 # return Sparsity(1,1)

            #         def get_sparsity_out(self,i):      # obj wrt x
            #             return Sparsity.dense(1,nx)
            #             # return sparsify(DM([[0,0,1,1],[1,0,1,0],[0,1,1,0]])).sparsity()

            #         # Evaluate numerically
            #         def eval(self, arg):
            #             # print('arg', arg)
            #             # print('arg[0]', arg[0])
            #             x = np.array(arg[0]).reshape((nx,))    # arg[0] is the input
            #             # print('x_shape', x.shape)
            #             return [grad(x).reshape((1,nx))]
                    
            #         def has_jacobian(self): return True
            #         def get_jacobian(self,name,inames,onames,opts):
            #             class HessFun(Callback):
            #                 def __init__(self, opts={}):
            #                     Callback.__init__(self)
            #                     self.construct(name, opts)

            #                 def get_n_in(self): return 3
            #                 def get_n_out(self): return 2

            #                 def get_sparsity_in(self,i):
            #                     if   i==0: return Sparsity.dense(nx,1)  # x
            #                     elif i==1: return Sparsity.dense(1,1)   # obj
            #                     elif i==2: return Sparsity.dense(nx,1)  # grad
                                
            #                 def get_sparsity_out(self,i):
            #                     if i==0: return Sparsity.dense(nx,nx)   # grad wrt x
            #                     if i==1: return Sparsity.dense(nx,1)    # grad wrt obj = 0
                            
            #                 def eval(self, arg):
            #                     x = np.array(arg[0]).reshape((nx,))
            #                     return [hess(x), np.zeros((nx,1))]
                            
            #             self.hess_callback = HessFun()
            #             return self.hess_callback

            #     # You are required to keep a reference alive to the returned Callback object
            #     self.grad_callback = GradFun()
            #     return self.grad_callback
            
        return Objective('f')
    
    def generate_constraint_callback(self,):
        from casadi import Callback, Sparsity

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

            # def has_jacobian(self): return True
            # def get_jacobian(self,name,inames,onames,opts):
            #     class JacFun(Callback):
            #         def __init__(self, opts={}):
            #             Callback.__init__(self)
            #             self.construct(name, opts)

            #         def get_n_in(self): return 2
            #         def get_n_out(self): return 1

            #         def get_sparsity_in(self,i):
            #             if i==0: # nominal input
            #                 return Sparsity.dense(nx,1)
            #             elif i==1: # nominal output
            #                 return Sparsity.dense(nc,1)
            #                 # return Sparsity(1,1)

            #         def get_sparsity_out(self,i):
            #             return Sparsity.dense(nc,nx)
            #             # return sparsify(DM([[0,0,1,1],[1,0,1,0],[0,1,1,0]])).sparsity()

            #         # Evaluate numerically
            #         def eval(self, arg):
            #             # print('arg', arg)
            #             # print('arg[0]', arg[0])
            #             x = np.array(arg[0]).reshape((nx,))    # arg[0] is the input
            #             # print('x', x)
            #             # print('x[0]', x[0])
            #             # print(np.array([[1, 1], [2*x[0], -1]]))
            #             return [jac(x)]

            #     # You are required to keep a reference alive to the returned Callback object
            #     self.jac_callback = JacFun()
            #     return self.jac_callback
            
        return Constraints('c')

    def generate_hess_lag_callback(self,):
        from casadi import Callback, Sparsity

        nx = self.nx
        nc = self.nc
        lag_hess = self.problem._compute_lagrangian_hessian
        obj_hess = self.problem._compute_objective_hessian
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
                if nc == 0:
                    hess_lag = obj_hess(x) * lam_f
                else:
                    lam_g = np.array(arg[3]).reshape((nc,)) # arg[3] is the lagrange multiplier for the constraints
                    if lam_f != 0:
                        z = lam_g / lam_f
                        hess_lag = (lag_hess(x, z)) * lam_f
                    else:
                        hess_lag = lag_hess(x, lam_g) - lag_hess(x, np.zeros((nc,)))

                return [hess_lag]
            
        return LagrangianHessian('HessLag')
    
    def generate_grad_f_callback(self,):
        from casadi import Callback, Sparsity

        nx = self.nx
        grad = self.problem._compute_objective_gradient
        class ObjectiveGradient(Callback):
            def __init__(self, name, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i): # args = [x, p(parameters)]
                if i==0:
                    return Sparsity.dense(nx,1)
                elif i==1:
                    return Sparsity.dense(0,0)

            def get_sparsity_out(self,i):
                return Sparsity.dense(1,nx)

            # Evaluate numerically
            def eval(self, arg):
                x = np.array(arg[0]).reshape((nx,))     # arg[0] is the decision variable
                return [grad(x).reshape((1,nx))]
            
        return ObjectiveGradient('GradF')
    
    def generate_jac_g_callback(self,):
        from casadi import Callback, Sparsity

        nx = self.nx
        nc = self.nc
        jac = self.problem._compute_constraint_jacobian
        class ConstraintJacobian(Callback):
            def __init__(self, name, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i): # args = [x, p(parameters)]
                if i==0:
                    return Sparsity.dense(nx,1)
                elif i==1:
                    return Sparsity.dense(0,0)

            def get_sparsity_out(self,i):
                return Sparsity.dense(nc,nx)

            # Evaluate numerically
            def eval(self, arg):
                x = np.array(arg[0]).reshape((nx,))     # arg[0] is the decision variable
                return [jac(x)]
            
        return ConstraintJacobian('JacG')