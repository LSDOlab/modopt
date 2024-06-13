from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name

import numpy as np
import warnings
import time

class ProblemLite(object):
    '''
    Lightweight base class for defining optimization problems in modOpt.
    Performs basic setup for optimization problems, without using array_manager.
    This class is useful for defining simple optimization problems
    with initial design variables, objective, constraints and their derivative functions.
    The ProblemLite() object can be used when the user wants to call the optimizer directly with
    x0, obj, con, grad, jac, obj_hess, etc. functions.

    Major differences from Problem() class:
        - No array_manager objects are used so no setup of matrices, vectors, etc.
        - No declarations of design variables, objectives, constraints, etc.
        - No setup() or setup_derivatives() method is called.
        - Functions and derivatives are directly called from the user-provided functions (thin wrapper).
        - Only single objective problems are supported.
        - Only a single design varaible vector and single constraint vector function are supported.
        - Every STORED VARIABLE in this class is SCALED by the user-provided scaling factors.
        - Objective and constraint functions are always called together in _funcs(x) method.
        - Gradient and Jacobian functions are always called together in _derivs(x) method.
        - Caches the function and first derivative values for the same input x to avoid redundant, consecutive evaluations.
        - Keeps track of the number of function and gradient evaluations and time taken for each.

    Still supports:
        - Feasibility problems and unconstrained optimization problems.
        - Finite differencing by default for unavailable derivatives.
        - Matrix-vector products vjp, jvp, obj_hvp, lag_hvp.
        - Caching of function and derivative values (although scaled) for the same input x.
        - Scaling of design variables, objectives, and constraints.
        - Bounds on design variables and constraints.
        - Lagrangian functions and derivatives for constrained problems.
        - Sparse or dense matrix formats for Jacobian and Hessian in user-provided functions, 
          if supported by the optimizer used.

    Attributes
    ----------
    name : str, default='unnamed_problem'
        Problem name assigned by the user.
    x0 : np.ndarray
        Initial guess for design variables.
    obj : callable
        Objective function.
    con : callable
        Constraints function.
    grad : callable
        Gradient of the objective function.
    jac : callable
        Jacobian of the constraints function.
    obj_hess : callable
        Hessian of the objective function.
    lag_hess : callable
        Hessian of the Lagrangian function.
    fd_step : float
        Finite difference step size for gradient and Jacobian computations.
    xl : float or np.ndarray
        Lower bounds on design variables.
    xu : float or np.ndarray
        Upper bounds on design variables.
    cl : float or np.ndarray
        Lower bounds on constraints.
    cu : float or np.ndarray
        Upper bounds on constraints.
    x_scaler : float or np.ndarray
        Scaling factor for design variables.
    f_scaler : float
        Scaling factor for the objective function.
    c_scaler : float or np.ndarray
        Scaling factor for constraints.
    jvp : callable
        Jacobian-vector product function.
    vjp : callable
        Vector-Jacobian product function.
    obj_hvp : callable
        Hessian-vector product function for the objective.
    lag_hvp : callable
        Hessian-vector product function for the Lagrangian.
    '''

    def __init__(self, x0, name='unnamed_problem', obj=None, con=None, grad=None, jac=None, obj_hess=None, lag_hess=None, 
                 fd_step=1e-8, xl=None, xu=None, cl=None, cu=None, x_scaler=1., f_scaler=1., c_scaler=1.,
                 jvp=None, vjp=None, obj_hvp=None, lag_hvp=None):
        self.problem_name = name
        self.options = OptionsDictionary()
        self.ny = 0
        self.nx = nx = x0.size
        self.fd_step = fd_step * 1.0

        self.f_scaler = f_scaler * np.ones((1,))
        self.x_scaler = x_scaler * np.ones((nx,))

        self.x0 = np.asfarray(x0) * x_scaler
        self.x_lower = xl * x_scaler if xl is not None else np.full(nx, -np.inf)
        self.x_upper = xu * x_scaler if xu is not None else np.full(nx,  np.inf)

        self.jvp = jvp
        self.vjp = vjp
        self.obj_hvp = obj_hvp
        self.lag_hvp = lag_hvp

        if obj is not None:
            self.obj = obj
        else:
            print('Objective function not provided. Running a feasibility problem.')
            self.obj = lambda x: 0.0
            self.grad = lambda x: np.zeros((self.nx,))
            self.obj_hess = lambda x: np.zeros((self.nx, self.nx))
        
        self.constrained = False
        self.nc = 0
        self.con = con
        if con is not None:
            self.constrained = True

        if grad is not None:
            self.grad = grad
        elif obj is not None:
            # FD gradient
            def fd_grad(x):
                f0 = self.obj(x)
                return np.array([(self.obj(x + fd_step*np.eye(self.nx)[:,i]) - f0) / fd_step for i in range(self.nx)])
            self.grad = fd_grad
        
        self.jac = jac
        if (con is not None) and (jac is None):
            # FD Jacobian
            def fd_jac(x):
                c0 = self.con(x)
                return np.array([(self.con(x + fd_step*np.eye(self.nx)[:,i]) - c0) / fd_step for i in range(self.nx)]).T
            self.jac = fd_jac
            
        if con is not None:
            self.lag = lambda x, mu: self.obj(x) + np.dot(mu, self.con(x))
            self.lag_grad = lambda x, mu: self.grad(x) + np.dot(mu, self.jac(x))

        if obj_hess is not None:
            self.obj_hess = obj_hess
        elif obj is not None:
            # FD Hessian
            def fd_obj_hess(x):
                g0 = self.grad(x)
                return np.array([(self.grad(x + fd_step*np.eye(self.nx)[:,i]) - g0) / fd_step for i in range(self.nx)])
            self.obj_hess = fd_obj_hess

        if con is not None:
            if lag_hess is not None:
                self.lag_hess = lag_hess
            else:
                # FD Lagrangian Hessian
                def fd_lag_hess(x, mu):
                    lg0 = self.lag_grad(x, mu)
                    return np.array([(self.lag_grad(x + fd_step*np.eye(self.nx)[:,i], mu) - lg0) / fd_step for i in range(self.nx)])
                self.lag_hess = fd_lag_hess

        self.nfev = 0
        self.ngev = 0
        self.fev_time = 0.0
        self.gev_time = 0.0
        self.warm_x = self.x = np.random.rand(self.nx)
        self.warm_x_derivs = np.random.rand(self.nx)
        # self.warm_x_2nd_derivs = np.random.rand(self.nx) # No caching of 2nd derivatives since memory expensive
        self.warm_mu = self.mu = np.zeros((self.nc,)) if self.constrained else None

        # Cached f, c, g, j are all scaled
        self.f = None
        self.g = None
        self.c = None
        self.j = None

        # For the first call, we expand _funcs(x0) below to avoid redundant calls to con(x0) 
        # to get the size of the constraints nc and to perform checks on sizes of functions.
        # self.c_scaler = c_scaler * 1.
        # self._funcs(self.x0)
        # if self.constrained:
        #     self.nc = nc = self.c.size

        # NOTE: x0 is unscaled while self.x0 is scaled
        ###### FIRST RUN FOR OBJECTIVE AND CONSTRAINTS ######
        f_start = time.time()
        f0 = self.obj(x0)
        if not np.isscalar(f0):
            if f0.shape != (1,):
                raise ValueError('Objective function must return a scalar or a 1D array with shape (1,).')
        self.f = (f0 * self.f_scaler)[0]
        if self.constrained:
            c0 = self.con(x0)
            self.nc = nc = c0.size
            if c0.shape != (nc,):
                raise ValueError(f'Constraints function must return a 1D array with shape (nc,) '
                                 f'where nc={nc} is the number of constraints.')
            self.c = c0 * c_scaler
        self.warm_x[:] = self.x0
        self.nfev += 1
        self.fev_time += time.time() - f_start
        #####################################################

        # Once nc is known from the first call to con(x0), update the cl, cu, c_scaler sizes
        if self.constrained:
            self.c_scaler = c_scaler * np.ones((nc,))
            self.c_lower = cl * c_scaler if cl is not None else np.full(nc, -np.inf)
            self.c_upper = cu * c_scaler if cu is not None else np.full(nc,  np.inf)
        else:
            self.c_lower = self.c_upper = self.c_scaler = None

        # For the first call, we expand _derivs(x0) here to perform checks on sizes of derivatives returned.
        # self._derivs(self.x0)

        ###### FIRST RUN FOR GRADIENT AND JACOBIAN ######
        g_start = time.time()
        g0 = self.grad(x0) 
        if g0.shape != (nx,):
            raise ValueError(f'Gradient function must return a 1D array with shape (nx,) '
                             f'where nx={nx} is the number of design variables, '
                             f'but returned shape {g0.shape}.')
        self.g = g0 * self.f_scaler / self.x_scaler
        if self.constrained:
            j0 = self.jac(x0)
            if j0.shape != (nc, nx):
                raise ValueError(f'Jacobian function must return a 2D array with shape (nc, nx) '
                                 f'where nc={nc} is the number of constraints and nx={nx} is the number of design variables, '
                                 f'but returned shape {j0.shape}.')
            self.j = j0 * np.outer(self.c_scaler, 1 / self.x_scaler)
        self.warm_x_derivs[:] = self.x0
        self.ngev += 1
        self.gev_time += time.time() - g_start
        #####################################################

        self.check_for_errors(x0, xl, xu, cl, cu, x_scaler, f_scaler, c_scaler)

    def check_for_errors(self, x0, xl, xu, cl, cu, x_scaler, f_scaler, c_scaler):
        '''
        Check for errors in the optimization problem setup.
        '''
        if self.nx == 0:
            raise ValueError('No design variables declared.')
        if x0.shape != (self.nx,):
            raise ValueError(f'The initial guess vector must be a 1D array but provided x0 has shape {x0.shape}.')
        if self.x_lower.shape != (self.nx,):
            raise ValueError(f'The lower bounds vector must be broadcastable to shape ({self.nx},) but got shape {xl.shape}.')
        if self.x_upper.shape != (self.nx,):
            raise ValueError(f'The upper bounds vector must be broadcastable to shape ({self.nx},) but got shape {xu.shape}.')
        if self.x_scaler.shape != (self.nx,):
            raise ValueError(f'The design variable scaler vector must be broadcastable to shape ({self.nx},) but got shape {x_scaler.shape}.')
        if not np.isscalar(f_scaler):
            if f_scaler.shape != (1,):
                raise ValueError(f'The objective scaler must be a scalar or a 1D array with shape (1,) but got shape {f_scaler.shape}.')
            
        if self.constrained:
            if self.nc == 0:
                raise ValueError('Constraints are declared but the number of constraints is zero.')
            if self.c_lower.shape != (self.nc,):
                raise ValueError(f'Lower bounds vector for constraints must be broadcastable to shape ({self.nc},) but got shape {cl.shape}.')
            if self.c_upper.shape != (self.nc,):
                raise ValueError(f'Upper bounds vector for constraints must be broadcastable to shape ({self.nc},) but got shape {cu.shape}.')
            if self.c_scaler.shape != (self.nc,):
                raise ValueError(f'The constraint scaler vector must be broadcastable to shape ({self.nc},) but got shape {c_scaler.shape}.')
        else:
            if cl is not None or cu is not None or c_scaler != 1.0:
                raise ValueError('If "con" function is not provided, "cl", "cu", and "c_scaler" must not be declared.')
            if self.jvp is not None or self.vjp is not None or self.lag_hvp is not None:
                raise ValueError('If "con" function is not provided, "jvp", "vjp", and "lag_hvp" must not be declared.')

    def _funcs(self, x):
        '''
        Compute the objective and constraints at the given x, if x is different from the previous x.
        '''
        if not np.array_equal(x, self.warm_x):
            f_start = time.time()
            self.f = (self.obj(x/self.x_scaler) * self.f_scaler)[0]
            if self.constrained:
                self.c = self.con(x/self.x_scaler) * self.c_scaler
            self.warm_x[:] = x
            self.nfev += 1
            self.fev_time += time.time() - f_start
        
    def _derivs(self, x):
        '''
        Compute the gradient and Jacobian at the given x, if x is different from the previous x.
        '''
        if not np.array_equal(x, self.warm_x_derivs):
            g_start = time.time()
            self.g = self.grad(x/self.x_scaler) * self.f_scaler / self.x_scaler
            if self.constrained:
                self.j = self.jac(x/self.x_scaler) * np.outer(self.c_scaler, 1 / self.x_scaler)
            self.warm_x_derivs[:] = x
            self.ngev += 1
            self.gev_time += time.time() - g_start

    def _compute_objective(self, x):
        '''
        Compute the (scaled) objective function at the given x.
        '''
        self._funcs(x)
        return self.f
    
    def _compute_objective_gradient(self, x):
        '''
        Compute the (scaled) gradient of the objective function at the given x.
        '''
        self._derivs(x)
        return self.g
    
    def _compute_constraints(self, x):
        '''
        Compute the (scaled) constraints at the given x.
        '''
        self._funcs(x)
        return self.c
    
    def _compute_constraint_jacobian(self, x):
        '''
        Compute the (scaled) Jacobian of the constraints at the given x.
        '''
        self._derivs(x)
        return self.j
    
    def _compute_objective_hessian(self, x):
        '''
        Compute the (scaled) Hessian of the objective at the given x.
        '''
        return self.obj_hess(x/self.x_scaler) * self.f_scaler * np.outer(1 / self.x_scaler, 1 / self.x_scaler)
    
    def _compute_lagrangian_hessian(self, x, mu):
        '''
        Compute the (scaled) Hessian of the Lagrangian at the given x and mu.
        '''
        self.warm_mu[:] = mu
        return self.lag_hess(x/self.x_scaler, mu*self.c_scaler/self.f_scaler) * self.f_scaler * np.outer(1 / self.x_scaler, 1 / self.x_scaler)
    
    def _compute_lagrangian(self, x, mu):
        '''
        Compute the (scaled) Lagrangian at the given x and mu.
        '''
        # return self.lag(x, mu)

        self._funcs(x)
        self.warm_mu[:] = mu
        return self.f + np.dot(mu, self.c)
    
    def _compute_lagrangian_gradient(self, x, mu):
        '''
        Compute the (scaled) gradient of the Lagrangian at the given x and mu.
        '''
        # return self.lag_grad(x, mu)
        
        self._derivs(x)
        self.warm_mu[:] = mu
        return self.g + np.dot(mu, self.j)
    
    def _compute_constraint_jvp(self, x, v):
        '''
        Compute the (scaled) Jacobian-vector product at the given x and v.
        '''
        if self.jvp is not None:
            return self.jvp(x/self.x_scaler, v/self.x_scaler) * self.c_scaler
        else:
            # warnings.warn("No constraint JVP function is declared. Computing full Jacobian to get JVP.")
            # j = self.jac(x/self.x_scaler) * np.outer(self.c_scaler, 1 / self.x_scaler)
            # return j @ v

            warnings.warn("No constraint JVP function is declared. Using finite differences.")
            c0 = self.con(x/self.x_scaler)
            h = self.fd_step * v / self.x_scaler
            c1 = self.con(x/self.x_scaler + h)
            fd_jvp = (c1 - c0) / self.fd_step
            return fd_jvp * self.c_scaler
    
    def _compute_constraint_vjp(self, x, v):
        '''
        Compute the (scaled) vector-Jacobian product at the given x and v.
        '''
        if self.vjp is not None:
            return self.vjp(x/self.x_scaler, v*self.c_scaler) / self.x_scaler
        else:
            warnings.warn("No constraint VJP function is declared. Computing full Jacobian to get VJP.")
            j = self.jac(x/self.x_scaler) * np.outer(self.c_scaler, 1 / self.x_scaler)
            return v @ j
        
    def _compute_objective_hvp(self, x, v):
        '''
        Compute the (scaled) objective Hessian-vector product at the given x and v.
        '''
        if self.obj_hvp is not None:
            return self.obj_hvp(x/self.x_scaler, v/self.x_scaler) * self.f_scaler / self.x_scaler
        else:
            # warnings.warn("No objective HVP function is declared. Computing full objective Hessian to get obj_HVP.")
            # oh = self.obj_hess(x/self.x_scaler) * f_scaler * np.outer(1 / self.x_scaler, 1 / self.x_scaler)
            # return oh @ v

            warnings.warn("No objective HVP function is declared. Using finite differences.")
            g0 = self.grad(x/self.x_scaler)
            h = self.fd_step * v / self.x_scaler
            g1 = self.grad(x/self.x_scaler + h)
            fd_hvp = (g1 - g0) / self.fd_step
            return fd_hvp * self.f_scaler / self.x_scaler
        
    def _compute_lagrangian_hvp(self, x, v, mu):
        '''
        Compute the (scaled) Lagrangian Hessian-vector product at the given x, v, and mu.
        '''
        self.warm_mu[:] = mu

        if self.lag_hvp is not None:
            return self.lag_hvp(x/self.x_scaler, v/self.x_scaler, mu*self.c_scaler/self.f_scaler) * self.f_scaler / self.x_scaler
        else:
            # warnings.warn("No Lagrangian HVP function is declared. Computing full lagrangian Hessian to get lag_HVP.")
            # lh = self.lag_hess(x/self.x_scaler, mu*self.c_scaler/self.f_scaler) * self.f_scaler * np.outer(1 / self.x_scaler, 1 / self.x_scaler)
            # return lh @ v

            warnings.warn("No Lagrangian HVP function is declared. Using finite differences.")
            lg0 = self.lag_grad(x/self.x_scaler, mu*self.c_scaler/self.f_scaler)
            h = self.fd_step * v / self.x_scaler
            lg1 = self.lag_grad(x/self.x_scaler + h, mu*self.c_scaler/self.f_scaler)
            fd_hvp = (lg1 - lg0) / self.fd_step
            return fd_hvp * self.f_scaler / self.x_scaler
        
    def compute_objective(self, dvs, obj):
        pass
    def compute_objective_gradient(self, dvs, grad):
        pass
    def compute_constraints(self, dvs, con):
        pass
    def compute_constraint_jacobian(self, dvs, jac):
        pass
    def compute_objective_hessian(self, dvs, obj_hess):
        pass
    def compute_lagrangian_hessian(self, dvs, z, obj_hess):
        pass

    def __str__(self):
        """
        Print the details of the UNSCALED optimization problem.
        """
        name = self.problem_name
        obj  = (self.f / self.f_scaler)[0] # self.f_scaler always has shape (1,)
        obj_scaler = self.f_scaler[0]
        dvs  = self.warm_x / self.x_scaler
        x_l  = self.x_lower / self.x_scaler
        x_u  = self.x_upper / self.x_scaler
        x_s  = self.x_scaler
        if self.constrained:
            cons = self.c / self.c_scaler
            c_l  = self.c_lower / self.c_scaler
            c_u  = self.c_upper / self.c_scaler
            c_s  = self.c_scaler
            mu = self.warm_mu

        # Print title : optimization problem name
        # title   = f'Optimization problem : {name}'
        # output  = f'\n\n{title}\n'
        # output += '='*100
        
        # PROBLEM OVERVIEW BEGINS
        # >>>>>>>>>>>>>>>>>>>>>>>
        title1  = 'Problem Overview :'
        # output  = '\n\t'+'-'*100
        output  = f'\n\t{title1}\n'
        output += '\t'+'-'*100
        output += f'\n\t' + pad_name('Problem name', 25) + f': {name}'
        output += f'\n\t' + pad_name('Objectives', 25) + f': obj'
        output += f'\n\t' + pad_name('Design variables', 25) + f': x -> {dvs.shape}'
        if self.constrained:
            output += f'\n\t' + pad_name('Constraints', 25) + f': con -> {cons.shape}'

        # <<<<<<<<<<<<<<<<<<<<<
        # PROBLEM OVERVIEW ENDS
        output += '\n\t' + '-'*100
        
        # PROBLEM DATA BEGINS
        # >>>>>>>>>>>>>>>>>>>>>>>
        title2 = 'Problem Data (UNSCALED):'
        output += f'\n\n\t{title2}\n'
        output += '\t' + '-'*100
        
        # Print objective data
        output += f'\n\tObjectives:\n'
        header = "\t{0} | {1} | {2} | {3}".format(
                                            pad_name('Index', 5),
                                            pad_name('Name', 10),
                                            pad_name('Scaler', 13),
                                            pad_name('Value', 13),
                                        )
        output += header
        obj_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {value:<+.6e}"
        output     += obj_template.format(idx=0, name='obj', scaler=obj_scaler, value=obj)
        
        # Print design variable data
        output += f'\n\n\tDesign Variables:\n'
        header = "\t{0} | {1} | {2} | {3} | {4} | {5}".format(
                                                        pad_name('Index', 5),
                                                        pad_name('Name', 10),
                                                        pad_name('Scaler', 13),
                                                        pad_name('Lower Limit', 13),
                                                        pad_name('Value', 13),
                                                        pad_name('Upper Limit', 13),
                                                        )
        output += header
        dv_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e}"

        for i, x in enumerate(dvs.flatten()):
            l = -1.e99 if x_l[i] == -np.inf else x_l[i]
            u = +1.e99 if x_u[i] == +np.inf else x_u[i]
            output += dv_template.format(idx=i, name='x'+f'[{i}]', scaler=x_s[i], lower=l, value=x, upper=u)

        # Print constraint data
        if self.constrained:
            output += f'\n\n\tConstraints:\n'
            header = "\t{0} | {1} | {2} | {3} | {4} | {5} | {6} ".format(
                                                            pad_name('Index', 5),
                                                            pad_name('Name', 10),
                                                            pad_name('Scaler', 13),
                                                            pad_name('Lower Limit', 13),
                                                            pad_name('Value', 13),
                                                            pad_name('Upper Limit', 13),
                                                            pad_name('Lag. mult.', 13),
                                                            )
            output += header
            if 1: # print lagrange multipliers
                con_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e} | {lag:<+.6e}"
            else:
                con_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e} | "

            for i, c in enumerate(cons.flatten()):
                l = -1.e99 if c_l[i] == -np.inf else c_l[i]
                u = +1.e99 if c_u[i] == +np.inf else c_u[i]
                if 1: # print lagrange multipliers
                    output += con_template.format(idx=i, name='con'+f'[{i}]', scaler=c_s[i], lower=l, value=c, upper=u, lag=mu[i])
                else:
                    output += con_template.format(idx=i, name='con'+f'[{i}]', scaler=c_s[i], lower=l, value=c, upper=u)

        # # <<<<<<<<<<<<<<<<<<<<<
        # # PROBLEM DATA ENDS
        output += '\n\t' + '-'*100 + '\n'
            
        return output