import numpy as np
from modopt import MeritFunction


# Note: Augmented Lagrangian is a function of  x, lag_mult and slack variables
# Note: This Merit function is for problems with constraints: c_e(x) = 0, c_i(x) >= 0
class AugmentedLagrangian(MeritFunction):
    """
    Augmented Lagrangian merit function with concatenated constraints: [c_e(x) = 0, c_i(x) >= 0].

    Parameters
    ----------
    f : callable
        Objective function.
    c : callable
        Constraint function representing concatenated constraints: [c_e(x) = 0, c_i(x) >= 0].
    g : callable
        Objective gradient.
    j : callable
        Constraint Jacobian.
    nx : int
        Number of optimization/design variables.
    nc : int
        Number of inequality constraints.
    nc_e : int
        Number of equality constraints.
    non_bound_indices : np.ndarray
        Indices of the non-bound constraints.
        This is used to ignore the bound constraints in the merit function evaluation.
        Default is None, which means all constraints are considered non-bound constraints.
    
    Attributes
    ----------
    rho : np.ndarray
        Vector of penalty parameters corresponding to the constraints.
    cache : dict
        Dictionary to store the latest function evaluations for 'f', 'c', 'g' and 'j'.
        Keys are the function names 'f', 'c', 'g' and 'j', 
        and values are tuples of the form (x, f(x)).
    eval_count : dict
        Dictionary to store the number of times each function has been evaluated.
        Keys are the function names 'f', 'c', 'g' and 'j',
        and values are the number of evaluations.
    """
    def initialize(self):
        self.options.declare('non_bound_indices', default=None, types=(np.ndarray, type(None)))

    def setup(self):
        nc = self.options['nc']
        self.rho = np.zeros(nc)
        if self.options['non_bound_indices'] is None:
            self.options['non_bound_indices'] = np.arange(nc)

    def set_rho(self, rho):
        self.rho[:] = rho

    def compute_function(self, v):
        nx   = self.options['nx']
        nc   = self.options['nc']
        nc_e = self.options['nc_e']
        rho  = self.rho

        x = v[:nx]
        lag_mult = v[nx:(nx + nc)]
        slacks = np.concatenate((np.zeros((nc_e,)), v[(nx + nc):]))

        self.update_functions_in_cache(['f', 'c'], x)
        obj = self.cache['f'][1]
        con = self.cache['c'][1]
        # obj = self.options['f'](x)
        # con = self.options['c'](x)

        return obj - np.dot(lag_mult, (con - slacks)) + 0.5 * np.inner(rho, (con - slacks)**2)

        # nbi  = self.options['non_bound_indices']
        # return obj - np.dot(lag_mult[nbi], (con - slacks)[nbi]) + 0.5 * np.inner(rho[nbi], (con - slacks)[nbi]**2)


    def evaluate_function(self, x, lag_mult, s, f, c):
        nc_e = self.options['nc_e']
        rho  = self.rho
        sl   = np.concatenate((np.zeros((nc_e,)), s))

        return f - np.dot(lag_mult, (c - sl)) + 0.5 * np.inner(rho, (c - sl)**2)
    
        # nbi  = self.options['non_bound_indices']
        # return f - np.dot(lag_mult[nbi], (c - sl)[nbi]) + 0.5 * np.inner(rho[nbi], (c - sl)[nbi]**2)

# Note: Gradient is evaluated with respect to x, lag_mult and slack variables
    def compute_gradient(self, v):
        nx = self.options['nx']
        nc = self.options['nc']
        nc_e = self.options['nc_e']
        rho = self.rho

        x = v[:nx]
        lag_mult = v[nx:(nx + nc)]
        slacks = np.concatenate((np.zeros((nc_e,)), v[(nx + nc):]))

        self.update_functions_in_cache(['c', 'g', 'j'], x)
        con  = self.cache['c'][1]
        grad = self.cache['g'][1]
        jac  = self.cache['j'][1]
        # con  = self.options['c'](x)
        # grad = self.options['g'](x)
        # jac  = self.options['j'](x)

        # grad_x = grad - np.matmul(jac.T, lag_mult - (rho * (con - slacks)))
        grad_x = grad - jac.T @ (lag_mult - (rho* (con - slacks)))
        grad_lag_mult = -(con - slacks)
        grad_slacks = lag_mult[nc_e:] - rho[nc_e:] * (con[nc_e:] - slacks[nc_e:])

        # nbi  = self.options['non_bound_indices']
        # grad_x = grad - jac[nbi].T @ (lag_mult[nbi] - (rho[nbi]* (con[nbi] - slacks[nbi])))
        # grad_lag_mult = -(con - slacks)
        # grad_slacks = lag_mult[nc_e:] - rho[nc_e:] * (con[nc_e:] - slacks[nc_e:])
        # grad_lag_mult[nbi] = 0.
        # grad_slacks[nbi[nc_e:]-nc_e] = 0.

        # print('compute_gradient', np.concatenate((grad_x, grad_lag_mult, grad_slacks)))

        return np.concatenate((grad_x, grad_lag_mult, grad_slacks))

    def evaluate_gradient(self, x, lag_mult, s, f, c, g, j):
        nc_e = self.options['nc_e']
        rho  = self.rho
        sl   = np.concatenate((np.zeros((nc_e,)), s))

        # grad_x = g - np.matmul(j.T, lag_mult - (rho * (c - s)))
        grad_x = g - j.T @ (lag_mult - (rho * (c - sl)))
        grad_lag_mult = -(c - sl)
        grad_slacks = lag_mult[nc_e:] - rho[nc_e:] * (c[nc_e:] - s)

        # nbi  = self.options['non_bound_indices']
        # grad_x = g - j[nbi].T @ (lag_mult[nbi] - (rho[nbi] * (c[nbi] - sl[nbi])))
        # grad_lag_mult = -(c - sl)
        # grad_slacks = lag_mult[nc_e:] - rho[nc_e:] * (c[nc_e:] - s)
        # grad_lag_mult[nbi] = 0.
        # grad_slacks[nbi[nc_e:]-nc_e] = 0.

        # print('evaluate_gradient', np.concatenate((grad_x, grad_lag_mult, grad_slacks)))

        return np.concatenate((grad_x, grad_lag_mult, grad_slacks))