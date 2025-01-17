import numpy as np
from modopt import MeritFunction


# Note: Augmented Lagrangian is a function of  x, lag_mult and slack variables
# Note: This Merit function is for problems in all-inequality form, i.e., c(x) >= 0
class AugmentedLagrangianIneq(MeritFunction):
    """
    Augmented Lagrangian merit function for pure inequality-constraints of the form c(x) >= 0.

    Parameters
    ----------
    f : callable
        Objective function.
    c : callable
        Constraint function representing the inequality constraints c(x) >= 0.
    g : callable
        Objective gradient.
    j : callable
        Constraint Jacobian.
    nx : int
        Number of optimization/design variables.
    nc : int
        Number of inequality constraints.
    
    Attributes
    ----------
    rho : np.ndarray
        Vector of penalty parameters corresponding to each inequality constraint.
    cache : dict
        Dictionary to store the latest function evaluations for 'f', 'c', 'g' and 'j'.
        Keys are the function names 'f', 'c', 'g' and 'j', 
        and values are tuples of the form (x, f(x)).
    eval_count : dict
        Dictionary to store the number of times each function has been evaluated.
        Keys are the function names 'f', 'c', 'g' and 'j',
        and values are the number of evaluations.
    """
    def setup(self):
        """
        Set up the merit function by initializing `rho` as a vector of zeros.
        """
        nc = self.options['nc']
        self.rho = np.zeros(nc)

    def set_rho(self, rho):
        """
        Set the penalty parameters `rho` for the inequality constraints.

        Parameters
        ----------
        rho : np.ndarray
            Penalty parameter vector for the inequality constraints.
            Must be a 1D numpy array of length `nc`.
        """
        self.rho[:] = rho

    def compute_function(self, v):
        """
        Compute the value of the augmented Lagrangian function at the point `v`.
        
        Parameters
        ----------
        v : np.ndarray
            Point at which to evaluate the augmented Lagrangian function.
            The point `v` is a concatenation of the design variables `x`, 
            Lagrange multipliers `lag_mult` and slack variables `slacks`.
            The point `v` has the form [x, lag_mult, slacks].
            
        Returns
        -------
        float
            Value of the augmented Lagrangian function at the point `v`.
        """
        nx = self.options['nx']
        nc = self.options['nc']
        rho = self.rho
        
        x = v[:nx]
        lag_mult = v[nx:(nx + nc)]
        slacks = v[(nx + nc):]
        
        self.update_functions_in_cache(['f', 'c'], x)
        obj = self.cache['f'][1]
        con = self.cache['c'][1]
        # obj = self.options['f'](x)
        # con = self.options['c'](x)

        return obj - np.dot(lag_mult, (con - slacks)) + 0.5 * np.inner(rho, (con - slacks)**2)

    def evaluate_function(self, x, lag_mult, s, f, c):
        """
        Evaluate the augmented Lagrangian function at the point [`x`, `lag_mult`, `s`],
        given the objective function value `f` and constraint function values `c`
        at the point `x`.

        Parameters
        ----------
        x : np.ndarray
            Design variables.
        lag_mult : np.ndarray
            Lagrange multipliers.
        s : np.ndarray
            Slack variables.
        f : float
            Objective function value at the point `x`.
        c : np.ndarray
            Constraint function values at the point `x`.

        Returns
        -------
        float
            Value of the augmented Lagrangian function at the point `x`.
        """
        rho = self.rho

        return f - np.dot(lag_mult,(c - s)) + 0.5 * np.inner(rho, (c - s)**2)

    # Note: Gradient is evaluated with respect to x, lag_mult and slack variables
    def compute_gradient(self, v):
        """
        Compute the gradient of the augmented Lagrangian function at the point `v`.

        Parameters
        ----------
        v : np.ndarray
            Point at which to evaluate the gradient of the augmented Lagrangian function.
            The point `v` is a concatenation of the design variables `x`, 
            Lagrange multipliers `lag_mult` and slack variables `slacks`.
            The point `v` has the form [x, lag_mult, slacks].

        Returns
        -------
        np.ndarray
            Gradient of the augmented Lagrangian function at the point `v`.
        """
        nx = self.options['nx']
        nc = self.options['nc']
        rho = self.rho
        x = v[:nx]
        lag_mult = v[nx:(nx + nc)]
        slacks = v[(nx + nc):]

        self.update_functions_in_cache(['c', 'g', 'j'], x)
        con  = self.cache['c'][1]
        grad = self.cache['g'][1]
        jac  = self.cache['j'][1]
        # con  = self.options['c'](x)
        # grad = self.options['g'](x)
        # jac  = self.options['j'](x)

        # grad_x = grad - np.matmul(jac.T, lag_mult - (rho * (con - slacks)))
        grad_x = grad - jac.T @ (lag_mult - (rho * (con - slacks)))
        grad_lag_mult = -(con - slacks)
        grad_slacks = lag_mult - rho * (con - slacks)

        return np.concatenate((grad_x, grad_lag_mult, grad_slacks))

    def evaluate_gradient(self, x, lag_mult, s, f, c, g, j):
        """
        Evaluate the gradient of the augmented Lagrangian function at the point 
        [`x`, `lag_mult`, `s`], given the objective function `f`, 
        constraint function `c`, objective gradient `g`, and constraint Jacobian `j`
        at the point `x`.

        Parameters
        ----------
        x : np.ndarray
            Design variables.
        lag_mult : np.ndarray
            Lagrange multipliers.
        s : np.ndarray
            Slack variables.
        f : float
            Objective function value at the point `x`.
        c : np.ndarray
            Constraint function values at the point `x`.
        g : np.ndarray
            Objective gradient at the point `x`.
        j : np.ndarray
            Constraint Jacobian at the point `x`.

        Returns
        -------
        np.ndarray
            Gradient of the augmented Lagrangian function at the point `x`.
        """
        rho = self.rho

        # grad_x = g - np.matmul(j.T, lag_mult - (rho * (c - s)))
        grad_x = g - j.T @ (lag_mult - (rho * (c - s)))
        grad_lag_mult = -(c - s)
        grad_slacks = lag_mult - rho * (c - s)

        return np.concatenate((grad_x, grad_lag_mult, grad_slacks))