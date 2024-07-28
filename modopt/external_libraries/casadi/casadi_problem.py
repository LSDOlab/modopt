import numpy as np
import warnings
import time
from modopt import ProblemLite
from modopt.utils.options_dictionary import OptionsDictionary

try:
    import casadi as ca
except:
    ca = None
    warnings.warn("'casadi' could not be imported. Install 'casadi' using `pip install casadi` for using CasADi Problems.")


class CasadiProblem(ProblemLite):
    '''
    Class that interfaces with CasADi expressions for objectives and constraints.
    '''
    def __init__(self, x0, name='unnamed_problem', ca_obj=None, ca_con=None, 
                 xl=None, xu=None, cl=None, cu=None, x_scaler=1., o_scaler=1., c_scaler=1., grad_free=False):
        '''
        Initialize the optimization problem with the given design variables, objective, constraints, and their derivatives.
        Derivatives are automatically computed using CasADi.
        Vector products are not supported.

        Attributes
        ----------
        name : str, default='unnamed_problem'
            Problem name assigned by the user.
        x0 : np.ndarray
            Initial guess for design variables.
        ca_obj : callable
            A Python function that returns the objective function expression in CasADi.
            Signature: ca_obj(x: ca.MX) -> ca.MX
        ca_con : callable
            A Python function that returns the constraints function expression in CasADi.
            Signature: ca_con(x: ca.MX) -> ca.MX
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
        o_scaler : float
            Scaling factor for the objective function.
        c_scaler : float or np.ndarray
            Scaling factor for constraints.
        grad_free : bool, default=False
            Flag to indicate if the problem is gradient-free.
            If True, the optimizer will not use the gradient information.
        '''
        nx = x0.size
        if x0.shape != (nx,):
            raise ValueError(f"Initial guess 'x0' must be a numpy 1d-array.")

        if ca is None:
            raise ImportError("'casadi' could not be imported. Install 'casadi' using `pip install casadi` for using CasADi Problems.")
        
        x = ca.MX.sym('x', nx)

        obj = None
        grad = None
        obj_hess = None

        if ca_obj is not None:
            obj_expr = ca_obj(x)
            _obj = ca.Function('o', [x], [obj_expr])
            obj  = lambda x: np.float64(_obj(x))
            if not grad_free:
                # grad_expr = ca.gradient(obj_expr, x)
                obj_hess_expr, grad_expr = ca.hessian(obj_expr, x)
                _grad     = ca.Function('g', [x], [grad_expr])
                grad      = lambda x: np.array(_grad(x)).flatten()

                _obj_hess     = ca.Function('h', [x], [obj_hess_expr])
                obj_hess      = lambda x: np.array(_obj_hess(x))

        con = None
        jac = None
        lag = None
        lag_grad = None
        lag_hess = None
        
        if ca_con is not None:
            con_expr = ca_con(x)
            _con = ca.Function('c', [x], [con_expr])
            nc = con_expr.size1() # number of rows in the constraint expression
            con  = lambda x: np.array(_con(x)).flatten()
            if not grad_free:
                jac_expr = ca.jacobian(con_expr, x)
                _jac     = ca.Function('j', [x], [jac_expr])
                jac      = lambda x: np.array(_jac(x))

        if ca_con is not None:
            lam = ca.MX.sym('lam', nc)
            if obj_expr is not None:
                lag_expr = obj_expr + ca.dot(lam, con_expr)
            else:
                lag_expr = ca.dot(lam, con_expr)
            _lag = ca.Function('lag', [x, lam], [lag_expr])
            lag  = lambda x, lam: np.float64(_lag(x, lam))

            if not grad_free:
                # lag_grad_expr = ca.gradient(lag_expr, x)
                lag_hess_expr, lag_grad_expr = ca.hessian(lag_expr, x)
                _lag_grad     = ca.Function('lag', [x, lam], [lag_grad_expr])
                lag_grad      = lambda x, lam: np.array(_lag_grad(x, lam)).flatten()

                _lag_hess     = ca.Function('lag', [x, lam], [lag_hess_expr])
                lag_hess      = lambda x, lam: np.array(_lag_hess(x, lam))
            
        super().__init__(x0, name=name, obj=obj, grad=grad, obj_hess=obj_hess, con=con, jac=jac,
                         lag=lag, lag_grad=lag_grad, lag_hess=lag_hess, xl=xl, xu=xu, cl=cl, cu=cu,
                         x_scaler=x_scaler, o_scaler=o_scaler, c_scaler=c_scaler, grad_free=grad_free)
