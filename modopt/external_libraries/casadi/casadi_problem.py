import numpy as np
from modopt import ProblemLite

class CasadiProblem(ProblemLite):
    '''
    Class that wraps CasADi **expressions** for objective and constraints.
    Depending on the ``order`` specified, this class will automatically generate the expressions for the objective gradient, 
    constraint Jacobian, objective Hessian, Lagrangian, Lagrangian gradient, and Lagrangian Hessian.
    All expressions will be turned into functions and then wrapped for use with ``Optimizer`` subclasses.
    Vector products (HVP, JVP, VJP) are not supported.
    '''
    def __init__(self, x0, name='unnamed_problem', ca_obj=None, ca_con=None, xl=None, xu=None, 
                 cl=None, cu=None, x_scaler=1., o_scaler=1., c_scaler=1., grad_free=False, order=1):
        '''
        Initialize the optimization problem with the given design variables, objective, and constraints.
        Derivatives are automatically generated using CasADi.

        Parameters
        ----------
        name : str, default='unnamed_problem'
            Problem name assigned by the user.
        x0 : np.ndarray
            Initial guess for design variables.
        ca_obj : callable
            A Python function that returns the objective function expression in CasADi.
            Signature: ca_obj(x: ca.MX) -> ca.MX
        ca_con : callable
            A Python function that returns the constraint function expression in CasADi.
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
            If True, CasadiProblem will not generate any derivatives.
        order : {1, 2}, default=1
            Order of the problem if ``grad_free=False``.
            Used for determining up to which order of derivatives need to be generated.
        '''
        nx = x0.size
        if x0.shape != (nx,):
            raise ValueError(f"Initial guess 'x0' must be a numpy 1d-array.")
        
        try:
            import casadi as ca
        except ImportError:
            raise ImportError("'casadi' could not be imported. Install 'casadi' using `pip install casadi` for using CasadiProblem.")

        x = ca.MX.sym('x', nx)

        obj = None
        grad = None
        obj_hess = None

        if ca_obj is not None:
            obj_expr = ca_obj(x)                        # expression
            _obj = ca.Function('o', [x], [obj_expr])    # function
            obj  = lambda x: np.float64(_obj(x))        # wrapped function
            if not grad_free:
                if order == 1:
                    grad_expr = ca.gradient(obj_expr, x)
                    _grad     = ca.Function('g', [x], [grad_expr])
                    grad      = lambda x: np.array(_grad(x)).flatten()
                elif order == 2:
                    obj_hess_expr, grad_expr = ca.hessian(obj_expr, x)        
                    _grad     = ca.Function('g', [x], [grad_expr])
                    grad      = lambda x: np.array(_grad(x)).flatten()
                    _obj_hess = ca.Function('h', [x], [obj_hess_expr])
                    obj_hess  = lambda x: np.array(_obj_hess(x))
                else:
                    raise ValueError(f"Higher order derivatives are not supported. 'order' must be 1 or 2.")

        con = None
        jac = None
        lag = None
        lag_grad = None
        lag_hess = None
        
        if ca_con is not None:
            con_expr = ca_con(x)
            _con = ca.Function('c', [x], [con_expr])
            con  = lambda x: np.array(_con(x)).flatten()
            if not grad_free:
                jac_expr = ca.jacobian(con_expr, x)
                _jac     = ca.Function('j', [x], [jac_expr])
                jac      = lambda x: np.array(_jac(x))

            # Lagrangian functions
            nc = con_expr.size1() # number of rows in the constraint expression
            lam = ca.MX.sym('lam', nc)
            if ca_obj is not None:
                lag_expr = obj_expr + ca.dot(lam, con_expr)
            else:
                lag_expr = ca.dot(lam, con_expr)
            _lag = ca.Function('lag', [x, lam], [lag_expr])
            lag  = lambda x, lam: np.float64(_lag(x, lam))

            if not grad_free:
                if order == 1:
                    lag_grad_expr = ca.gradient(lag_expr, x)
                    _lag_grad     = ca.Function('lg', [x, lam], [lag_grad_expr])
                    lag_grad      = lambda x, lam: np.array(_lag_grad(x, lam)).flatten()
                elif order == 2:
                    lag_hess_expr, lag_grad_expr = ca.hessian(lag_expr, x)
                    _lag_grad = ca.Function('lg', [x, lam], [lag_grad_expr])
                    lag_grad  = lambda x, lam: np.array(_lag_grad(x, lam)).flatten()
                    _lag_hess = ca.Function('lh', [x, lam], [lag_hess_expr])
                    lag_hess  = lambda x, lam: np.array(_lag_hess(x, lam))
                else:
                    raise ValueError(f"Higher order derivatives are not supported. 'order' must be 1 or 2.")
            
        super().__init__(x0, name=name, obj=obj, grad=grad, obj_hess=obj_hess, con=con, jac=jac,
                         lag=lag, lag_grad=lag_grad, lag_hess=lag_hess, xl=xl, xu=xu, cl=cl, cu=cu,
                         x_scaler=x_scaler, o_scaler=o_scaler, c_scaler=c_scaler, grad_free=grad_free)