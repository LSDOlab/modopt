import numpy as np
from modopt import ProblemLite

class JaxProblem(ProblemLite):
    '''
    Class that wraps **jittable** Jax functions for objective and constraints.
    Depending on the ``order`` specified, this class will automatically generate the functions for the objective gradient,
    constraint Jacobian, objective Hessian, Lagrangian, Lagrangian gradient, and Lagrangian Hessian.
    All functions will be turned into jitted functions and then wrapped for use with ``Optimizer`` subclasses.
    Vector products (HVP, JVP, VJP) are not supported.
    '''
    def __init__(self, x0, nc=None, name='unnamed_problem', jax_obj=None, jax_con=None, xl=None, xu=None, 
                 cl=None, cu=None, x_scaler=1., o_scaler=1., c_scaler=1., grad_free=False, order=1):
        '''
        Initialize the optimization problem with the given design variables, objective, and constraints.
        Derivatives are automatically generated using Jax.

        Parameters
        ----------
        name : str, default='unnamed_problem'
            Problem name assigned by the user.
        x0 : np.ndarray
            Initial guess for design variables.
        nc : int
            Number of constraints.
            Used for determining jacfwd or jacrev.
            If None, jacrev is used.
        jax_obj : callable
            Objective function written in Jax. (must be jittable)
            Signature: jax_obj(x: jnp.array) -> jnp.float
        jax_con : callable
            Constraint function written in Jax. (must be jittable)
            Signature: jax_con(x: jnp.array) -> jnp.array
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
            If True, JaxProblem will not generate any derivatives.
        order : {1, 2}, default=1
            Order of the problem if ``grad_free=False``.
            Used for determining up to which order of derivatives need to be generated.
        '''
        nx = x0.size
        if x0.shape != (nx,):
            raise ValueError(f"Initial guess 'x0' must be a numpy 1d-array.")
    
        try:
            import jax
        except ImportError: 
            raise ImportError("'jax' could not be imported. Install 'jax' using `pip install jax[cpu]` for using JaxProblem.")

        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)

        # Determine if jacrev or jacfwd should be used
        jacrev = True
        if nc is not None:
            if not isinstance(nc, int) or nc < 0:
                raise ValueError("'nc' must be an integer greater than or equal to 0.")
            if nc >= nx:
                jacrev = False
  
        obj = None
        grad = None
        obj_hess = None
        # obj_hvp = None

        if jax_obj is not None:
            _obj = jax.jit(jax_obj)
            obj  = lambda x: np.float64(_obj(x))
            if not grad_free:
                if order == 1:
                    _grad = jax.jit(jax.grad(jax_obj))
                    grad  = lambda x: np.array(_grad(x))
                elif order == 2:
                    _grad     = jax.jit(jax.grad(jax_obj))
                    grad      = lambda x: np.array(_grad(x))
                    _obj_hess = jax.jit(jax.hessian(jax_obj))
                    obj_hess  = lambda x: np.array(_obj_hess(x))

                    # def jax_obj_hvp(x, v):
                    #     return jax.grad(lambda x: jnp.vdot(jax.grad(jax_obj)(x), v))(x)
                    
                    # _obj_hvp = jax.jit(jax_obj_hvp)
                    # obj_hvp  = lambda x, v: np.array(_obj_hvp(x, v))

                    # # obj_hvp = jax.grad(lambda x: jnp.vdot(jax.grad(jax_obj)(x), v))(x)
                else:
                    raise ValueError(f"Higher order derivatives are not supported. 'order' must be 1 or 2.")

        con = None
        jac = None
        lag = None
        lag_grad = None
        lag_hess = None
        # lag_hvp  = None
        
        if jax_con is not None:
            _con = jax.jit(jax_con)
            con  = lambda x: np.array(_con(x))
            if not grad_free:
                if jacrev:
                    _jac = jax.jit(jax.jacrev(jax_con)) # Note: jax.jacobian = jax.jacrev
                else:
                    _jac = jax.jit(jax.jacfwd(jax_con))
                jac  = lambda x: np.array(_jac(x))

            # Lagrangian functions
            if jax_obj is not None:
                jax_lag = lambda x, lam: jax_obj(x) + jnp.dot(jax_con(x), lam)
            else:
                jax_lag = lambda x, lam: jax.dot(jax_con(x), lam)

            _lag = jax.jit(jax_lag)
            lag  = lambda x, lam: np.float64(_lag(x, lam))

            if not grad_free:
                if order==1:
                    _lag_grad = jax.jit(jax.grad(jax_lag))
                    lag_grad  = lambda x, lam: np.array(_lag_grad(x, lam))
                elif order==2:
                    _lag_grad = jax.jit(jax.grad(jax_lag))
                    lag_grad  = lambda x, lam: np.array(_lag_grad(x, lam))
                    _lag_hess = jax.jit(jax.hessian(jax_lag))
                    lag_hess  = lambda x, lam: np.array(_lag_hess(x, lam))

                    # def jax_lag_hvp(x, lam, v):
                    #     return jax.grad(lambda x, lam: jnp.vdot(jax.grad(jax_lag)(x, lam), v))(x, lam)
                    
                    # _lag_hvp = jax.jit(jax_lag_hvp)
                    # lag_hvp  = lambda x, lam, v: np.array(_lag_hvp(x, lam, v))
                else:
                    raise ValueError(f"Higher order derivatives are not supported. 'order' must be 1 or 2.")

        super().__init__(x0, name=name, obj=obj, grad=grad, obj_hess=obj_hess, con=con, jac=jac,
                         lag=lag, lag_grad=lag_grad, lag_hess=lag_hess, xl=xl, xu=xu, cl=cl, cu=cu,
                         x_scaler=x_scaler, o_scaler=o_scaler, c_scaler=c_scaler, grad_free=grad_free)
                        #  obj_hvp=obj_hvp, lag_hvp=lag_hvp)