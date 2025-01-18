import numpy as np

from modopt import ApproximateHessian

from scipy.optimize import BFGS


class BFGSScipy(ApproximateHessian):
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update using SciPy's BFGS implementation.

    Parameters
    ----------
    nx : int
        Number of optimization variables.
    store_hessian : bool, default=True
        Store the Hessian approximation.
    store_inverse : bool, default=False
        Store the inverse Hessian approximation.
    min_curvature : float, default=0.0
        Curvature below which ``exception_strategy`` is triggered.
        Default is `1e-8` when ``exception_strategy = 'skip_update'`` 
        and `0.2` when ``exception_strategy = 'damp_update'``.
    exception_strategy : {'skip_update', 'damp_update'}, default='damp_update'
        Strategy to proceed when the ``min_curvature`` condition is violated.

        - `'skip_update'`: Skip the update and keep the previous approximation.
        - `'damp_update'`: Interpolate between the computed BFGS update and the previous approximation.
    init_scale : {float, 'auto'}, default='auto'
        Initial scaling factor for the Hessian approximation.

        - `float`: Use ``init_scale*np.eye(nx)`` as the initial approximation.
        - `'auto'`: Use an automatic heuristic to compute the initial scaling factor.

    Attributes
    ----------
    B_k : np.ndarray
        Hessian approximation of shape `(nx, nx)`.
        Available only if ``store_hessian`` is ``True``.
    M_k : np.ndarray
        Inverse Hessian approximation of shape `(nx, nx)`.
        Available only if ``store_inverse`` is ``True``.
    """
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)
        self.options.declare('store_inverse', default=False, types=bool)
        self.options.declare('exception_strategy',
                             default='damp_update',
                             values=['skip_update', 'damp_update'])
        self.options.declare('min_curvature', default=0.0, types=float)
        self.options.declare('init_scale',
                             default='auto',
                             types=(float, str))

    def setup(self, ):
        if self.options['min_curvature'] == 0.0:
            if self.options['exception_strategy'] == 'skip_update':
                self.options['min_curvature'] = 1e-8
            else:
                self.options['min_curvature'] = 0.2

        if self.options['store_hessian']:
            self.B = BFGS(
                exception_strategy=self.options['exception_strategy'],
                min_curvature=self.options['min_curvature'],
                init_scale=self.options['init_scale'])
            approx_type = 'hess'
            self.B.initialize(self.options['nx'], approx_type)
        if self.options['store_inverse']:
            self.M = BFGS(
                exception_strategy=self.options['exception_strategy'],
                min_curvature=self.options['min_curvature'],
                init_scale=self.options['init_scale'])
            approx_type = 'inv_hess'
            self.M.initialize(self.options['nx'], approx_type)

    def update(self, d, w):
        """
        Update the stored Hessian approximation `B_k` or its inverse `M_k` using the BFGS formula.
        The update is performed using the step ``d`` and the gradient difference ``w``.
        
        Parameters
        ----------
        d : np.ndarray
            Step taken in the optimization space.
        w : np.ndarray
            Gradient difference along the step ``d``.
        """

        if self.options['store_hessian']:
            self.B.update(d, w)
            self.B_k = self.B.get_matrix()

        if self.options['store_inverse']:
            self.M.update(d, w)
            self.M_k = self.M.get_matrix()
