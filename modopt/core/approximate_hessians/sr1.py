import numpy as np

from modopt import ApproximateHessian


class SR1(ApproximateHessian):
    """
    Symmetric rank-one (SR1) Hessian update.

    Parameters
    ----------
    nx : int
        Number of optimization variables.

    Attributes
    ----------
    B_k : np.ndarray
        Hessian approximation of shape `(nx, nx)`.
    """
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)

    def update(self, d, w):
        """
        Update the stored Hessian approximation `B_k` using the SR1 formula.
        The update is performed using the step ``d`` and the gradient difference ``w``.

        Parameters
        ----------
        d : np.ndarray
            Step taken in the optimization space.
        w : np.ndarray
            Gradient difference along the step ``d``.
        """
        if self.options['store_hessian']:
            vec = w - np.dot(self.B_k, d)
            self.B_k += np.outer(vec, vec) / np.inner(vec, d)