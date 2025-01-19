import numpy as np

from modopt import ApproximateHessian


class PSB(ApproximateHessian):
    """
    Powell-Symmetric-Broyden (PSB) Hessian update.

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
        Update the stored Hessian approximation `B_k` using the PSB formula.
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
            dTd = np.dot(d, d)

            self.B_k += (np.outer(vec, d) + np.outer(d, vec)) / dTd - (np.dot(vec, d) / dTd**2) * np.outer(d, d)