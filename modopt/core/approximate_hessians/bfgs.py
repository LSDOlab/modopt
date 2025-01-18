import numpy as np

from modopt import ApproximateHessian


class BFGS(ApproximateHessian):
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update.

    Parameters
    ----------
    nx : int
        Number of optimization variables.
    store_hessian : bool, default=True
        Store the Hessian approximation.
    store_inverse : bool, default=False
        Store the inverse Hessian approximation.

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
            Bd  = np.dot(self.B_k, d)
            dBd = np.dot(d, Bd)
            wTd = np.dot(w, d)

            self.B_k += -np.outer(Bd, Bd) / dBd + np.outer(w, w) / wTd

        if self.options['store_inverse']:
            Mw  = np.dot(self.M_k, w)
            wMw = np.dot(w, Mw)
            wTd = np.dot(w, d)

            vec = d / wTd - Mw / wMw

            self.M_k += -np.outer(Mw, Mw) / wMw + np.outer(d, d) / wTd + wMw * np.outer(vec, vec)