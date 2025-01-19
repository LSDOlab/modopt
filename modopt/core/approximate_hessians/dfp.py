import numpy as np

from modopt import ApproximateHessian


class DFP(ApproximateHessian):
    """
    Davidon-Fletcher-Powell (DFP) Hessian update.

    Parameters
    ----------
    nx : int
        Number of optimization variables.
    store_hessian : bool, default=False
        Store the Hessian approximation.
    store_inverse : bool, default=True
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
        self.options.declare('store_hessian', default=False, types=bool)
        self.options.declare('store_inverse', default=True, types=bool)

    def update(self, d, w):
        """
        Update the stored Hessian approximation `B_k` or its inverse `M_k` using the DFP formula.
        The update is performed using the step ``d`` and the gradient difference ``w``.

        Parameters
        ----------
        d : np.ndarray
            Step taken in the optimization space.
        w : np.ndarray
            Gradient difference along the step ``d``.
        """
        if self.options['store_hessian']:
            Bd = np.dot(self.B_k, d)
            dTBd = np.inner(Bd, d)
            wTd = np.inner(w, d)

            vec = w / wTd - Bd / dTBd

            self.B_k += -np.outer(Bd, Bd) / dTBd + np.outer(w, w) / wTd + dTBd * np.outer(vec, vec)

        if self.options['store_inverse']:
            Mw  = np.dot(self.M_k, w)
            wMw = np.dot(w, Mw)
            wTd = np.dot(w, d)

            self.M_k += -np.outer(Mw, Mw) / wMw + np.outer(d, d) / wTd
