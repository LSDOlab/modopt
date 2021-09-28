import numpy as np

from modopt.api import ApproximateHessian


class DFP(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=False, types=bool)
        self.options.declare('store_inverse', default=True, types=bool)

    def update(self, dk, wk):
        if self.options['store_hessian']:
            Bd = np.dot(self.B_k, dk)
            dTBd = np.inner(Bd, dk)
            wTd = np.inner(wk, dk)

            vec = wk / wTd - Bd / dTBd

            self.B_k += -np.outer(Bd, Bd) / dTBd
            +np.outer(wk, wk) / wTd
            +dTBd * np.outer(vec, vec)

        if self.options['store_inverse']:
            Mw = np.dot(self.M_k, wk)

            self.M_k += -np.outer(Mw, Mw) / np.inner(Mw, wk)
            +np.outer(dk, dk) / np.inner(wk, dk)
