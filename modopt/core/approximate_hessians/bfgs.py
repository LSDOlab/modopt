import numpy as np

from modopt import ApproximateHessian


class BFGS(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)
        self.options.declare('store_inverse', default=False, types=bool)

    def update(self, d, w):
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