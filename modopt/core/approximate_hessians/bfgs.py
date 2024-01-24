import numpy as np

from modopt import ApproximateHessian


class BFGS(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)
        self.options.declare('store_inverse', default=False, types=bool)

    def update(self, dk, wk):
        if self.options['store_hessian']:
            Bd = np.dot(self.B_k, dk)

            self.B_k += -np.outer(Bd, Bd) / np.inner(Bd, dk) + np.outer(
                wk, wk) / np.inner(wk, dk)

        if self.options['store_inverse']:
            Mw = np.dot(self.M_k, wk)
            wTMw = np.inner(Mw, wk)
            dTw = np.inner(wk, dk)

            vec = dk / dTw - Mw / wTMw

            self.M_k += -np.outer(Mw, Mw) / wTMw + np.outer(
                dk, dk) / dTw + wTMw * np.outer(vec, vec)
