import numpy as np

from modopt import ApproximateHessian


class BFGSM1(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)
        self.options.declare('store_inverse', default=False, types=bool)

    def update(self, dk, wk):
        if self.options['store_hessian']:
            tol1 = 1e-14

            Bd = self.B_k.dot(dk)
            wTd = np.dot(wk, dk)

            sign = 1. if wTd >= 0. else -1.
            if abs(wTd) > tol1:
                self.B_k += np.outer(wk, wk) / wTd
            else:
                self.B_k += np.outer(wk, wk) / sign / tol1

            dTBd = np.dot(dk, Bd)
            sign = 1. if dTBd >= 0. else -1.
            if abs(dTBd) > tol1:
                self.B_k -= np.outer(Bd, Bd) / dTBd
            else:
                self.B_k -= np.outer(Bd, Bd) / sign / tol1

        if self.options['store_inverse']:
            Mw = np.dot(self.M_k, wk)
            wTMw = np.inner(Mw, wk)
            dTw = np.inner(wk, dk)

            vec = dk / dTw - Mw / wTMw

            self.M_k += -np.outer(Mw, Mw) / wTMw + np.outer(
                dk, dk) / dTw + wTMw * np.outer(vec, vec)
