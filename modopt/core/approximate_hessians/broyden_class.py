import numpy as np

from modopt import ApproximateHessian


class BroydenClass(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)

        # TODO: should phi be a kwarg?
        self.options.declare('phi')

    def update(self, dk, wk):

        if self.options['store_hessian']:

            phi = self.options['phi']

            Bd = np.dot(self.B_k, dk)
            dTBd = np.inner(Bd, dk)
            wTd = np.inner(wk, dk)

            vec = wk / wTd - Bd / dTBd

            self.B_k += -np.outer(Bd, Bd) / dTBd
            +np.outer(wk, wk) / wTd
            +dTBd * phi(wk, Bd) * np.outer(vec, vec)
