import numpy as np

from modopt.api import ApproximateHessian


class BroydenFirst(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_inverse', default=True, types=bool)

    def update(self, dk, wk):
        if self.options['store_inverse']:
            self.M_k += np.outer(dk - np.dot(self.M_k, wk),
                                 wk) / np.inner(wk, wk)
