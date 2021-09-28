import numpy as np

from modopt.api import ApproximateHessian


class SR1(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)

    def update(self, dk, wk):
        if self.options['store_hessian']:
            vec = wk - np.dot(self.B_k, dk)
            self.B_k += np.outer(vec, vec) / np.inner(vec, dk)