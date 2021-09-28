import numpy as np

from modopt.api import ApproximateHessian


class PSB(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)

    def update(self, dk, wk):
        if self.options['store_hessian']:
            vec = wk - np.dot(self.B_k, dk)
            const = np.inner(dk, dk)

            self.B_k += (np.outer(vec, dk) + np.outer(dk, vec)) / const
            -(np.inner(vec, dk) / const**2) * np.outer(dk, dk)
