import numpy as np

from modopt import ApproximateHessian


class Broyden(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)
        self.options.declare('store_inverse', default=False, types=bool)

    def update(self, dk, wk):
        if self.options['store_hessian']:
            self.B_k += np.outer(wk - np.dot(self.B_k, dk),
                                 dk) / np.inner(dk, dk)

        if self.options['store_inverse']:
            self.M_k += np.outer(dk - np.dot(self.M_k, wk),
                                 np.matmul(dk.T, self.M_k)) / np.inner(
                                     dk, np.dot(self.M_k, wk))
