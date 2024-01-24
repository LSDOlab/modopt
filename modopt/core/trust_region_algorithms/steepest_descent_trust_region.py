import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt import TrustRegion


class SteepestDescentTrustRegion(TrustRegion):
    def initialize(self):

        # Maximum number of iterations allowed before convergence
        self.options.declare('max_itr', default=20, types=int, lower=3)

    def update(self, x, p, f0=None, g0=None):
        pass

    def subproblem_solve(self):
        delta = self.options['delta']

        # Compute unit vector along steepest descent direction in the first iteration
        if self.itr == 0:
            self.norm_g0 = np.linalg.norm(self.g0)
            self.unit_g0 = self.g0 / self.norm_g0

        return delta * self.unit_g0
