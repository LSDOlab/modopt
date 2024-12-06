import numpy as np

from modopt import ApproximateHessian

from scipy.optimize import BFGS


class BFGSScipy(ApproximateHessian):
    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)
        self.options.declare('store_inverse', default=False, types=bool)
        self.options.declare('exception_strategy',
                             default='damp_update',
                             values=['skip_update', 'damp_update'])
        self.options.declare('min_curvature', default=0.0, types=float)
        self.options.declare('init_scale',
                             default='auto',
                             types=(float, str))

    def setup(self, ):
        if self.options['min_curvature'] == 0.0:
            if self.options['exception_strategy'] == 'skip_update':
                self.options['min_curvature'] = 1e-8
            elif self.options['exception_strategy'] == 'damp_update':
                self.options['min_curvature'] = 0.2
            else:
                raise ValueError(
                    'Invalid value encountered for "exception_strategy"'
                )

        self.B = BFGS(
            exception_strategy=self.options['exception_strategy'],
            min_curvature=self.options['min_curvature'],
            init_scale=self.options['init_scale'])

        if self.options['store_hessian']:
            approx_type = 'hess'
        else:
            approx_type = 'inv_hess'

        self.B.initialize(self.options['nx'], approx_type)

    def update(self, dk, wk):
        self.B.update(dk, wk)

        if self.options['store_hessian']:
            self.B_k = self.B.get_matrix()

        if self.options['store_inverse']:
            self.M_k = self.B.get_matrix()
