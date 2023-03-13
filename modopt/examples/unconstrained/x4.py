import numpy as np

from modopt.api import Problem


class X4(Problem):
    def initialize(self, ):
        self.problem_name = 'x^4'

    def setup(self):
        self.add_design_variables(
            'x',
            shape=(2, ),
            lower=None,
            upper=None,
            equals=None,
            # )
            vals=np.full((2), 0.1))

        self.add_objective('f')

    # def setup_derivatives(self):
    #     self.declare_objective_gradient(wrt='x',
    #                                     shape=(25, ),
    #                                     vals=None)
    #     self.declare_objective_hessian(of='x',
    #                                    wrt='x',
    #                                    shape=(25, 25),
    #                                    vals=None)
    # self.declare_objective_hvp(wrt='x', shape=(25, ), vals=None)

    def compute_objective(self, dvs, obj):
        x4 = np.power(dvs['x'], 4)
        obj['f'] = np.sum(x4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * np.power(dvs['x'], 3)

    def compute_objective_hessian(self, dvs, hess):
        nx = dvs['x'].shape[0]
        diag = 12 * np.power(dvs['x'], 2).reshape(nx, )
        hess['x', 'x'] = np.diag(diag)

    # def compute_objective_hvp(self, x, p):
    #     nx = x.shape[0]
    #     diag = 12 * np.power(x, 2).reshape(nx, )
    #     hess = np.diag(diag)
    #     hvp = np.matmul(hess, p)

    #     return hvp