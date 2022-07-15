import numpy as np

from modopt.api import Problem


class Quadratic(Problem):
    def initialize(self, ):
        # self.options.declare('problem_name', default='x^4', types=str)
        self.problem_name = 'x^2 + y^2'

    def setup(self):
        # Make optional
        self.add_design_variables('x',
                                  shape=(1, ),
                                  lower=np.array([
                                      0.,
                                  ]),
                                  upper=None,
                                  equals=None,
                                  vals=np.array([
                                      acti500.,
                                  ]))

        self.add_design_variables('y',
                                  shape=(1, ),
                                  lower=None,
                                  upper=None,
                                  equals=None,
                                  vals=np.full((1, ), 5.))

        self.set_objective('f', )
        # scaler=1.0)

        self.add_constraints(
            'x+y',
            shape=(1, ),
            lower=None,
            upper=None,
            equals=np.array([
                1.,
            ]),
        )

        self.add_constraints(
            'x-y',
            shape=(1, ),
            lower=np.array([
                1.,
            ]),
            upper=None,
            equals=None,
        )

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x', vals=None)
        self.declare_objective_gradient(wrt='y', vals=None)
        self.declare_constraint_jacobian(of='x+y',
                                         wrt='x',
                                         vals=np.array([
                                             1.,
                                         ]))
        self.declare_constraint_jacobian(of='x+y',
                                         wrt='y',
                                         vals=np.array([
                                             1.,
                                         ]))
        self.declare_constraint_jacobian(of='x-y',
                                         wrt='x',
                                         vals=np.array([
                                             1.,
                                         ]))
        self.declare_constraint_jacobian(of='x-y',
                                         wrt='y',
                                         vals=np.array([
                                             -1.,
                                         ]))
        # self.declare_objective_hessian(of='x',
        #                                wrt='x',
        #                                shape=(2, 2),
        #                                vals=np.array([[2., 0], [0,
        #                                                         2.]]))
        # self.declare_objective_hvp(wrt='x', shape=(25, ), vals=None)

    # def compute_objective(self, x, f):
    #     f = x[0]**2 + x[1]**2

    #     return f

    # def compute_objective(self):
    #     dv = self.x
    #     self.f = dv['x']**2 + dv['y']**2

    def compute_objective(self, dvs, obj):
        obj['f'] = dvs['x']**2 + dvs['y']**2

    # def compute_objective_gradient(self, x):
    #     g = 2 * x
    #     return g

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']
        grad['y'] = 2 * dvs['y']

    # def compute_constraints(self, x):
    #     c = np.array([x[0] + x[1], x[0] - x[1]])
    #     return c

    def compute_constraints(self, dvs, cons):
        cons['x+y'] = dvs['x'] + dvs['y']
        cons['x-y'] = dvs['x'] - dvs['y']

    # def compute_constraint_jacobian(self, x):
    #     j = np.array([[1., 1.], [1., -1.]])
    #     return j

    def compute_constraint_jacobian(self, dvs, jac):
        pass
        # jac['x+y', 'x'] = 1.
        # jac['x+y', 'y'] = 1.
        # jac['x-y', 'x'] = 1.
        # jac['x-y', 'y'] = -1.

    # def compute_objective_hessian(self, x):
    #     nx = x.shape[0]
    #     diag = 12 * np.power(x, 2).reshape(nx, )
    #     hess = np.diag(diag)
    #     return hess

    # def compute_objective_hvp(self, x, p):
    #     nx = x.shape[0]
    #     diag = 12 * np.power(x, 2).reshape(nx, )
    #     hess = np.diag(diag)
    #     hvp = np.matmul(hess, p)

    #     return hvp