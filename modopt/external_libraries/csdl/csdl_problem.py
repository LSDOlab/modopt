import numpy as np

from modopt.api import Problem as OptProblem
# from csdl_lite import Simulator
# from csdl_om import Simulator
from python_csdl_backend import Simulator
from csdl import Model


class CSDLProblem(OptProblem):
    def initialize(self, ):
        self.options.declare('problem_name',
                             default='unnamed_problem',
                             types=str)
        self.options.declare('simulator')
        # TODO:
        # self.options.declare('simulator', types=type(Simulator()))

    def setup(self, ):

        # Only for csdl problems
        # (Since it is not constructed explicitly using the Problem() class)
        self.problem_name = self.options['problem_name']

        # Set problem dimensions
        sim = self.options['simulator']
        self.x0 = sim.design_variables()
        self.nx = len(self.x0)
        self.nc = len(sim.constraints())

    def _setup_bounds(self):
        sim = self.options['simulator']

        # Set design variable bounds
        dv_meta = sim.get_design_variable_metadata()
        x_l = []
        x_u = []
        for key in sim.dv_keys:
            shape = sim[key].shape
            l = dv_meta[key]['lower']
            u = dv_meta[key]['upper']

            x_l = np.concatenate((x_l, (l * np.ones(shape)).flatten()))
            x_u = np.concatenate((x_u, (u * np.ones(shape)).flatten()))

        self.x_lower = np.where(x_l == -1.0e30, -np.inf, x_l)
        self.x_upper = np.where(x_u == 1.0e30, np.inf, x_u)

        # Set constraint bounds
        c_meta = sim.get_constraints_metadata()
        c_l = []
        c_u = []
        for key in sim.constraint_keys:
            shape = sim[key].shape
            e = c_meta[key]['equals']
            if e is None:
                l = c_meta[key]['lower']
                u = c_meta[key]['upper']
            else:
                l = e
                u = e

            c_l = np.concatenate((c_l, (l * np.ones(shape)).flatten()))
            c_u = np.concatenate((c_u, (u * np.ones(shape)).flatten()))

        self.c_lower = np.where(c_l == -1.0e30, -np.inf, c_l)
        self.c_upper = np.where(c_u == 1.0e30, np.inf, c_u)

    def objective(self, x):
        sim = self.options['simulator']
        sim.update_design_variables(x)
        failure_flag = sim.run(check_failure=True)
        print('Computing objective..........')
        return sim.objective()
        # return failure_flag, sim.objective()

    def objective_gradient(self, x):
        sim = self.options['simulator']
        sim.update_design_variables(x)
        f1 = sim.run(check_failure=True)
        f2 = sim.compute_total_derivatives(check_failure=True)
        failure_flag = (f1 and f2)
        print('Computing gradient..........')
        return sim.objective_gradient()
        # return failure_flag, sim.objective_gradient()

    def constraints(self, x):
        sim = self.options['simulator']
        sim.update_design_variables(x)
        failure_flag = sim.run(check_failure=True)
        print('Computing constraints..........')
        return sim.constraints()
        # return failure_flag, sim.constraints()

    def constraint_jacobian(self, x):
        sim = self.options['simulator']
        sim.update_design_variables(x)
        f1 = sim.run(check_failure=True)
        f2 = sim.compute_total_derivatives(check_failure=True)
        failure_flag = (f1 and f2)
        print('Computing jacobian..........')
        return sim.constraint_jacobian()
        # return failure_flag, sim.constraint_jacobian()

    def compute_all(self, x):
        sim = self.options['simulator']
        sim.update_design_variables(x)
        f1 = sim.run(check_failure=True)
        f2 = sim.compute_total_derivatives(check_failure=True)
        failure_flag = (f1 and f2)

        print('Computing all at once..........')
        return failure_flag, sim.objective(), sim.constraints(), sim.objective_gradient(), sim.constraint_jacobian()
