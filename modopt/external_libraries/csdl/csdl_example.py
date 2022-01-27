from csdl_om import Simulator
from csdl import Model
import numpy as np

# minimize x^2 + y^2 subject to x>=0, x+y=1, x-y>=1.


class QuadraticFunc(Model):
    def initialize(self):
        # self.parameters.declare('surfaces', types=list)
        pass

    def define(self):
        # add_parameter
        # surfaces = self.parameters['surfaces']

        # add_inputs
        x = self.create_input('x', val=1.)
        y = self.create_input('y', val=1.)

        z = x**2 + y**2

        # add_outputs
        self.register_output('z', z)

        constraint_1 = x + y
        constraint_2 = x - y
        self.register_output('constraint_1', constraint_1)
        self.register_output('constraint_2', constraint_2)

        # define optimization problem
        self.add_design_variable('x', lower=0.)
        self.add_design_variable('y', lower=0.)
        self.add_objective('z')
        self.add_constraint('constraint_1', equals=1.)
        self.add_constraint('constraint_2', lower=1.)


if __name__ == "__main__":
    import openmdao.api as om

    sim = Simulator(QuadraticFunc())

    # sim.design_variables()
    # sim.constraints()
    # sim.objective_gradient()
    # sim.constraint_jacobian()

    sim.prob.driver = om.ScipyOptimizeDriver()
    sim.prob.driver.options['optimizer'] = 'SLSQP'

    # sim.prob.run_model()
    sim.prob.run_driver()
    # sim.visualize_model()
    # sim.run()
    print(sim['x'])
    print(sim['y'])
    print(sim['z'])