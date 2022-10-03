from csdl import Model
import pytest
import numpy.testing
import csdl
import numpy as np

# minimize x^2 + y^2 subject to x>=0, x+y=1, x-y>=1.


class QuadraticFunc(Model):
    def initialize(self):
        pass

    def define(self):
        # add_inputs
        x = self.create_input('x', val=1.)
        y = self.create_input('y', val=1.)

        z = x**2 + y**2

        # add_outputs
        self.register_output('z', z)

        constraint_1 = x + y
        constraint_2 = x - y

        constraint = self.create_output('constraint', shape=(2, ))
        constraint[0] = constraint_1
        constraint[1] = constraint_2
        # self.register_output('constraint_1', constraint_1)
        # self.register_output('constraint_2', constraint_2)
        # self.register_output('constraint', constraint)

        # define optimization problem
        self.add_design_variable('x', lower=0., scaler=5.)
        self.add_design_variable('y', scaler=7.)
        self.add_objective('z', scaler=10)
        # self.add_constraint('constraint_1', equals=1., scaler=2.)
        # self.add_constraint('constraint_2', lower=1., scaler=0.1)
        self.add_constraint('constraint',
                            lower=np.array([1., 1.]),
                            upper=np.array([1., np.inf]),
                            scaler=np.array([2., 0.1]))


if __name__ == "__main__":
    # def test_scaler():
    # from csdl_om import Simulator
    from python_csdl_backend import Simulator

    sim = Simulator(QuadraticFunc())

    from modopt.csdl_library import CSDLProblem

    prob = CSDLProblem(
        problem_name='quadratic',
        simulator=sim,
    )

    # from modopt.optimization_algorithms import SQP
    # from modopt.scipy_library import SLSQP
    from modopt.snopt_library import SNOPT

    # optimizer = SLSQP(
    #     prob,
    #     ftol=1e-6,
    #     maxiter=20,
    #     outputs=['x'],
    # )
    # optimizer.options['ftol'] = 1e-6

    # optimizer = SQP(prob, max_itr=20)

    optimizer = SNOPT(prob,
                      Infinite_bound=1.0e20,
                      Verify_level=-1,
                    #   append2file=True)
                      append2file=False)

    # Check first derivatives at the initial guess, if needed
    optimizer.check_first_derivatives(prob.x0)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization (summary_table contains information from each iteration)
    # optimizer.print_results(summary_table=True)

    # # sim.prob.set_val('v', val=0.)
    # sim.prob.driver = om.ScipyOptimizeDriver()
    # sim.prob.driver.options['optimizer'] = 'SLSQP'

    # # sim.prob.run_model()
    # sim.prob.run_driver()
    # sim.run()

    print(sim['x'])
    print(sim['y'])
    print(sim['z'])
    # print(sim['constraint_1'])
    # print(sim['constraint_2'])
    print(sim['constraint'])
    print(optimizer.snopt_output)

    # numpy.testing.assert_almost_equal(actual=optimizer.outputs['x'][-1,0], desired=sim['x']*5)
    # numpy.testing.assert_almost_equal(actual=optimizer.outputs['x'][-1,1], desired=sim['y']*7)
    # numpy.testing.assert_almost_equal(actual=optimizer.outputs['obj'][-1], desired=sim['z']*10)
    # numpy.testing.assert_almost_equal(actual=optimizer.outputs['con'][-1], desired=sim['z']*10)

    # numpy.testing.assert_almost_equal(actual=optimizer.snopt_output['x'][-1,0], desired=sim['x']*5)
    # numpy.testing.assert_almost_equal(actual=optimizer.snopt_output['x'][-1,1], desired=sim['y']*7)
    # numpy.testing.assert_almost_equal(actual=optimizer.snopt_output['obj'][-1], desired=sim['z']*10)
    # numpy.testing.assert_almost_equal(actual=optimizer.snopt_output['con'][-1], desired=sim['z']*10)
