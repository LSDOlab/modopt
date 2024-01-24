from csdl import Model

# minimize x^2 + y^2 subject to x>=0, x+y=1, x-y>=1.


class QuadraticFunc(Model):
    def initialize(self):
        pass

    def define(self):
        # add_inputs
        x = self.create_input('x', val=1.)
        y = self.create_input('y', val=1.)

        z = x**4 + y**4

        # add_outputs
        self.register_output('z', z)

        constraint_1 = x + y
        constraint_2 = x - y
        self.register_output('constraint_1', constraint_1)
        self.register_output('constraint_2', constraint_2)

        # define optimization problem
        self.add_design_variable('x', lower=0.)
        self.add_design_variable('y')
        self.add_objective('z')
        self.add_constraint('constraint_1', equals=1.)
        self.add_constraint('constraint_2', lower=1.)


if __name__ == "__main__":
    from csdl_om import Simulator

    sim = Simulator(QuadraticFunc())

    from modopt import CSDLProblem

    prob = CSDLProblem(
        problem_name='quadratic',
        simulator=sim,
    )

    from modopt import SQP
    from modopt import SLSQP
    from modopt import SNOPT

    # optimizer = SLSQP(
    #     prob,
    #     ftol=1e-6,
    #     maxiter=20,
    #     outputs=['x'],
    # )
    # optimizer.options['ftol'] = 1e-6

    # optimizer = SQP(prob, max_itr=20)

    optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3)
    # ftol=1e-6,
    # maxiter=20,
    # )
    # outputs=['x'])

    # Check first derivatives at the initial guess, if needed
    # optimizer.check_first_derivatives(prob.x0)

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
