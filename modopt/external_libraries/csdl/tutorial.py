from csdl import Model


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
        self.register_output('constraint_1', constraint_1)
        self.register_output('constraint_2', constraint_2)

        # define optimization problem
        self.add_design_variable('x', lower=1.5)
        self.add_design_variable('y', lower=-1.)
        self.add_objective('z')
        self.add_constraint('constraint_1', equals=1.)
        self.add_constraint('constraint_2', lower=1.)


if __name__ == "__main__":
    from csdl_om import Simulator

    sim = Simulator(QuadraticFunc())

    from modopt.csdl_library import CSDLProblem

    # Setup your optimization problem
    prob = CSDLProblem(
        problem_name='quadratic',
        simulator=sim,
    )

    from modopt.scipy_library import SLSQP
    from modopt.optimization_algorithms import SQP
    from modopt.snopt_library import SNOPT

    # Setup your optimizer with the problem
    optimizer = SLSQP(prob, maxiter=20)
    # optimizer = SQP(prob, max_itr=20)
    # optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3)

    # Check first derivatives at the initial guess, if needed
    optimizer.check_first_derivatives(prob.x0)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization
    # (summary_table contains information from each iteration)
    optimizer.print_results(summary_table=True)

    # import openmdao.api as om
    # sim.prob.driver = om.ScipyOptimizeDriver()
    # sim.prob.driver.options['optimizer'] = 'SLSQP'

    # # sim.prob.run_model()
    # sim.prob.run_driver()

    print(sim['x'])
    print(sim['y'])