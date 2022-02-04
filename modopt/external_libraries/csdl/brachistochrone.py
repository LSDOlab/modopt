from csdl import Model, sin, cos, sum, expand
from ozone.api import ODEProblem
import numpy as np
import matplotlib.pyplot as plt


# ODE Model
class BrachistochroneODE(Model):
    def initialize(self):
        n = self.parameters.declare('num_nodes')

    def define(self):
        n = self.parameters['num_nodes']

        y = self.create_input('y', shape=(n, ))
        x = self.create_input('x', shape=(n, ))
        v = self.create_input('v', shape=(n, ))
        theta = self.create_input('theta', shape=(n, ))

        self.register_output('dxdt', v * sin(theta))
        self.register_output('dydt', v * cos(theta))
        self.register_output('dvdt', -9.81 * cos(theta))


class Run(Model):
    def define(self):

        dt = 0.1
        nt = 10

        # initial condition
        y_0 = self.create_input('y_0', val=0.0)
        x_0 = self.create_input('x_0', val=0.0)
        v_0 = self.create_input('v_0', val=0.0)

        # Timestep vector
        dt = self.create_input('dt', val=0.05)
        n_vec = self.create_input('n', val=np.ones(nt))
        self.register_output('h', expand(dt, nt) * n_vec)

        # Theta is the dynamic parameter design variable
        theta = self.create_input('theta', -np.ones(nt) * 0.1)

        # Ode problem
        ode_problem = ODEProblem('RK4',
                                 'time-marching',
                                 nt,
                                 visualization='end')
        ode_problem.add_state('x',
                              'dxdt',
                              initial_condition_name='x_0',
                              output='x_solved')
        ode_problem.add_state('y',
                              'dydt',
                              initial_condition_name='y_0',
                              output='y_solved')
        ode_problem.add_state('v', 'dvdt', initial_condition_name='v_0')
        ode_problem.add_parameter('theta', dynamic=True, shape=(nt, ))
        ode_problem.add_times(step_vector='h')
        ode_problem.set_ode_system(BrachistochroneODE)

        # Add integrator
        self.add(ode_problem.create_solver_model())

        # solved solution
        x_solved = self.declare_variable('x_solved', shape=(nt + 1, ))
        y_solved = self.declare_variable('y_solved', shape=(nt + 1, ))
        h = self.declare_variable('h', shape=(nt, ))

        # final position
        y_final = y_solved[nt]
        x_final = x_solved[nt]

        # constraint variables: Final position and minimum dt
        x_constraint = x_final
        y_constraint = y_final
        dt_constraint = dt * 1.1

        # final time
        self.register_output('time_end', sum(h))
        self.register_output('x_constraint', x_constraint)
        self.register_output('y_constraint', y_constraint)
        self.register_output('dt_constraint', dt_constraint)

        # define optimization problem
        self.add_design_variable('theta')
        self.add_design_variable('dt')
        self.add_objective('time_end')
        self.add_constraint('x_constraint', equals=2.0)
        self.add_constraint('y_constraint', equals=-2.0)
        self.add_constraint('dt_constraint', lower=0.05)


if __name__ == "__main__":
    from csdl_om import Simulator

    sim = Simulator(Run(), mode='rev')
    # sim.prob.run_model()
    # sim.prob.check_totals(compact_print=True)
    # exit()

    from modopt.csdl_library import CSDLProblem

    # Setup your optimization problem
    prob = CSDLProblem(
        problem_name='ode_sample',
        simulator=sim,
    )

    from modopt.scipy_library import SLSQP
    from modopt.optimization_algorithms import SQP
    from modopt.snopt_library import SNOPT

    # Setup your optimizer with the problem
    # optimizer = SLSQP(prob, maxiter=40)
    # optimizer = SQP(prob, max_itr=40)
    optimizer = SNOPT(prob,
                      Major_iterations=20,
                      Infinite_bound=1.0e20,
                      Verify_level=3)

    # Check first derivatives at the initial guess, if needed
    # optimizer.check_first_derivatives(prob.x0)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization
    # (summary_table contains information from each iteration)
    optimizer.print_results(summary_table=True)

    # plot optimized solution
    plt.figure()
    plt.plot(sim.prob['x_solved'], sim.prob['y_solved'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Brachistochrone Curve')

    plt.show()
