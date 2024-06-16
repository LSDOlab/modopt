'''Optimization with OpenMDAO models: minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1'''

import openmdao.api as om
import numpy as np

# minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

class QuarticFunc(om.ExplicitComponent):
    def setup(self): 
        # add_inputs
        self.add_input('x', 1.)
        self.add_input('y', 1.)
        
        # add_outputs
        self.add_output('objective')
        self.add_output('constraint_1')
        self.add_output('constraint_2')

        # declare_partials
        self.declare_partials(of='objective', wrt='*')
        # self.declare_partials(of='objective', wrt='*', method='cs')
        self.declare_partials(of='constraint_1', wrt='x', val=1.)
        self.declare_partials(of='constraint_1', wrt='y', val=1.)
        self.declare_partials(of='constraint_2', wrt='x', val=1.)
        self.declare_partials(of='constraint_2', wrt='y', val=-1.)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['objective'] = x**4 + y**4
        outputs['constraint_1'] = x + y
        outputs['constraint_2'] = x - y

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']

        partials['objective', 'x'] = 4 * x**3
        partials['objective', 'y'] = 4 * y**3


if __name__ == "__main__":
    # Create OpenMDAO Problem
    om_prob = om.Problem()
    # Add subsystem to the OpenMDAO Problem model
    om_prob.model.add_subsystem('quartic', QuarticFunc(), promotes=['*'])

    # Add optimization variables and functions to the Problem model
    om_prob.model.add_design_var('x', lower=0., scaler=4., adder=1.)
    om_prob.model.add_design_var('y', scaler=1., adder=0.)
    om_prob.model.add_objective('objective', scaler=1., adder=1.)
    om_prob.model.add_constraint('constraint_1', equals=1., scaler=1., adder=0.)
    om_prob.model.add_constraint('constraint_2', lower=1., scaler=1., adder=0.)

    # Setup the OpenMDAO problem
    om_prob.setup()

    # Set initial values
    om_prob.set_val('x', 5.0)
    om_prob.set_val('y', 10.0)

    # Print UNSCALED variables
    print('Initial x value: ', om_prob.get_val('x'))
    print('Initial y value: ', om_prob.get_val('y'))

    # Run the model with initial values to compute the objective and constraints
    # om_prob.run_model()
    print('Initial objective value: ', om_prob.get_val('objective'))
    print('Initial constraint 1 value: ', om_prob.get_val('constraint_1'))
    print('Initial constraint 2 value: ', om_prob.get_val('constraint_2'))

    from modopt import OpenMDAOProblem

    # Instantiate the modopt OpenMDAOProblem() object that wraps for modopt
    # the Problem() object defined earlier, and name your problem
    prob = OpenMDAOProblem(problem_name='quartic', om_problem=om_prob)

    from modopt import SQP, SLSQP, SNOPT

    # Setup your preferred optimizer (here, SLSQP) with the Problem object 
    # Pass in the options for your chosen optimizer
    optimizer = SLSQP(prob, ftol=1e-6, maxiter=20, outputs=['x'])
    # optimizer = SQP(prob, maxiter=20)
    # optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3, Verbose=True)

    # Check first derivatives at the initial guess, if needed
    # optimizer.check_first_derivatives(prob.x0)
    # om_prob.check_totals(compact_print=True)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization (summary_table contains information from each iteration)
    optimizer.print_results(summary_table=True)

    print('Num model evals: ', prob.model_evals)
    print('Num deriv evals: ', prob.deriv_evals)
    print('Optimal x value: ', om_prob.get_val('x'))
    print('Optimal y value: ', om_prob.get_val('y'))
    print('Optimal objective value: ', om_prob.get_val('objective'))
    print('Optimal constraint 1 value: ', om_prob.get_val('constraint_1'))
    print('Optimal constraint 2 value: ', om_prob.get_val('constraint_2'))
