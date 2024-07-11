# Test openMDAO interface

import pytest

from modopt import OpenMDAOProblem
import openmdao.api as om

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

# Create OpenMDAO Problem
om_prob = om.Problem()
# Add subsystem to the OpenMDAO Problem model
om_prob.model.add_subsystem('quartic', QuarticFunc(), promotes=['*'])

# Add optimization variables and functions to the Problem model
om_prob.model.add_design_var('x', lower=0., scaler=100., adder=1.)
om_prob.model.add_design_var('y', scaler=0.2, adder=0.)
om_prob.model.add_objective('objective', scaler=3., adder=1.)
om_prob.model.add_constraint('constraint_1', equals=1., scaler=20., adder=0.)
om_prob.model.add_constraint('constraint_2', lower=1., scaler=5., adder=1.)

# Setup the OpenMDAO problem
om_prob.setup()

# Set initial values
om_prob.set_val('x', 1.0)
om_prob.set_val('y', 2.0)

prob = OpenMDAOProblem(problem_name='quartic', om_problem=om_prob)

@pytest.mark.interfaces
@pytest.mark.openmdao
def test_openmdao():
    import numpy as np
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

    assert prob.problem_name == 'quartic'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.obj_scaler, {})                                 # no scaling done by modopt
    assert_array_equal(prob.x_scaler, np.array([], dtype=np.float64))       # no scaling done by modopt
    assert_array_equal(prob.c_scaler, np.array([], dtype=np.float64))       # no scaling done by modopt
    assert_array_equal(prob.x0, np.array([200., 0.4]))
    assert_array_equal(prob.x_lower, np.array([100., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [20., 10.])
    assert_array_equal(prob.c_upper, [20., np.inf])

    from modopt import SQP, SLSQP, SNOPT, PySLSQP
    optimizer = SLSQP(prob, solver_options={'maxiter': 20, 'disp': True, 'ftol': 1e-6})
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20})
    # optimizer = SQP(prob, maxiter=20)
    snopt_options = {
        'Inifinite_bound': 1.0e20,
        'Verify_level': 3,
        'Verbose': True,
    }
    # optimizer = SNOPT(prob, solver_options=snopt_options)

    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results()

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_almost_equal(optimizer.results['x'], [200., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 6., decimal=11)

if __name__ == '__main__':
    test_openmdao()
    print("All tests passed!")