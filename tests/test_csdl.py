# Test csdl and csdl_alpha interfaces

import pytest

@pytest.mark.interfaces
@pytest.mark.csdl
def test_csdl():
    import numpy as np
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import CSDLProblem
    from csdl import Model

    # minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

    class QuarticFunc(Model):
        def initialize(self):
            pass

        def define(self):
            x = self.create_input('x', val=1.)
            y = self.create_input('y', val=2.)

            z = x**4 + y**4
            self.register_output('z', z)

            constraint_1 = x + y
            constraint_2 = x - y
            self.register_output('constraint_1', constraint_1)
            self.register_output('constraint_2', constraint_2)

            self.add_design_variable('x', lower=0., scaler=100.)
            self.add_design_variable('y', scaler=0.2)
            self.add_objective('z', scaler=3.)
            self.add_constraint('constraint_1', equals=1., scaler=20.)
            self.add_constraint('constraint_2', lower=1., scaler=5.)

    # from csdl_om import Simulator
    from python_csdl_backend import Simulator

    sim = Simulator(QuarticFunc())
    
    prob = CSDLProblem(problem_name='quartic',simulator=sim)

    assert prob.problem_name == 'quartic'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.obj_scaler, {})                                 # no scaling done by modopt
    assert_array_equal(prob.x_scaler, np.array([], dtype=np.float64))       # no scaling done by modopt
    assert_array_equal(prob.c_scaler, np.array([], dtype=np.float64))       # no scaling done by modopt
    assert_array_equal(prob.x0, np.array([100., 0.4]))
    assert_array_equal(prob.x_lower, np.array([0, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [20., 5.])
    assert_array_equal(prob.c_upper, [20., np.inf])

    from modopt import SQP, SLSQP, SNOPT, PySLSQP
    optimizer = SLSQP(prob, ftol=1e-6, maxiter=20, disp=True)
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20})
    # optimizer = SQP(prob, maxiter=20)
    # optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3, Verbose=True)

    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(summary_table=True)

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 3., decimal=11)
    # assert_almost_equal(optimizer.results['objective'], 3., decimal=11)


@pytest.mark.interfaces
@pytest.mark.csdl_alpha
def test_csdl_alpha():
    import numpy as np
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import CSDLAlphaProblem
    import csdl_alpha as csdl

    # minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.
    rec = csdl.Recorder()
    rec.start()

    x = csdl.Variable(name = 'x', value=1.)
    y = csdl.Variable(name = 'y', value=2.)
    x.set_as_design_variable(lower = 0.0, scaler=100.0)
    y.set_as_design_variable(scaler=0.2)

    z = x**4 + y**4
    z.add_name('z')
    z.set_as_objective(scaler=3.)

    constraint_1 = x + y
    constraint_2 = x - y
    constraint_1.add_name('constraint_1')
    constraint_2.add_name('constraint_2')
    constraint_1.set_as_constraint(lower=1., upper=1., scaler=20.)
    constraint_2.set_as_constraint(lower=1., scaler=5.)

    rec.stop()

    sim = csdl.experimental.PySimulator(rec)

    prob = CSDLAlphaProblem(problem_name='quartic', simulator=sim)

    assert prob.problem_name == 'quartic'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.obj_scaler, {})
    assert_array_equal(prob.f_scaler, 3.)
    assert_array_equal(prob.x_scaler, np.array([100., 0.2]))
    assert_array_equal(prob.c_scaler, np.array([20., 5.]))
    assert_array_equal(prob.x0, np.array([100., 0.4]))
    assert_array_equal(prob.x_lower, np.array([0, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [20., 5.])
    assert_array_equal(prob.c_upper, [20., np.inf])

    from modopt import SQP, SLSQP, SNOPT, PySLSQP
    optimizer = SLSQP(prob, ftol=1e-6, maxiter=20, disp=True)
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20})
    # optimizer = SQP(prob, maxiter=20)
    # optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3, Verbose=True)

    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results(summary_table=True)

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 3., decimal=11)
    # assert_almost_equal(optimizer.results['objective'], 3., decimal=11)


# @pytest.mark.interfaces
# @pytest.mark.csdl_alpha
# def test_csdl_alpha_jax():
#     import numpy as np
#     from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
#     from modopt import CSDLAlphaProblem
#     import csdl_alpha as csdl

#     # minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.
#     rec = csdl.Recorder()
#     rec.start()

#     x = csdl.Variable(name = 'x', value=1.)
#     y = csdl.Variable(name = 'y', value=2.)
#     x.set_as_design_variable(lower = 0.0, scaler=100.0)
#     y.set_as_design_variable(scaler=0.2)

#     z = x**4 + y**4
#     z.add_name('z')
#     z.set_as_objective(scaler=3.)

#     constraint_1 = x + y
#     constraint_2 = x - y
#     constraint_1.add_name('constraint_1')
#     constraint_2.add_name('constraint_2')
#     constraint_1.set_as_constraint(lower=1., upper=1., scaler=20.)
#     constraint_2.set_as_constraint(lower=1., scaler=5.)

#     rec.stop()

#     sim = csdl.experimental.JaxSimulator(rec)

#     prob = CSDLAlphaProblem(problem_name='quartic', simulator=sim)

#     assert prob.problem_name == 'quartic'
#     assert prob.constrained == True
#     assert prob.options['jac_format'] == 'dense'
#     assert_array_equal(prob.obj_scaler, {})
#     assert_array_equal(prob.f_scaler, 3.)
#     assert_array_equal(prob.x_scaler, np.array([100., 0.2]))
#     assert_array_equal(prob.c_scaler, np.array([20., 5.]))
#     assert_array_equal(prob.x0, np.array([100., 0.4]))
#     assert_array_equal(prob.x_lower, np.array([0, -np.inf]))
#     assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
#     assert_array_equal(prob.c_lower, [20., 5.])
#     assert_array_equal(prob.c_upper, [20., np.inf])

#     from modopt import SQP, SLSQP, SNOPT, PySLSQP
#     optimizer = SLSQP(prob, ftol=1e-6, maxiter=20, disp=True)
#     # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20})
#     # optimizer = SQP(prob, maxiter=20)
#     # optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3, Verbose=True)

#     optimizer.check_first_derivatives(prob.x0)
#     optimizer.solve()
#     optimizer.print_results(summary_table=True)

#     assert optimizer.results['success'] == True
#     assert optimizer.results['message'] == 'Optimization terminated successfully'
#     assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
#     assert_almost_equal(optimizer.results['fun'], 3., decimal=11)
#     # assert_almost_equal(optimizer.results['objective'], 3., decimal=11)


if __name__ == '__main__':
    test_csdl()
    test_csdl_alpha()
    # test_csdl_alpha_jax()
    print("All tests passed!")