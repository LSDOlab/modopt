from all_problem_types import *
# Important NOTE: All 8 problem types of both Problem and ProblemLite classes 
# are tested here yield the exact same results to numerical precision. 
# This is because the ProblemLite class and the Problem class
# are identical in terms of their computations.
# This is helpful for testing either one of the classes against the other.

# Another NOTE: All 16 problems yield teh exact same results to numerical precision
# when using the SLSQP or PySLSQP optimizers.
# This is beacuse both optimizers use the same underlying SLSQP algorithm in Fortran.

# All optimizations will be tested using Scipy SLSQP
from modopt import SLSQP
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal

def test_ProblemClass(): # 8 problems
    from modopt import PySLSQP, SNOPT

    prob = Unconstrained()
    assert prob.problem_name == 'unconstrained'
    assert prob.constrained == False
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.x0, np.array([500., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert prob.c_lower == None
    assert prob.c_upper == None
    
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-12, 'maxiter':20})
    optimizer = SLSQP(prob, maxiter=20, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=3)
    assert_almost_equal(optimizer.results['fun'], 0., decimal=11)

    prob = Feasibility()
    assert prob.problem_name == 'feasibility'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.x0, np.array([500., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1., 1.])
    assert_array_equal(prob.c_upper, [1., 1.])

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-12, 'maxiter':20})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=10)
    assert_almost_equal(optimizer.results['fun'], 0., decimal=11)

    # This one's a difficult problem for SLSQP (and PySLSQP) starting from [500., 5.]
    prob = BoundConstrained()
    assert prob.problem_name == 'bound_constrained'
    assert prob.constrained == False
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.x0, np.array([50., 5.]))
    assert_array_equal(prob.x_lower, np.array([1., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, 10.]))
    assert prob.c_lower == None
    assert prob.c_upper == None

    # This one's a difficult problem for SLSQP (and PySLSQP)
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':50})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    # optimizer = SNOPT(prob,)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=1)
    assert_almost_equal(optimizer.results['fun'], 1., decimal=6)

    prob = EqConstrained()
    assert prob.problem_name == 'eq_constrained'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.x0, np.array([500., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1.])
    assert_array_equal(prob.c_upper, [1.])

    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [0.5, 0.5], decimal=6)
    assert_almost_equal(optimizer.results['fun'], 0.125, decimal=11)

    prob = IneqConstrained()
    assert prob.problem_name == 'ineq_constrained'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.x0, np.array([50., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1.])
    assert_array_equal(prob.c_upper, [np.inf])

    optimizer = SLSQP(prob, maxiter=50, ftol=1e-12, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [0.5, -0.5], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 0.125, decimal=11)

    prob = Constrained()
    assert prob.problem_name == 'constrained'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.x0, np.array([50., 5.]))
    assert_array_equal(prob.x_lower, np.array([0., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1., 1.])
    assert_array_equal(prob.c_upper, [1., np.inf])

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 1., decimal=11)
    # assert_almost_equal(optimizer.results['objective'], 1., decimal=11)

    prob = Scaling()
    assert prob.problem_name == 'scaling'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.obj_scaler, {'f': 20.})
    assert_array_equal(prob.x_scaler, [2., 0.2])
    assert_array_equal(prob.c_scaler, [5., 0.5])
    assert_array_equal(prob.x0, np.array([100., 1.]))
    assert_array_equal(prob.x_lower, np.array([0., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [5., 0.5])
    assert_array_equal(prob.c_upper, [5., np.inf])

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    # assert_almost_equal(optimizer.results['objective'], 20., decimal=6)

    prob = FiniteDiff()
    assert prob.problem_name == 'finite_diff'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.obj_scaler, {'f': 20.})
    assert_array_equal(prob.x_scaler, [2., 0.2])
    assert_array_equal(prob.c_scaler, [5., 0.5])
    assert_array_equal(prob.x0, np.array([50., 1.]))
    assert_array_equal(prob.x_lower, np.array([0., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [5., 0.5])
    assert_array_equal(prob.c_upper, [5., np.inf])

    # With FD derivs, this one's a difficult problem for SLSQP (and PySLSQP) starting from even [30., 5.]
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':50, 'summary_filename':'fd_summary.out'})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=7)
    # assert_almost_equal(optimizer.results['objective'], 20., decimal=7)

    # Check scaled finite difference derivatives at scaled solution [2., 0.] (unscaled [1., 0.])
    assert_array_almost_equal(prob._compute_objective_gradient(np.array([2., 0.])), [40., 0.], decimal=4)
    assert_array_almost_equal(prob._compute_objective_hessian(np.array([2., 0.])), [[60., 0.], [0., 0.]], decimal=3)
    assert_array_almost_equal(prob._compute_objective_hvp(np.array([2.,0.]), np.array([1.,1.])), [60., 0.], decimal=3)

    assert_array_almost_equal(prob._compute_constraint_jacobian(np.array([2., 0.])), [[2.5, 25.], [0.5, -2.5]], decimal=6)
    assert_array_almost_equal(prob._compute_constraint_jvp(np.array([2., 0.]), np.array([1.,1.])), [27.5, -2.0], decimal=7)



def test_ProblemLiteClass(): # 8 problems
    from modopt import PySLSQP, SNOPT

    prob = unconstrained_lite()
    assert prob.problem_name == 'unconstrained_lite'
    assert prob.constrained == False
    assert_array_equal(prob.x0, np.array([500., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert prob.c_lower == None
    assert prob.c_upper == None
    
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-12, 'maxiter':20})
    optimizer = SLSQP(prob, maxiter=20, disp=True)
    # optimizer = SNOPT(prob,)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=3)
    assert_almost_equal(optimizer.results['fun'], 0., decimal=11)

    prob = feasibility_lite()
    assert prob.problem_name == 'feasibility_lite'
    assert prob.constrained == True
    assert_array_equal(prob.x0, np.array([500., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1., 1.])
    assert_array_equal(prob.c_upper, [1., 1.])

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-12, 'maxiter':20})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=10)
    assert_almost_equal(optimizer.results['fun'], 0., decimal=11)

    prob = bound_constrained_lite()
    assert prob.problem_name == 'bound_constrained_lite'
    assert prob.constrained == False
    assert_array_equal(prob.x0, np.array([50., 5.]))
    assert_array_equal(prob.x_lower, np.array([1., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, 10.]))
    assert prob.c_lower == None
    assert prob.c_upper == None

    # This one's a difficult problem for SLSQP (and PySLSQP) starting from [500., 5.]
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':50})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    # optimizer = SNOPT(prob,)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=1)
    assert_almost_equal(optimizer.results['fun'], 1., decimal=6)

    prob = eq_constrained_lite()
    assert prob.problem_name == 'eq_constrained_lite'
    assert prob.constrained == True
    assert_array_equal(prob.x0, np.array([500., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1.])
    assert_array_equal(prob.c_upper, [1.])

    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [0.5, 0.5], decimal=6)
    assert_almost_equal(optimizer.results['fun'], 0.125, decimal=11)

    prob = ineq_constrained_lite()
    assert prob.problem_name == 'ineq_constrained_lite'
    assert prob.constrained == True
    assert_array_equal(prob.x0, np.array([50., 5.]))
    assert_array_equal(prob.x_lower, np.array([-np.inf, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1.])
    assert_array_equal(prob.c_upper, [np.inf])

    optimizer = SLSQP(prob, maxiter=50, ftol=1e-12, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [0.5, -0.5], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 0.125, decimal=11)

    prob = constrained_lite()
    assert prob.problem_name == 'constrained_lite'
    assert prob.constrained == True
    assert_array_equal(prob.x0, np.array([50., 5.]))
    assert_array_equal(prob.x_lower, np.array([0., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [1., 1.])
    assert_array_equal(prob.c_upper, [1., np.inf])

    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 1., decimal=11)

    prob = scaling_lite()
    assert prob.problem_name == 'scaling_lite'
    assert prob.constrained == True
    assert_array_equal(prob.f_scaler,  [20.])
    assert_array_equal(prob.x_scaler, [2., 0.2])
    assert_array_equal(prob.c_scaler, [5., 0.5])
    assert_array_equal(prob.x0, np.array([100., 1.]))
    assert_array_equal(prob.x_lower, np.array([-0., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [5., 0.5])
    assert_array_equal(prob.c_upper, [5., np.inf])

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_lite_summary.out'})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    # assert_almost_equal(optimizer.results['objective'], 20., decimal=6)

    prob = finite_diff_lite()
    assert prob.problem_name == 'finite_diff_lite'
    assert prob.constrained == True
    assert_array_equal(prob.f_scaler, [20.])
    assert_array_equal(prob.x_scaler, [2., 0.2])
    assert_array_equal(prob.c_scaler, [5., 0.5])
    assert_array_equal(prob.x0, np.array([50., 1.]))
    assert_array_equal(prob.x_lower, np.array([0., -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [5., 0.5])
    assert_array_equal(prob.c_upper, [5., np.inf])

    # With FD derivs, this one's a difficult problem for SLSQP (and PySLSQP) starting from even [30., 5.]
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'fd_lite_summary.out'})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    # optimizer = SNOPT(prob,)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=10)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=8)

    # Check scaled finite difference derivatives at scaled solution [2., 0.] (unscaled [1., 0.])
    assert_array_almost_equal(prob._compute_objective_gradient(np.array([2., 0.])), [40., 0.], decimal=4)
    assert_array_almost_equal(prob._compute_objective_hessian(np.array([2., 0.])), [[60., 0.], [0., 0.]], decimal=3)
    assert_array_almost_equal(prob._compute_objective_hvp(np.array([2.,0.]), np.array([1.,1.])), [60., 0.], decimal=3)

    assert_array_almost_equal(prob._compute_constraint_jacobian(np.array([2., 0.])), [[2.5, 25.], [0.5, -2.5]], decimal=6)
    assert_array_almost_equal(prob._compute_constraint_jvp(np.array([2., 0.]), np.array([1.,1.])), [27.5, -2.0], decimal=7)




if __name__ == "__main__":
    test_ProblemClass()
    test_ProblemLiteClass()
