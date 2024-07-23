# Test benchmarking utilities

import numpy as np
from modopt import ProblemLite
import time

# Problem 1: minimize sum(x^4)
x0 = np.array([100., 50.])
name = 'unconstrained'
obj  = lambda x: np.sum(x**4)
grad = lambda x: 4 * x**3
obj_hess = lambda x: 12 * np.diag(x**2)
prob1 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess)
sol1  = np.array([0., 0.])

# Problem 2: minimize sum(x^2) ; [1, 2] <= x <= [inf, inf]
x0 = np.array([100., 50.])
name = 'bound-constrained'
obj  = lambda x: np.sum(x**2)
grad = lambda x: 2 * x
obj_hess = lambda x: 2 * np.eye(len(x))
xl = np.array([1.0, 2.0])
prob2 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, xl=xl)
sol2  = np.array([1., 2.])

# Problem 3: minimize sum(x^2) ; x[0] + x[1] = 1
x0 = np.array([100., 50.])
name = 'equality-constrained'
obj  = lambda x: np.sum(x**2)
grad = lambda x: 2 * x
obj_hess = lambda x: 2 * np.eye(len(x))
con = lambda x: np.array([x[0] + x[1]])
jac = lambda x: np.array([[1, 1]])
lag_hess = lambda x, v: 2 * np.eye(len(x))
cl = 1.0
cu = 1.0
prob3 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess, 
                    con=con, jac=jac, lag_hess=lag_hess, cl=cl, cu=cu)
sol3  = np.array([0.5, 0.5])

# Problem 4: minimize sum(x^2) ; x[0] - x[1] >= 1
x0 = np.array([100., 50.])
name = 'inequality-constrained'
obj  = lambda x: np.sum(x**2)
grad = lambda x: 2 * x
obj_hess = lambda x: 2 * np.eye(len(x))
con = lambda x: np.array([x[0] - x[1]])
jac = lambda x: np.array([[1, -1]])
lag_hess = lambda x, v: 2 * np.eye(len(x))
cl = 1.0
prob4 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess,
                    con=con, jac=jac, lag_hess=lag_hess, cl=cl)
sol4  = np.array([0.5, -0.5])

sol = [sol1, sol2, sol3, sol4]

def run_benchmarking():
    from modopt import SLSQP, IPOPT
    performance = {}
    for i, prob in enumerate([prob1, prob2, prob3, prob4]):
        print('\nProblem:', prob.problem_name)
        print('='*50)
        
        # SLSQP
        print('\tSLSQP \n\t-----')
        optimizer  = SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8})
        start_time = time.time()
        results    = optimizer.solve()
        
        opt_time   = time.time() - start_time
        nev        = results['total_callbacks']
        success    = results['success']
        print('\tTime:', opt_time)
        print('\tEvaluations:', nev)
        print('\tSuccess:', success)
        print('\tOptimized vars:', results['x'])
        print('\tOptimized obj:',  results['fun'])
        performance[prob.problem_name, 'SLSQP'] = {'time': opt_time,
                                                   'nev': nev,
                                                   'success': success}

        # IPOPT
        print('\tIPOPT \n\t-----')
        optimizer  = IPOPT(prob, solver_options={'max_iter': 100, 'tol': 1e-12, 'print_level': 0})
        start_time = time.time()
        results    = optimizer.solve()

        opt_time   = time.time() - start_time
        nev        = results['total_callbacks']
        success    = np.allclose(results['x'], sol[i], atol=1e-2)
        print('\tTime:', opt_time)
        print('\tSuccess:', success)
        print('\tOptimized vars:', results['x'])
        print('\tOptimized obj:',  results['f'])
        performance[prob.problem_name, 'IPOPT'] = {'time': opt_time,
                                                   'nev': nev,
                                                   'success': success} 

    # Print performance
    print('\nPerformance')
    print('='*50)
    for key, value in performance.items():
        print(f"{str(key):40}:", value)

    return performance

import pytest

@pytest.mark.slsqp
@pytest.mark.ipopt
@pytest.mark.interfaces
@pytest.mark.benchmarking
def test_generate_performance_profiles():
    performance = run_benchmarking()
    from modopt.benchmarking import generate_performance_profiles
    generate_performance_profiles(performance)

@pytest.mark.slsqp
@pytest.mark.ipopt
@pytest.mark.interfaces
@pytest.mark.benchmarking
@pytest.mark.visualization
def test_plot_performance_profiles():
    performance = run_benchmarking()
    from modopt.benchmarking import plot_performance_profiles
    plot_performance_profiles(performance, save_figname='performance.pdf')