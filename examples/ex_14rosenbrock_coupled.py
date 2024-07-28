'''Benchmark optimization algorithms on coupled Rosenbrock problem'''

import numpy as np
from modopt import ProblemLite, optimize
from modopt.postprocessing import load_variables
import time
import contextlib
import io
import matplotlib.pyplot as plt

def obj(x):
    nx = len(x)
    f = 0
    for i in range(nx-1):
        f += (1 - x[i])**2 + 100*(x[i + 1] - x[i]**2)**2
    return f

def grad(x):
    nx = len(x)
    g = np.zeros((nx,))
    for i in range(nx-1):
        g[[i, i + 1]] += np.array(
            [-2 * (1 - x[i]) - 400*x[i]*(x[i + 1] - x[i]**2), 200 * (x[i + 1] - x[i]**2)])
    return g

def obj_hess(x):
    nx = len(x)
    hess = np.zeros((nx, nx))
    for i in range(nx-1):
        hess[i : i+2, i : i+2] += np.array(
            [[2 + 800*x[i]**2 - 400*(x[i + 1] - x[i]**2), -400*x[i]], [-400*x[i], 200]])
    return hess

# from modopt import IPOPT
# for nx in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
#     x0   = np.array([-1.2, 1.] * int(nx/2))
#     name = f'rosenbrock(c)_{nx}d'

#     prob = ProblemLite(x0, name=name, obj=obj, grad=grad, obj_hess=obj_hess)
#     print('\nProblem:', prob.problem_name)
#     print('='*50)

#     optimizer = IPOPT(prob, solver_options={
#         'hessian_approximation': 'exact',
#         'derivative_test_print_all': 'no',
#         'derivative_test': 'second-order'})
#     results = optimizer.solve()

# exit()

# Benchmarking optimization algorithms
algs = ['SNOPT', 'IPOPT', 'IPOPT-2', 'PySLSQP', 'BFGS', 'LBFGSB', 
        'COBYLA', 'COBYQA', 'NelderMead', 'TrustConstr', 'TrustConstr-2']

performance = {}
history = {}
time_loop = 1

for nx in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    sol_g    = np.ones(nx)
    sol_l    = np.ones(nx)
    sol_l[0] = -1.0
    sols     = [sol_g, sol_l]

    x0   = np.array([-1.2, 1.] * int(nx/2))
    name = f'rosenbrock(c)_{nx}d'

    prob = ProblemLite(x0, name=name, obj=obj, grad=grad, obj_hess=obj_hess)
    print('\nProblem:', prob.problem_name)
    print('='*50)
    
    for alg in algs:
        solver = alg
        options = {}
        if alg=='IPOPT-2':
            solver = 'IPOPT'
            options = {'hessian_approximation': 'exact'}
        elif alg=='TrustConstr-2':
            solver = 'TrustConstr'
            options = {'maxiter': 2000}
        elif alg == 'TrustConstr':
            options = {'ignore_exact_hessian': True, 'maxiter': 2000}
        elif alg == 'SNOPT':
            options = {'Verbose': False, 'Major optimality': 1e-9, 'Major iterations': 2000}
        elif alg in ['PySLSQP', 'BFGS']:
            options = {'maxiter': 2000}
        elif alg in ['LBFGSB']:
            options = {'maxiter': 2000, 'maxfun':3000}
            
        if (alg in ['COBYLA', 'COBYQA', 'NelderMead'] and nx >= 16) or (alg in ['IPOPT'] and nx >= 256):
            performance[prob.problem_name, alg] = {'time': 1e6,
                                                   'success': False,
                                                   'nev': 1e6,
                                                   'objective': 1e6}
            continue
        
        print(f'\t{alg} \n\t------------------------')
        start_time = time.time()
        for i in range(time_loop-1):
            with contextlib.redirect_stdout(io.StringIO()):
                results = optimize(prob, solver=solver, solver_options=options, recording=False, turn_off_outputs=True)
        results  = optimize(prob, solver=solver, solver_options=options, recording=True)
        opt_time = (time.time() - start_time) / time_loop
        success  = np.allclose(results['x'], sols[0], atol=1e-1) or np.allclose(results['x'], sols[1], atol=1e-1)
        
        nev       = prob._callback_count
        o_evals   = prob._obj_count
        g_evals   = prob._grad_count
        h_evals   = prob._hess_count
        objective = prob._compute_objective(results['x'])
        print('\tTime:', opt_time)
        print('\tSuccess:', success)
        print('\tEvaluations:', nev)
        print('\tObj evals:', o_evals)
        print('\tGrad evals:', g_evals)
        print('\tHess evals:', h_evals)
        print('\tOptimized vars:', results['x'])

        obj_hist = load_variables(f"{results['out_dir']}/record.hdf5", 'obj')['callback_obj']
        history[prob.problem_name, alg] = obj_hist

        performance[prob.problem_name, alg] = {'time': opt_time,
                                               'success': success,
                                               'nev': nev,
                                               'objective': objective}
        
    plt.figure()
    for alg in algs:
        if (alg in ['COBYLA', 'COBYQA', 'NelderMead'] and nx >= 16) or (alg in ['IPOPT'] and nx >= 256):
            continue
        y_data = history[prob.problem_name, alg]
        plt.semilogy(y_data, label=f"{alg} ({len(y_data)})")
    plt.xlabel('Evaluations')
    plt.ylabel('Objective')
    plt.title(f'{prob.problem_name} minimization')
    plt.legend()
    plt.grid()
    plt.savefig(f"{prob.problem_name}-objective-cb.pdf")
    plt.close()

# Print performance
print('\nPerformance')
print('='*50)
for key, value in performance.items():
    print(f"{str(key):45}:", value)

from modopt.benchmarking import plot_performance_profiles
plot_performance_profiles(performance, save_figname='performance.pdf')