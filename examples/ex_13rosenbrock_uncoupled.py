'''Benchmark algorithms using the uncoupled Rosenbrock problem'''

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
    if nx % 2 != 0:
        raise ValueError('Number of variables must be even for uncoupled Rosenbrock function')
    for i in range(int(nx/2)):
        f += (1 - x[2*i])**2 + 100*(x[2*i + 1] - x[2*i]**2)**2
    return f

def grad(x):
    nx = len(x)
    g = np.zeros((nx))
    if nx % 2 != 0:
        raise ValueError('Number of variables must be even for uncoupled Rosenbrock function')
    for i in range(int(nx/2)):
        g[[2*i, 2*i + 1]] = np.array(
            [-2 * (1 - x[2*i]) - 400*x[2*i]*(x[2*i + 1] - x[2*i]**2), 200 * (x[2*i + 1] - x[2*i]**2)])
    return g

def obj_hess(x):
    nx = len(x)
    hess = np.zeros((nx, nx))
    if nx % 2 != 0:
        raise ValueError('Number of variables must be even for uncoupled Rosenbrock function')
    for i in range(int(nx/2)):
        hess[2*i : 2*(i+1), 2*i : 2*(i+1)] = np.array(
            [[2 + 800*x[2*i]**2 - 400*(x[2*i + 1] - x[2*i]**2), -400*x[2*i]], [-400*x[2*i], 200]])
    return hess

# Benchmarking optimization algorithms
algs = ['SNOPT', 'IPOPT', 'IPOPT-2', 'PySLSQP', 'BFGS', 'LBFGSB', 
        'COBYLA', 'COBYQA', 'NelderMead', 'TrustConstr', 'TrustConstr-2']

performance = {}
history = {}
time_loop = 1

for nx2 in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2056]:
    nx = 2*nx2
    sol = np.ones(nx)
    x0 = np.array([-1.2, 1] * nx2)
    name = f'rosenbrock(u)_{nx}d'

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
        elif alg == 'TrustConstr':
            options = {'ignore_exact_hessian': True}
        elif alg == 'SNOPT':
            options = {'Verbose': False}
            
        if (alg in ['COBYLA', 'COBYQA', 'NelderMead'] and nx >= 16) or (alg == 'PySLSQP' and nx >= 128) or (alg == 'BFGS' and nx >= 256):
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
        success  = np.allclose(results['x'], sol, atol=1e-1)
        
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
        if (alg in ['COBYLA', 'COBYQA', 'NelderMead'] and nx >= 16) or (alg == 'PySLSQP' and nx >= 128) or (alg == 'BFGS' and nx >= 256):
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