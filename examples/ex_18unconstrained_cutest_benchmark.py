'''Benchmark algorithms on unconstrained CUTEst problems (nx<=100) with second-order derivatives'''

import numpy as np
from modopt import CUTEstProblem, optimize
import pycutest
from modopt.postprocessing import load_variables
import time
import contextlib
import io
import matplotlib.pyplot as plt

# Benchmarking optimization algorithms
algs = ['SNOPT', 'IPOPT', 'IPOPT-2', 'PySLSQP', 'BFGS', 'LBFGSB', 'TrustConstr', 'TrustConstr-2']
        # 'COBYLA', 'COBYQA', 'NelderMead', 'TrustConstr', 'TrustConstr-2']

performance = {}
history = {}
time_loop = 1

# Limits on the the number of optimization variables
nx_lower = 0
nx_upper = 100

# List of all unconstrained problems (total = 240) with second-order derivatives
prob_list_U = pycutest.find_problems(constraints='unconstrained', degree=[2,2])

# List of all problems with fixed variables (total = 19)
# These problems are equivalently unconstrained problems if we import them
# setting 'drop_fixed_variables=True' (which is the default)
prob_list_X = pycutest.find_problems(constraints='fixed', degree=[2,2])

# Sorted list of all unconstrained (after dropping fixed variables) problems
prob_list = sorted(prob_list_U + prob_list_X)
print('# of problems:', len(prob_list_U), len(prob_list_X), len(prob_list)) # 240 19 259 (same as degree=[0,2])

# Loop over all unconstrained problems, importing and solving them
# if the number of variables is >= nx_lower and <= nx_upper
for i, prob_name in enumerate(prob_list):
    
    # Default number of optimization variables for the given problem
    nx = pycutest.problem_properties(prob_name)['n']

    # nx = 'variable' means n can be set by the user
    sifParams = {}
    if nx == 'variable':
        # pycutest problem object
        sifParams = {'N': 100}
        sifParams = {'N': 75} if prob_name == 'BRATU1D' else sifParams
        sifParams = {'N/2': 50} if prob_name == 'BROYDN7D' else sifParams
        sifParams = {'NS': 49} if prob_name == 'CHAINWOO' else sifParams
        sifParams = {'N': 50} if prob_name in ['CHNROSNB', 'CHNRSNBM'] else sifParams
        sifParams = {'P': 10} if prob_name in ['CLPLATEA', 'CLPLATEB', 'CLPLATEC'] else sifParams
        sifParams = {'M': 49} if prob_name == 'CRAGGLVY' else sifParams
        sifParams = {'M': 30} if prob_name in ['DIXMAANA', 'DIXMAANB', 'DIXMAANC', 'DIXMAAND', 'DIXMAANE', 'DIXMAANF', 'DIXMAANG', 'DIXMAANH', 'DIXMAANI', 'DIXMAANJ', 'DIXMAANK', 'DIXMAANL', 'DIXMAANM', 'DIXMAANN', 'DIXMAANO', 'DIXMAANP'] else sifParams
        sifParams = {'M': 10} if prob_name in ['DRCAV1LQ', 'DRCAV2LQ', 'DRCAV3LQ'] else sifParams
        sifParams = {'N': 36} if prob_name == 'EDENSCH' else sifParams
        # sifParams = {'N': 50} if prob_name in ['EIGENALS', 'EIGENBLS'] else sifParams
        # sifParams = {'M': 25} if prob_name == 'EIGENCLS' else sifParams
        sifParams = {'N': 50} if prob_name in ['ERRINROS', 'ERRINRSM'] else sifParams
        sifParams = {'P': 8} if prob_name in ['FMINSRF2', 'FMINSURF'] else sifParams
        sifParams = {'N':10} if prob_name in ['HILBERTA', 'HILBERTB'] else sifParams

        sifParams = {'N':10} if prob_name in ['INTEQNELS'] else sifParams
        sifParams = {'P':8} if prob_name in ['LMINSURF'] else sifParams
        # No user-settable sifParams for the following problems
        sifParams = {} if prob_name in ['LUKSAN11LS', 'LUKSAN12LS', 'LUKSAN13LS', 'LUKSAN14LS', 'LUKSAN15LS', 'LUKSAN16LS', 'LUKSAN17LS', 'LUKSAN21LS', 'LUKSAN22LS'] else sifParams
        sifParams = {'N/2':5} if prob_name in ['MODBEALE'] else sifParams
        sifParams = {'P':10} if prob_name in ['MSQRTALS', 'MSQRTBLS'] else sifParams
        sifParams = {'P':8} if prob_name in ['NLMSURF'] else sifParams
        sifParams = {'P':10} if prob_name in ['NONMSQRT'] else sifParams
        sifParams = {'NX':10, 'NY':10} if prob_name in ['ODC'] else sifParams
        sifParams = {'N/2':50} if prob_name in ['PENALTY3'] else sifParams
        sifParams = {'NKNOTS':32} if prob_name in ['RAYBENDL'] else sifParams
        sifParams = {'NK':32} if prob_name in ['RAYBENDS'] else sifParams
        sifParams = {'M':34} if prob_name in ['SPMSRTLS'] else sifParams
        sifParams = {'N/2':50} if prob_name in ['SROSENBR'] else sifParams
        sifParams = {'NX':10, 'NY':10} if prob_name in ['SSC'] else sifParams
        sifParams = {'N':99} if prob_name in ['VAREIGVL'] else sifParams
        sifParams = {'N':31} if prob_name in ['WATSON'] else sifParams
        sifParams = {'NS':25} if prob_name in ['WOODS'] else sifParams
        sifParams = {'N':2} if prob_name in ['YATP2LS'] else sifParams

    elif nx < nx_lower or nx > nx_upper:
        continue

    if prob_name.startswith('DMN'):
        continue # Skip problems that are slow to load even after caching
    if prob_name in ['EIGENALS', 'EIGENBLS', 'EIGENCLS', 'NCB20', 'TESTQUAD', 'YATP1LS']:
        continue # Skip these problems since they have a large number (2550, 110, 1000, 120) of variables not within 100
    
    # Import pycutest problem
    pc_prob = pycutest.import_problem(prob_name, sifParams=sifParams)
    print(f'[{i}.]', 'Problem name [num_vars]:', prob_name, f'[{pc_prob.n}]')
    print('='*50)

    # Create modopt problem
    prob = CUTEstProblem(cutest_problem=pc_prob)
    
    for alg in algs:
        solver = alg
        options = {}
        if alg=='IPOPT-2':
            solver = 'IPOPT'
            options = {'hessian_approximation': 'exact', 'max_iter':500, 'print_level': 0}
        elif alg=='IPOPT':
            options = {'hessian_approximation': 'limited-memory', 'max_iter':500, 'print_level': 0}
        elif alg=='TrustConstr-2':
            solver = 'TrustConstr'
            options = {'maxiter': 500}
        elif alg == 'TrustConstr':
            options = {'ignore_exact_hessian': True, 'maxiter': 500}
        elif alg == 'SNOPT':
            options = {'Verbose': False, 'Major optimality': 1e-6, 'Major iterations': 500}
        elif alg == 'PySLSQP':
            options = {'maxiter': 500, 'iprint': 0}
        elif alg == 'BFGS':
            options = {'maxiter': 500}
        elif alg in ['LBFGSB']:
            options = {'maxiter': 500, 'maxfun':1500}
            
        # if (alg in ['COBYLA', 'COBYQA', 'NelderMead'] and nx >= 16) or (alg in ['IPOPT'] and nx >= 256):
        #     performance[prob.problem_name, alg] = {'time': 1e6,
        #                                            'success': False,
        #                                            'nev': 1e6,
        #                                            'objective': 1e6}
        #     continue
        
        print(f'\t{alg} \n\t------------------------')
        start_time = time.time()
        for i in range(time_loop-1):
            with contextlib.redirect_stdout(io.StringIO()):
                results = optimize(prob, solver=solver, solver_options=options, recording=False, turn_off_outputs=True)
        results  = optimize(prob, solver=solver, solver_options=options, recording=True)
        opt_time = (time.time() - start_time) / time_loop

        if alg == 'SNOPT':
            success = (results['info'] == 1)
        elif alg in ['IPOPT', 'IPOPT-2']:
            with open(f"{results['out_dir']}/ipopt_output.txt", 'r') as f:
                ipopt_output = f.read()
            success = ('EXIT: Optimal Solution Found.' in ipopt_output)
        else:
            success  = results['success']
        
        nev       = prob._callback_count
        o_evals   = prob._obj_count
        g_evals   = prob._grad_count
        h_evals   = prob._hess_count
        objective = prob._compute_objective(results['x'])
        # print('\tTime:', opt_time)
        # print('\tSuccess:', success)
        # print('\tEvaluations:', nev)
        # print('\tObj evals:', o_evals)
        # print('\tGrad evals:', g_evals)
        # print('\tHess evals:', h_evals)
        # print('\tOptimized vars:', results['x'])

        # obj_hist = load_variables(f"{results['out_dir']}/record.hdf5", 'obj')['callback_obj']
        # history[prob.problem_name, alg] = obj_hist

        performance[prob.problem_name, alg] = {'time': opt_time,
                                               'success': success,
                                               'nev': nev,
                                               'objective': objective}

# Print performance
print('\nPerformance')
print('='*50)
for key, value in performance.items():
    print(f"{str(key):45}:", value)

import pickle
with open('performance.pkl', 'wb') as f:
    pickle.dump(performance, f)

from modopt.benchmarking import plot_performance_profiles
plot_performance_profiles(performance, save_figname='performance.pdf')