'''Cantilever beam optimization with CSDL'''

import numpy as np
import scipy.sparse as sp
from modopt import CSDLAlphaProblem, SLSQP
import time
import csdl_alpha as csdl

E0, L0, b0, vol0, F0 = 1., 1., 0.1, 0.01, -1.

def get_problem(n_el): # 23 statements excluding comments, returns, name assignment, 
                       # and an extra line for concatenating u with [0., 0.].
    rec = csdl.Recorder()
    rec.start()

    # add design variables
    x = csdl.Variable(name = 'x', shape=(n_el,), value=1.0)
    x.set_as_design_variable(lower = 1e-2)

    # add objective
    E, L, b, vol = E0, L0, b0, vol0
    L_el = L / n_el
    n_nodes = n_el + 1

    # Moment of inertia
    I = b * x**3 / 12

    # Force vector
    F = np.zeros((n_nodes*2,))
    F[-2] = F0

    # Stiffness matrix
    c_el = E / L_el**3 * np.array([[12, 6*L_el, -12, 6*L_el],
                                   [6*L_el, 4*L_el**2, -6*L_el, 2*L_el**2],
                                   [-12, -6*L_el, 12, -6*L_el],
                                   [6*L_el, 2*L_el**2, -6*L_el, 4*L_el**2]])
    
    K = csdl.Variable(name='K', value=np.zeros((n_nodes*2, n_nodes*2)))
    for i in range(n_el):
        K = K.set(csdl.slice[2*i:2*i+4, 2*i:2*i+4], K[2*i:2*i+4, 2*i:2*i+4] + c_el * I[i])
        # K[2*i:2*i+4, 2*i:2*i+4] += c_el * I[i]
    u = csdl.solve_linear(K[2:,2:], F[2:])
    # Missing statement for concatenating u with [0., 0.]. Add 1 extra line.
    c = csdl.vdot(F[2:], u)
    c.add_name('compliance')
    c.set_as_objective()

    # add constraints
    v = L_el * b * csdl.sum(x)
    v.add_name('volume')
    v.set_as_constraint(lower=vol0, upper=vol0)

    rec.stop()

    # Create a Simulator object from the Recorder object
    # sim = csdl.experimental.PySimulator(rec)
    sim = csdl.experimental.JaxSimulator(rec)

    # Instantiate your problem using the csdl Simulator object and name your problem
    prob = CSDLAlphaProblem(problem_name=f'Cantilever beam {n_el} elements CSDL', simulator=sim)

    return prob

# f = get_problem(50)._compute_objective(np.ones(50))
# print('f:', f)
# c = get_problem(50)._compute_constraints(np.ones(50))
# print('c:', c)

# SLSQP
print('\tSLSQP \n\t-----')
n_el = 50
optimizer = SLSQP(get_problem(n_el), solver_options={'maxiter': 200, 'ftol': 1e-9})
optimizer.check_first_derivatives(np.ones(50))
start_time = time.time()
optimizer.solve()
opt_time = time.time() - start_time
success = optimizer.results['success']

print('\tTime:', opt_time)
print('\tSuccess:', success)
print('\tOptimized vars:', optimizer.results['x'])
print('\tOptimized obj:', optimizer.results['fun'])

optimizer.print_results()

import matplotlib.pyplot as plt

plt.figure()
plt.plot(optimizer.results['x'])
plt.xlabel('Lengthwise location')
plt.ylabel('Optimized thickness')
plt.show()

assert np.allclose(optimizer.results['x'],
                    [0.14915754,  0.14764328,  0.14611321,  0.14456715,  0.14300421,  0.14142417,
                    0.13982611,  0.13820976,  0.13657406,  0.13491866,  0.13324268,  0.13154528,
                    0.12982575,  0.12808305,  0.12631658,  0.12452477,  0.12270701,  0.12086183,
                    0.11898809,  0.11708424,  0.11514904,  0.11318072,  0.11117762,  0.10913764,
                    0.10705891,  0.10493903,  0.10277539,  0.10056526,  0.09830546,  0.09599246,
                    0.09362243,  0.09119084,  0.08869265,  0.08612198,  0.08347229,  0.08073573,
                    0.07790323,  0.07496382,  0.07190453,  0.06870925,  0.0653583,   0.06182632,
                    0.05808044,  0.05407658,  0.04975295,  0.0450185,   0.03972912,  0.03363155,
                    0.02620192,  0.01610863], rtol=0, atol=1e-5)