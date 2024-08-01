'''Cantilever beam optimization with Jax'''

import numpy as np
from modopt import JaxProblem, SLSQP
import time

import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)

E0, L0, b0, vol0, F0 = 1., 1., 0.1, 0.01, -1.

# METHOD 1: Use Jax functions directly in mo.JaxProblem. 
#           ModOpt will auto-generate the gradient, Jacobian, and objective Hessian.
#           ModOpt will also auto-generate the Lagrangian, its gradient, and Hessian.
#           No need to manually generate or jit functions or their derivatives and then wrap them.

def get_problem(n_el): # 16 statements excluding comments, and returns.

    E, L, b, vol = E0, L0, b0, vol0
    L_el = L / n_el
    n_nodes = n_el + 1

    def jax_obj(x):
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
        K = jnp.zeros((n_nodes*2, n_nodes*2))
        for i in range(n_el):
            K = K.at[2*i:2*i+4, 2*i:2*i+4].add(c_el * I[i])
            # K[2*i:2*i+4, 2*i:2*i+4] += c_el * I[i]

        # Displacement vector - solve for u in Ku = F
        # Apply boundary conditions: u[0] = u[1] = 0,
        # F[0:1] are unknown reaction forces at the left end. F[0:1] = K[0:1,2:].dot(u[2:])
        u = jnp.concatenate((np.array([0., 0.]), jnp.linalg.solve(K[2:,2:], F[2:])))
        # Compliance
        c = jnp.dot(F, u)

        return c

    def jax_con(x):
        return jnp.array([L_el * b * jnp.sum(x) - vol])

    return JaxProblem(x0=np.ones(n_el), nc=1, jax_obj=jax_obj, jax_con=jax_con,
                      name=f'cantilever_{n_el}_jax',
                      xl=1e-2, cl=0., cu=0.)

if __name__ == '__main__':
    # # Test to see if the problem is correctly defined
    # prob = get_problem(50)
    # print(prob._compute_objective(np.ones(50)))       # 39.99999999905752
    # print(prob._compute_constraints(np.ones(50)))     # [0.09]
    # exit()

    # SLSQP
    print('\tSLSQP \n\t-----')
    n_el = 50
    optimizer = SLSQP(get_problem(n_el), solver_options={'maxiter': 1000, 'ftol': 1e-9})
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

# # METHOD 2: Create jitted Jax functions and derivatives, and
# #           wrap them manually before passing to ProblemLite.
# from modopt import JaxProblem, SLSQP
# def get_problem(n_el): # 16 lines excluding comments, and returns.

#     E, L, b, vol = E0, L0, b0, vol0
#     L_el = L / n_el
#     n_nodes = n_el + 1

#     def jax_obj(x):
#         # Moment of inertia
#         I = b * x**3 / 12

#         # Force vector
#         F = np.zeros((n_nodes*2,))
#         F[-2] = F0

#         # Stiffness matrix
#         c_el = E / L_el**3 * np.array([[12, 6*L_el, -12, 6*L_el],
#                                     [6*L_el, 4*L_el**2, -6*L_el, 2*L_el**2],
#                                     [-12, -6*L_el, 12, -6*L_el],
#                                     [6*L_el, 2*L_el**2, -6*L_el, 4*L_el**2]])
#         K = jnp.zeros((n_nodes*2, n_nodes*2))
#         for i in range(n_el):
#             K = K.at[2*i:2*i+4, 2*i:2*i+4].add(c_el * I[i])
#             # K[2*i:2*i+4, 2*i:2*i+4] += c_el * I[i]

#         # Displacement vector - solve for u in Ku = F
#         # Apply boundary conditions: u[0] = u[1] = 0,
#         # F[0:1] are unknown reaction forces at the left end. F[0:1] = K[0:1,2:].dot(u[2:])
#         u = jnp.concatenate((np.array([0., 0.]), jnp.linalg.solve(K[2:,2:], F[2:])))
#         # Compliance
#         c = jnp.dot(F, u)

#         return c

#     def jax_con(x):
#         return jnp.array([L_el * b * jnp.sum(x) - vol])

#     _obj = jax.jit(jax_obj)
#     _con = jax.jit(jax_con)

#     _grad = jax.jit(jax.grad(jax_obj))
#     _jac = jax.jit(jax.jacfwd(jax_con))

#     obj  = lambda x: np.float64(_obj(x))
#     grad = lambda x: np.array(_grad(x))
#     con  = lambda x: np.array(_con(x))
#     jac  = lambda x: np.array(_jac(x))

#     return ProblemLite(x0=np.ones(n_el), obj=obj, grad=grad, con=con, jac=jac,
#                        name=f'Cantilever beam {n_el} elements Jax',
#                        xl=1e-2, cl=0., cu=0.)

# if __name__ == '__main__':
#     # SLSQP
#     print('\tSLSQP \n\t-----')
#     optimizer = SLSQP(get_problem(50), solver_options={'maxiter': 1000, 'ftol': 1e-9})
#     start_time = time.time()
#     optimizer.solve()
#     opt_time = time.time() - start_time
#     success = optimizer.results['success']

#     print('\tTime:', opt_time)
#     print('\tSuccess:', success)
#     print('\tOptimized vars:', optimizer.results['x'])
#     print('\tOptimized obj:', optimizer.results['fun'])

#     optimizer.print_results()

#     assert np.allclose(optimizer.results['x'],
#                         [0.14915754,  0.14764328,  0.14611321,  0.14456715,  0.14300421,  0.14142417,
#                         0.13982611,  0.13820976,  0.13657406,  0.13491866,  0.13324268,  0.13154528,
#                         0.12982575,  0.12808305,  0.12631658,  0.12452477,  0.12270701,  0.12086183,
#                         0.11898809,  0.11708424,  0.11514904,  0.11318072,  0.11117762,  0.10913764,
#                         0.10705891,  0.10493903,  0.10277539,  0.10056526,  0.09830546,  0.09599246,
#                         0.09362243,  0.09119084,  0.08869265,  0.08612198,  0.08347229,  0.08073573,
#                         0.07790323,  0.07496382,  0.07190453,  0.06870925,  0.0653583,   0.06182632,
#                         0.05808044,  0.05407658,  0.04975295,  0.0450185,   0.03972912,  0.03363155,
#                         0.02620192,  0.01610863], rtol=0, atol=1e-5)