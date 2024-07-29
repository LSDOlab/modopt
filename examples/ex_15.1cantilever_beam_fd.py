'''Cantilever beam optimization with finite difference gradients'''

import numpy as np
from modopt import ProblemLite, SLSQP
import time

E0, L0, b0, vol0, F0 = 1., 1., 0.1, 0.01, -1.

def get_problem(n_el): # 16 lines excluding comments, returns, and jac function.

    E, L, b, vol = E0, L0, b0, vol0
    L_el = L / n_el
    n_nodes = n_el + 1

    def obj(x):
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
        K = np.zeros((n_nodes*2, n_nodes*2))
        for i in range(n_el):
            K[2*i:2*i+4, 2*i:2*i+4] += c_el * I[i]

        # Displacement vector - solve for u in Ku = F
        # Apply boundary conditions: u[0] = u[1] = 0, 
        # F[0:1] are unknown reaction forces at the left end. F[0:1] = K[0:1,2:].dot(u[2:])
        u = np.concatenate(([0., 0.], np.linalg.solve(K[2:,2:], F[2:])))

        # # Construction of sparse stiffness matrix from dense and solution
        # A_sparse = sp.csr_matrix(K[2:,2:])
        # u = np.concatenate(([0., 0.], sp.linalg.spsolve(A_sparse, F[2:])))

        # # Construction of sparse stiffness matrix and solution
        # nnz  = 4 + 12 * (n_el - 1)
        # data = np.zeros((nnz,))
        # rows = np.zeros((nnz,), dtype=int)
        # cols = np.zeros((nnz,), dtype=int)
        # data[0:4] = c_el[2:,2:].flatten() * I[0]
        # rows[0:4] = np.array([0, 0, 1, 1])
        # cols[0:4] = np.array([0, 1, 0, 1])
        # j = 4
        # for i in range(1, n_el):
        #     # NW quadrant: sum with SE quadrant of previous element
        #     data[j-4:j] += c_el[:2,:2].flatten() * I[i]

        #     # NE quadrant
        #     data[j:j+4] = c_el[:2,2:].flatten() * I[i]
        #     rows[j:j+4] = np.array([2*i-2, 2*i-2, 2*i-1, 2*i-1])
        #     cols[j:j+4] = np.array([2*i, 2*i+1, 2*i, 2*i+1])

        #     # SW quadrant
        #     data[j+4:j+8] = c_el[2:,:2].flatten() * I[i]
        #     rows[j+4:j+8] = np.array([2*i, 2*i, 2*i+1, 2*i+1])
        #     cols[j+4:j+8] = np.array([2*i-2, 2*i-1, 2*i-2, 2*i-1])

        #     # SE quadrant
        #     data[j+8:j+12] = c_el[2:,2:].flatten() * I[i]
        #     rows[j+8:j+12] = np.array([2*i, 2*i, 2*i+1, 2*i+1])
        #     cols[j+8:j+12] = np.array([2*i, 2*i+1, 2*i, 2*i+1])

        #     j += 12
        
        # K_sparse = sp.csc_matrix((data, (rows, cols)), shape=(2*n_nodes-2, 2*n_nodes-2))
        # u = np.concatenate(([0., 0.], sp.linalg.spsolve(K_sparse, F[2:])))

        # Compliance
        c = F.dot(u)

        return c

    def con(x):
        return np.array([L/n_el * b * np.sum(x) - vol])

    def jac(x):
        return L/n_el * b * np.ones((1, n_el))

    return ProblemLite(x0=np.ones(n_el), obj=obj, con=con, jac=jac,
                       name=f'Cantilever beam {n_el} elements FD',
                       xl=1e-2, cl=0., cu=0.)

# SLSQP
print('\tSLSQP \n\t-----')
optimizer = SLSQP(get_problem(50), solver_options={'maxiter': 1000, 'ftol': 1e-9})
start_time = time.time()
optimizer.solve()
opt_time = time.time() - start_time
success = optimizer.results['success']

print('\tTime:', opt_time)
print('\tSuccess:', success)
print('\tOptimized vars:', optimizer.results['x'])
print('\tOptimized obj:', optimizer.results['fun'])

optimizer.print_results()

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