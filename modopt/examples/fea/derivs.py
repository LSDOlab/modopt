import numpy as np
from fea import FEA


num_elements = 400
num_design_variables = 10
dvs = np.ones(num_design_variables, dtype=complex)

fea_object = FEA(num_elements=num_elements, num_design_variables=num_design_variables)

c = fea_object.evaluate(dvs)

# FD
h = 1e-6
grad_fd = np.zeros(num_design_variables)
c0 = fea_object.evaluate(dvs)
for ind in range(num_design_variables):
    dvs[ind] += h
    c = fea_object.evaluate(dvs)
    dvs[ind] -= h
    grad_fd[ind] = (c - c0) / h

# CS
h = 1e-16
grad_cs = np.zeros(num_design_variables)
for ind in range(num_design_variables):
    dvs[ind] += complex(0, h)
    c = fea_object.evaluate(dvs)
    dvs[ind] -= complex(0, h)
    grad_cs[ind] = np.imag(c) / h

K = fea_object.compute_K()
u = fea_object.solve(K)
pRpx = fea_object.compute_pRph(u)
pRpy = fea_object.compute_K()
pFpx = 0.
pFpy = fea_object.get_forces()

# Direct
dydx = np.linalg.solve(pRpy, -pRpx)
grad_dr = pFpx + pFpy.dot(dydx)
grad_dr = np.real(grad_dr)

# Adjoint
dfdr = np.linalg.solve(pRpy.T, -pFpy)
grad_aj = pFpx + pRpx.T.dot(dfdr)
grad_aj = np.real(grad_aj)

print(grad_fd)
print(grad_cs)
print(grad_dr)
print(grad_aj)

print(np.linalg.norm(grad_cs - grad_fd))
print(np.linalg.norm(grad_dr - grad_fd))
print(np.linalg.norm(grad_aj - grad_fd))