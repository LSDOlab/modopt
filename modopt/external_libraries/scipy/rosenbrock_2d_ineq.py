import numpy as np

import sys

sys.path.append("..")

from modopt.api import Problem


class Rosenbrock2d(Problem):
    a = 1
    b = 100

    def initialize(self):
        self.problem_name = 'rosenbrock_2d'

    def evaluate_objective(self, x):
        f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        return f

    def evaluate_constraints_and_residuals(self, x):
        c_lower = x[0]**2 + x[1]**2 - 2
        c_upper = -c_lower
        return np.array([c_lower, c_upper])

    def compute_gradient(self, x):
        grad = np.array([
            -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1),
            200 * (x[1] - x[0]**2)
        ])
        return grad

    def compute_constraint_and_residual_jacobian(self, x):
        jac_lower = np.array([[2 * x[0], 2 * x[1]]])
        jac_upper = -jac_lower
        return np.concatenate((jac_lower, jac_upper), axis=0)

    def compute_hessian(self, x):
        hess = np.array([[2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
                         [-400 * x[0], 200]])

        return hess

    def compute_hvp(self, x, p):
        hess = np.array([[2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
                         [-400 * x[0], 200]])
        hvp = np.matmul(hess, p)

        return hvp


def main():
    x = np.array([-1, 1])
    nx = np.size(x)
    step = 1e-8

    grad_exact = Rosenbrock2d.grad(x)
    grad_fd = np.zeros((nx))
    for i in range(nx):
        dx = np.zeros((nx))
        dx[i] = step
        grad_fd[i] = (Rosenbrock2d.func(x + dx) -
                      Rosenbrock2d.func(x)) / step
    grad_check = grad_exact - grad_fd
    print("Grad check:", grad_exact, grad_fd, grad_check,
          np.linalg.norm(grad_check))

    hess_exact = Rosenbrock2d.hessian(x)
    hess_fd = np.zeros((nx, nx))
    for i in range(nx):
        dx = np.zeros((nx))
        dx[i] = step
        hess_fd[i] = (Rosenbrock2d.grad(x + dx) -
                      Rosenbrock2d.grad(x)) / step
    hess_check = hess_exact - hess_fd
    print("Hess check:", hess_exact, hess_fd, hess_check,
          np.linalg.norm(hess_check))
