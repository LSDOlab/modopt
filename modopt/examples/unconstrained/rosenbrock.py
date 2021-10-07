import numpy as np

from modopt.api import Problem


class Rosenbrock(Problem):
    # a = 1
    # b = 100
    def initialize(self, ):
        self.problem_name = 'rosenbrock'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=None,
                                  upper=None,
                                  equals=None,
                                  vals=np.array([-1.2, 1.]))

        self.add_objective('obj')

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(
            wrt='x',
            shape=(2, ),
        )

    def compute_objective(self, x):
        f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        return f

    def compute_objective_gradient(self, x):
        grad = np.array([
            -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1),
            200 * (x[1] - x[0]**2)
        ])
        return grad

    def compute_objective_hessian(self, x):
        hess = np.array([[2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
                         [-400 * x[0], 200]])

        return hess

    def compute_objective_hvp(self, x, p):
        hess = np.array([[2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
                         [-400 * x[0], 200]])
        hvp = np.matmul(hess, p)

        return hvp


def main():
    x = np.array([-1, 1])
    nx = np.size(x)
    step = 1e-8

    grad_exact = Rosenbrock.grad(x)
    grad_fd = np.zeros((nx))
    for i in range(nx):
        dx = np.zeros((nx))
        dx[i] = step
        grad_fd[i] = (Rosenbrock.func(x + dx) -
                      Rosenbrock.func(x)) / step
    grad_check = grad_exact - grad_fd
    print("Grad check:", grad_exact, grad_fd, grad_check,
          np.linalg.norm(grad_check))

    hess_exact = Rosenbrock.hessian(x)
    hess_fd = np.zeros((nx, nx))
    for i in range(nx):
        dx = np.zeros((nx))
        dx[i] = step
        hess_fd[i] = (Rosenbrock.grad(x + dx) -
                      Rosenbrock.grad(x)) / step
    hess_check = hess_exact - hess_fd
    print("Hess check:", hess_exact, hess_fd, hess_check,
          np.linalg.norm(hess_check))
