import numpy as np
from optimize import snopta
import time

from .snopt_optimizer import SNOPTOptimizer


class SNOPTa(SNOPTOptimizer):
    def declare_options(self):
        self.solver_name += 'a'

        # Solver-specific options exactly as in SNOPT with defaults
        # self.options.declare('maxiter', default=100, types=int)

    def setup(self):
        self.update_SNOPT_options_object()
        self.setup_bounds()
        if self.problem.constrained:
            self.setup_constraints()

    # def setup_constraints(self, ):
    #     pass

    def solve(self):
        # Assign shorter names to variables and methods
        x0 = self.x0

        ObjRow = 1
        n = self.problem.nx
        nF = self.problem.nc + 1

        obj = self.obj
        grad = self.grad
        G = grad(x0)

        if self.problem.nc > 0:
            con = self.con
            jac = self.jac
            G = np.append(G.reshape(1, n), jac(x0), axis=0)

        start_time = time.time()

        c_lower = self.c_lower
        c_upper = self.c_upper
        x_lower = self.x_lower
        x_upper = self.x_upper

        inf = self.solver_options['Infinite bound']

        f_lower = np.concatenate([
            inf,
        ], c_lower)
        f_upper = np.concatenate([
            inf,
        ], c_upper)

        def snopta_objconFG(status, x, needF, F, needG, G):
            F[0] = obj(x)
            G[0] = grad(x)

            if self.problem.nc > 0:
                F[1:] = con(x)
                G[1:] = jac(x)

            return status, F, G

        results = snopta(
            snopta_objconFG,
            n,
            nF,
            x0=x0,
            xlow=x_lower,
            xupp=x_upper,
            Flow=f_lower,
            Fupp=f_upper,
            ObjRow=ObjRow,
            # A=A,
            G=G,
            # xnames=xnames,
            # Fnames=Fnames,
            name=self.problem_name,
            options=self.SNOPT_options_object)

        end_time = time.time()
        self.total_time = end_time - start_time

        print(results)

        self.results = results
