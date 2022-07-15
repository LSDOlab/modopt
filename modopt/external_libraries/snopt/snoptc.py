import numpy as np
from optimize import snoptc
import time

from .snopt_optimizer import SNOPTOptimizer


class SNOPTc(SNOPTOptimizer):
    def declare_options(self):
        self.solver_name += 'c'

        # Solver-specific options exactly as in SNOPT with defaults
        # self.options.declare('maxiter', default=100, types=int)

    def declare_outputs(self, ):
        self.default_outputs_format = {
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
        }

        self.options.declare('outputs', types=list, default=['x'])

    def setup(self):
        self.update_SNOPT_options_object()
        self.setup_bounds()
        self.setup_constraints()

    # def setup_constraints(self, ):
    #     pass

    def solve(self):
        # Assign shorter names to variables and methods
        x0 = self.x0
        x0c0 = x0.copy()
        # self.update_outputs(x0)

        # ObjRow = 1
        n = self.problem.nx
        m = self.problem.nc
        # print(m)
        # nF = self.problem.nc + 1

        obj = self.obj
        grad = self.grad
        G = np.ones((n, ))

        if self.problem.nc > 0:
            con = self.con
            jac = self.jac
            J = np.ones((m, n))
            x0c0 = np.concatenate((x0, con(x0)))
        # callback = self.update_outputs

        start_time = time.time()

        bl = np.concatenate((self.x_lower, self.c_lower))
        bu = np.concatenate((self.x_upper, self.c_upper))

        # options = SNOPT_options()
        inf = 1.0e+20

        # options.setOption('Infinite bound', inf)
        # options.setOption('Verify level', 3)
        # options.setOption('Print filename', 'sntoyc.out')

        nnCon = m
        nnJac = n
        nnObj = n

        # x0 = np.zeros(n + m, float)

        # J = np.array([[100.0, 100.0, 1.0, 0], [0.0, 100.0, 0, 1.0],
        #               [2.0, 4.0, 0, 0], [0.0, 0.0, 3.0, 5.0]])

        inf = self.options['Infinite_bound']

        def snopta_objconFG(mode, nnjac, x, fObj, gObj, fCon, gCon,
                            nState):
            fObj = obj(x)
            gObj = grad(x)

            if self.problem.nc > 0:
                fCon = con(x)
                gCon = jac(x).flatten('f')
            else:
                fCon = 0.
                gcon = [0.]

            return mode, fObj, gObj, fCon, gCon

        if self.problem.nc == 0:
            nnJac = 0
            nnCon = 0
            m = 1
            locA = np.ones((n + 1, ))
            locA[0] = 0
            result = snoptc(snopta_objconFG,
                            nnObj=nnObj,
                            nnCon=nnCon,
                            nnJac=nnJac,
                            x0=np.append(x0, 0.),
                            J=(np.array([0]), np.array([0]), locA),
                            name=self.problem_name,
                            iObj=0,
                            bl=np.append(bl, -inf),
                            bu=np.append(bu, inf),
                            options=self.SNOPT_options_object,
                            m=m,
                            n=n)
        else:
            result = snoptc(snopta_objconFG,
                            nnObj=nnObj,
                            nnCon=nnCon,
                            nnJac=nnJac,
                            x0=x0c0,
                            J=J,
                            name=self.problem_name,
                            iObj=0,
                            bl=bl,
                            bu=bu,
                            options=self.SNOPT_options_object)

        end_time = time.time()
        self.total_time = end_time - start_time

        self.snopt_output = result
