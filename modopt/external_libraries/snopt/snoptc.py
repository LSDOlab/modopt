import numpy as np
import warnings
try:
    from optimize import snoptc
except:
    warnings.warn("snoptc from 'optimize' could not be imported")
try:
    from optimize.solvers import snopt7_python as fsnopt
except:
    warnings.warn("snopt7_python from 'optimize.solvers' could not be imported")
import time

from .snopt_optimizer import SNOPTOptimizer


class SNOPTc(SNOPTOptimizer):
    def setup(self):
        self.solver_name += '-c'
        self.update_SNOPT_options_object()
        self.setup_bounds()
        if self.problem.constrained:
            self.setup_constraints()
        
        if self.options['hot_start_from'] is not None:
            if self.solver_options['Verify level'] is None: # Verify level default is 0
                warnings.warn("Hot-starting might fail with default 'Verify level' (=0).")
            elif self.solver_options['Verify level'] >= 0:
                warnings.warn("Hot-starting might fail with 'Verify level' >= 0.")

    def solve(self):
        append = self.solver_options['append2file']
        check_failure = self.solver_options['continue_on_failure']
        # Assign shorter names to variables and methods
        x0 = self.x0
        x0c0 = x0.copy()

        # ObjRow = 1
        n = self.problem.nx
        m = self.problem.nc
        # nF = self.problem.nc + 1

        obj = self.obj
        grad = self.grad
        G = np.ones((n, ))

        if self.problem.nc > 0:
            con = self.con
            jac = self.jac
            J = np.ones((m, n))
            x0c0 = np.concatenate((x0, np.zeros((m, ))))

        start_time = time.time()
        if self.problem.constrained:
            bl = np.concatenate((self.x_lower, self.c_lower))
            bu = np.concatenate((self.x_upper, self.c_upper))
        else:
            bl = self.x_lower * 1.
            bu = self.x_upper * 1.
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

        inf = self.solver_options['Infinite bound']

        def snoptc_objconFG(mode, nnjac, x, fObj, gObj, fCon, gCon,
                            nState):
            if hasattr(self, "compute_all"):
                if callable(self.compute_all):
                    failure_flag, fObj, fCon, gObj, gCon = self.compute_all(x, check_failure=check_failure)
                    # Note that if the function fails at the initial point then optimization fails
                    if failure_flag:
                        mode = -1
                        warnings.warn("Failed model/derivative evaluation!!! Shortening step in line search ...")

                    if self.problem.nc == 0:
                        fCon = 0.
                        gCon = [0.]
                    else:
                        gCon = gCon.flatten('f')

            else:
                fObj = obj(x)
                gObj = grad(x)

                if self.problem.nc > 0:
                    fCon = con(x)
                    gCon = jac(x).flatten('f')
                else:
                    fCon = 0.
                    gCon = [0.]

            # Flush the summary file
            fsnopt.pyflush(6)
            # Flush the print file
            fsnopt.pyflush(9)

            return mode, fObj, gObj, fCon, gCon

        if self.problem.nc == 0:
            nnJac = 0
            nnCon = 0
            m = 1
            locA = np.ones((n + 1, ))
            locA[0] = 0
            solution = snoptc(snoptc_objconFG,
                              nnObj=nnObj,
                              nnCon=nnCon,
                              nnJac=nnJac,
                              x0=np.append(x0, 0.),
                              J=(np.array([0]), np.array([0]), locA),
                              name='',
                              iObj=0,
                              bl=np.append(bl, -inf),
                              bu=np.append(bu, inf),
                              options=self.SNOPT_options_object,
                              m=m,
                              n=n,
                              append2file=append)
        else:
            solution = snoptc(snoptc_objconFG,
                              nnObj=nnObj,
                              nnCon=nnCon,
                              nnJac=nnJac,
                              x0=x0c0,
                              J=J,
                              name='',
                              iObj=0,
                              bl=bl,
                              bu=bu,
                              options=self.SNOPT_options_object,
                              append2file=append)

        self.total_time = time.time() - start_time

        n = self.problem.nx
        m = self.problem.nc
        self.results = {}
        self.results['x']           = solution.x[:n]
        self.results['c']           = solution.x[n:n+m]
        self.results['x_states']    = solution.states[:n]
        self.results['c_states']    = solution.states[n:n+m]
        self.results['lmult_x']     = solution.rc[:n]
        self.results['lmult_c']     = solution.rc[n:n+m]
        self.results['info']        = solution.info

        self.results['niter']       = solution.iterations
        self.results['n_majiter']   = solution.major_itns
        self.results['nS']          = solution.nS
        self.results['nInf']        = solution.num_inf
        self.results['sInf']        = solution.sum_inf
        self.results['obj']         = solution.objective

        self.run_post_processing()

        return self.results
