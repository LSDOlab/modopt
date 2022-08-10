import numpy as np
from optimize import snoptc
from optimize.solvers          import snopt7_python as fsnopt
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

    def _getPenaltyParam(self, iw, rw):
        """
        Retrieves the full penalty parameter vector from the work arrays.
        """
        nnCon = iw[23 - 1]
        lxPen = iw[304 - 1] - 1
        xPen = rw[lxPen : lxPen + nnCon]
        return xPen

    def _snstop(self, ktcond, mjrprtlvl, minimize, m, maxs, n, nb, nncon0, nncon, nnobj0, nnobj, ns, 
                itn, nmajor, nminor, nswap, ninfe, sinfe, condzhz, iobj, scaleobj, objadd, fobj, fmerit, penparm, step,
                primalinf, dualinf, maxvi, maxvirel, hs, nej, nlocj, locj, indj, jcol, negcon, scales, bl, bu, fx, fcon, gcon, gobj, 
                ycon, pi, rc, rg, x, cu, iu, ru, cw, iw, rw):
        print('Inside snstop')
#                 iAbort,
# &     KTcond, mjrPrtlvl, minimize,
# &     m, maxS, n, nb, nnCon0, nnCon, nnObj0, nnObj, nS,
# &     itn, nMajor, nMinor, nSwap, nInfE, sInfE,
# &     condZHZ, iObj, scaleObj, objAdd,
# &     fObj, fMerit, penParm, step,
# &     primalInf, dualInf, maxVi, maxViRel, hs,
# &     neJ, nlocJ, locJ, indJ, Jcol, negCon,
# &     scales, bl, bu, Fx, fCon, gCon, gObj,
# &     yCon, pi, rc, rg, x,
# &     cu, lencu, iu, leniu, ru, lenru,
# &     cw, lencw, iw, leniw, rw, lenrw )
        # fmt: on
        """
        This routine is called every major iteration in SNOPT, after solving QP but before line search
        We use it to determine the correct major iteration counting, and save some parameters in the history file.
        If 'snSTOP function handle' is set to a function handle, then the callback is performed at the end of this function.

        returning with iabort != 0 will terminate SNOPT immediately
        """
        iterDict = {
            "isMajor": True,
            "nMajor": nmajor,
            "nMinor": nminor,
        }
        for saveVar in self.options['optvars2save']:
            if saveVar == "merit":
                iterDict[saveVar] = fmerit
            elif saveVar == "feas":
                iterDict[saveVar] = primalinf
            elif saveVar == "opt":
                iterDict[saveVar] = dualinf
            elif saveVar == "penalty":
                penParam = self._getPenaltyParam(iw, rw)
                iterDict[saveVar] = penParam
            # elif saveVar == "Hessian":
            #     H = self._getHessian(iw, rw)
            #     iterDict[saveVar] = H
            elif saveVar == "step":
                iterDict[saveVar] = step
            elif saveVar == "condZHZ":
                iterDict[saveVar] = condzhz
            elif saveVar == "slacks":
                iterDict[saveVar] = x[n:]
            elif saveVar == "lag_mult":
                iterDict[saveVar] = ycon
        
        if self.recorder:
            save_dict = {}
            for var_name in self.recorder.dash_instance.vars['optimizer']['var_names']:
                save_dict[var_name] = iterDict[var_name]
            self.recorder.record(save_dict, 'optimizer')

        # if self.storeHistory:
        #     currX = x[:n]  # only the first n component is x, the rest are the slacks
        #     if nmajor == 0:
        #         callCounter = 0
        #     else:
        #         xuser_vec = self.optProb._mapXtoUser(currX)
        #         callCounter = self.hist._searchCallCounter(xuser_vec)
        #     if callCounter is not None:
        #         self.hist.write(callCounter, iterDict)
        #         # this adds funcs etc. to the iterDict by fetching it from the history file
        #         iterDict = self.hist.read(callCounter)
        #         # update funcs with any additional entries that may be added
        #         if "funcs" in self.cache.keys():
        #             iterDict["funcs"].update(self.cache["funcs"])

        # perform callback if requested
        snstop_handle = self.options['snstop_function_handle']
        if snstop_handle is not None:
            if len(self.options['optvars2save']) == 0:
                raise KeyError("snstop_function_handle must be used with a nonempty list 'optvars2save'")
            iabort = snstop_handle(iterDict)
            # if no return, assume everything went fine
            if iabort is None:
                iabort = 0
        else:
            iabort = 0
        return iabort

    def solve(self):
        append=self.options['append2file']
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

        def snoptc_objconFG(mode, nnjac, x, fObj, gObj, fCon, gCon, nState):
            if nState>=2:
                return
            if hasattr(self, "compute_all"):
                if callable(self.compute_all):
                    failure_flag, fObj, fCon, gObj, gCon = self.compute_all(x)
                    # Note that if the function fails at the initial point then optimization fails
                    if failure_flag:
                        mode = -1
                        print('Failed model/derivative evaluation!!!')

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
            result = snoptc(snoptc_objconFG,
                            self._snstop,
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
                            n=n,
                            append2file=append)
        else:
            result = snoptc(snoptc_objconFG,
                            self._snstop,
                            nnObj=nnObj,
                            nnCon=nnCon,
                            nnJac=nnJac,
                            x0=x0c0,
                            J=J,
                            name=self.problem_name,
                            iObj=0,
                            bl=bl,
                            bu=bu,
                            options=self.SNOPT_options_object,
                            append2file=append)

        end_time = time.time()
        self.total_time = end_time - start_time

        self.snopt_output = result
