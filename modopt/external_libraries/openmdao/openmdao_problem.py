import numpy as np
import scipy as sp
import warnings
from modopt import Problem
from modopt.core.recording_and_hotstart import hot_start, record

class OpenMDAOProblem(Problem):
    def initialize(self, ):
        try:
            from openmdao.api import Problem as OMProblem
        except ImportError:
            raise ImportError("Problem() from 'openmdao' could not be imported")

        self.options.declare('problem_name', default='unnamed_problem', types=str)
        self.options.declare('om_problem', types=OMProblem)

    def setup(self, ):
        # Only for csdl/OpenMDAO problems
        # (Since it is not constructed explicitly using the Problem() class)
        self.problem_name = self.options['problem_name']

        # Set problem dimensions
        om_prob = self.options['om_problem']
        try: # the fixed ordering used for state names in vectors comes from self.hstate_to_res dict
            self.hstate_to_res = om_prob.get_hybrid_state_and_residual_names() # dict is ordered if Python >=3.7
            self.SURF_mode = True if self.hstate_to_res else False
            # if self.SURF_mode:
            #     self.state_names     = list(self.hstate_to_res.keys())
            #     self.res_names       = list(self.hstate_to_res.values())
            #     self.num_states      = len(self.state_names)
            #     self.ny = sum([sim[state_name].size for state_name in self.state_names])
            #     self.nr = self.ny
            #     self.y0 = self._get_hybrid_state_vector()
            #     from array_manager.api import VectorComponentsDict, Vector, MatrixComponentsDict, Matrix, CSCMatrix
            #     self.y_dict = VectorComponentsDict()
            #     for state_name in self.state_names: self.y_dict[state_name] = dict(shape=(sim[state_name].size,))
            #     self.y = Vector(self.y_dict)
            #     self.y.allocate(data=self.y0, setup_views=True)
            #     self.res = Vector(self.y_dict)
            #     self.res.allocate(data=np.ones((self.nr,)), setup_views=True)
            #     self.warm_tol_dict   = {}
            #     self.warm_guess_dict = {}
            #     self.tol_dict0       = dict([(state_name, 1e-12) for state_name in self.state_names])
            #     self.guess_dict0     = dict([(state_name, sim[state_name].flatten()) for state_name in self.state_names])
        except:
            self.SURF_mode = False
            warnings.warn('This version of OpenMDAO wrapper does not support SURF paradigm.')

        # Run model fwd eval once before accessing any values from the driver
        om_prob.run_model()

        dv_values_dict = om_prob.driver.get_design_var_values()
        self.dv_names = list(dv_values_dict.keys())
        dv_values = list(dv_values_dict.values())
        self.dv_sizes  = [value.size for value in dv_values]
        self.x0 = np.concatenate([value.flatten() for value in dv_values]) * 1.
        self.nx = len(self.x0)


        obj_values_dict = om_prob.driver.get_objective_values()
        # modopt only supports a SINGLE objective with OpenMDAO
        self.obj_name = list(obj_values_dict.keys())[0]

        con_values_dict = om_prob.driver.get_constraint_values()
        self.con_names = list(con_values_dict.keys())
        con_values = list(con_values_dict.values())
        self.nc = sum([value.size for value in con_values])
        self.declared_variables = ['dv', 'obj', 'grad']
        self.o_scaler = self.x_scaler = 1.0
        self.c_scaler = None
        if self.nc > 0:
            self.constrained = True
            self.declared_variables += ['con', 'jac']
            self.c_scaler = 1.0

        self.model_evals = 1                    # num of model evals
        self.fail1 = False                      # NotImplemented: failure of functions
        self.fail2 = False                      # NotImplemented: failure of functions or derivatives
        self.warm_x         = self.x0 * 1.      # (x0 - 1.) to keep it different from initial dv values
        # self.totals_dict = om_prob.driver._compute_totals()  # <- no options for return_format, only flat_dict
        # self.totals_dict = om_prob.compute_totals(driver_scaling=True)  # <- returns UNSCALED derivs by default
        self.totals = om_prob.compute_totals(driver_scaling=True, return_format='array')
        self.deriv_evals = 1                    # num of deriv evals
        self.warm_x_deriv   = self.x0 * 1.      # (x0 - 2.)to keep it different from initial dv and warm_x values

#     def check_if_smaller_tol(self, tol_dict):   # only for SURF    
#         if not(self.warm_tol_dict):
#             return True
#         for state_name, warm_tol in self.warm_tol_dict.items():
#             if tol_dict[state_name] < warm_tol:
#                 return True
#         return False

    def setup_derivatives(self):
        pass

    def _setup_scalers(self, ):
        pass
    
    def raise_issues_with_user_setup(self, ):
        pass
    
    def check_if_warm_and_run_model(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        om_prob = self.options['om_problem']
        if not self.SURF_mode:      # for pure RS/FS 
            if (not np.array_equal(self.warm_x, x)) or force_rerun:
                start = 0
                for (dv_name, dv_size) in zip(self.dv_names, self.dv_sizes):
                    om_prob.driver.set_design_var(dv_name, x[start:start+dv_size])
                    start += dv_size

                om_prob.run_model()
                # self.fail1 = sim.run(check_failure=check_failure)
                self.model_evals += 1
                self.warm_x[:] = x
            return
#         else:                       # only for SURF
#             if (not np.array_equal(self.warm_x, x)) or (self.check_if_smaller_tol(tol_dict)) or force_rerun:
#                 sim.update_design_variables(x)
#                 self._update_implicit_guess_and_tol(guess_dict, tol_dict)  # move this to sim (For Mark)
#                 for state_name in self.state_names: 
#                     self.warm_guess_dict[state_name] = guess_dict[state_name]
#                     self.warm_tol_dict[state_name]   = tol_dict[state_name]
#                 self.fail1 = sim.run(check_failure=check_failure)
#                 self.model_evals += 1
#                 self.warm_x[:] = x
#             return
                
    def check_if_warm_and_compute_derivatives(self, x, guess_dict=None, tol_dict=None, 
                                              force_rerun=False, check_failure=False):
        om_prob = self.options['om_problem']
        if not self.SURF_mode:      # for pure RS/FS                     
            if not np.array_equal(self.warm_x_deriv, x) or force_rerun:
                self.check_if_warm_and_run_model(x, force_rerun=force_rerun, check_failure=check_failure)
                # self.totals_dict = om_prob.driver._compute_totals() # <- no options for return_format, only flat_dict
                # self.totals_dict = om_prob.compute_totals(driver_scaling=True) # <- UNSCALED by default
                self.totals = om_prob.compute_totals(driver_scaling=True, return_format='array')
                self.deriv_evals += 1
                # f2 = sim.compute_total_derivatives(check_failure=check_failure)
                # self.fail2 = (self.fail1 and f2)
                self.warm_x_deriv[:] = x
            return
#         else:                       # only for SURF
#             if (not np.array_equal(self.warm_x_deriv, x)) or (self.check_if_smaller_tol(tol_dict)) or force_rerun:
#                 self.check_if_warm_and_run_model(x, guess_dict, tol_dict, force_rerun, check_failure)
#                 f2 = sim.compute_SURF_derivatives(check_failure=check_failure)
#                 self.deriv_evals += 1
#                 self.fail2 = (self.fail1 and f2)
#                 self.warm_x_deriv[:] = x
#             return

    def _setup_bounds(self): # x and c bounds don't include states y and residuals R for SURF
        om_prob = self.options['om_problem']

        # Set design variable bounds
        dv_meta = om_prob.driver._designvars
        x_l = []
        x_u = []
        # Note: no equals for design variables
        for dv_name in dv_meta.keys():
            size = dv_meta[dv_name]['size']
            l = dv_meta[dv_name]['lower']
            u = dv_meta[dv_name]['upper']

            x_l = np.concatenate((x_l, (l * np.ones((size,))).flatten()))
            x_u = np.concatenate((x_u, (u * np.ones((size,))).flatten()))

        self.x_lower = np.where(x_l == -1.0e30, -np.inf, x_l)
        self.x_upper = np.where(x_u == 1.0e30, np.inf, x_u)

        # if self.SURF_mode:
        #     self.x_lower = np.concatenate((self.x_lower, np.full((self.ny,), -np.inf)))
        #     self.x_upper = np.concatenate((self.x_upper, np.full((self.ny,),  np.inf)))

        # Set constraint bounds
        c_meta = om_prob.driver._cons
        c_l = []
        c_u = []
        for con_name in c_meta.keys():
            size = c_meta[con_name]['size']
            e = c_meta[con_name]['equals']
            if e is None:
                l = c_meta[con_name]['lower']
                u = c_meta[con_name]['upper']
            else:
                l = e
                u = e

            c_l = np.concatenate((c_l, (l * np.ones((size,))).flatten()))
            c_u = np.concatenate((c_u, (u * np.ones((size,))).flatten()))

        self.c_lower = np.where(c_l == -1.0e30, -np.inf, c_l)
        self.c_upper = np.where(c_u ==  1.0e30,  np.inf, c_u)

        # if self.SURF_mode:
        #     # For R(x,y) >= 0
        #     self.c_lower = np.concatenate((self.c_lower, np.full((self.ny,), 0.)))
        #     self.c_upper = np.concatenate((self.c_upper, np.full((self.ny,), np.inf)))
        #     # For R(x,y) <= 0
        #     self.c_lower = np.concatenate((self.c_lower, np.full((self.ny,), -np.inf)))
        #     self.c_upper = np.concatenate((self.c_upper, np.full((self.ny,), 0.)))

    # TODO: Add decorators for checking if x is warm and for updating dvs
    @record(['x'], ['obj'])
    @hot_start(['x'], ['obj'])
    def _compute_objective(self, x, guess_dict=None, tol_dict=None, 
                           force_rerun=False, check_failure=False):
        om_prob = self.options['om_problem']
        # print('Computing objective >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        # print('---------Computed objective---------')
        obj_values = list(om_prob.driver.get_objective_values().values())
        return obj_values[0][0]
        # return failure_flag, sim.objective()

    @record(['x'], ['grad'])
    @hot_start(['x'], ['grad'])
    def _compute_objective_gradient(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        om_prob = self.options['om_problem']
        # print('Computing gradient >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        # print('---------Computed gradient---------')
        # totals_dict contains objective gradients as matrices
        # return np.concatenate([self.totals_dict[self.obj_name, dv_name][0] for dv_name in self.dv_names])
        return self.totals[0]

    @record(['x'], ['con'])
    @hot_start(['x'], ['con'])
    def _compute_constraints(self, x, guess_dict=None, tol_dict=None, 
                             force_rerun=False, check_failure=False):
        om_prob = self.options['om_problem']
        # print('Computing constraints >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        # print('---------Computed constraints---------')

        con_values = list(om_prob.driver.get_constraint_values().values())
        # c = np.concatenate([value.flatten() for value in con_values])
        return np.concatenate(con_values)

    @record(['x'], ['jac'])
    @hot_start(['x'], ['jac'])
    def _compute_constraint_jacobian(self, x, guess_dict=None, tol_dict=None, 
                                     force_rerun=False, check_failure=False):
        om_prob = self.options['om_problem']
        # print('Computing Jacobian >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        # print('---------Computed Jacobian---------')
        
        # return np.concatenate([np.concatenate([self.totals_dict[con_name, dv_name] for con_name in self.con_names]) for dv_name in self.dv_names], axis=1)
        return self.totals[1:]

    @record(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    @hot_start(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    def _compute_all(self, x, force_rerun=False, check_failure=False):                              # only for SNOPTC, (NOT meant for SURF)
        om_prob = self.options['om_problem']
        # print('Computing all at once >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, force_rerun=force_rerun, check_failure=check_failure)                 # This is rqd, ow warm derivs skip model evals
        self.check_if_warm_and_compute_derivatives(x, force_rerun=force_rerun, check_failure=check_failure)
        # print('---------Computed all at once---------')

        obj_values = list(om_prob.driver.get_objective_values().values())
        obj = obj_values[0][0]
        
        con_values = list(om_prob.driver.get_constraint_values().values())
        con  = np.concatenate(con_values)

        # grad = np.concatenate([self.totals_dict[self.obj_name, dv_name][0] for dv_name in self.dv_names])
        # jac  = np.concatenate([np.concatenate([self.totals_dict[con_name, dv_name] for con_name in self.con_names]) for dv_name in self.dv_names], axis=1)
        grad = self.totals[0]
        jac  = self.totals[1:]
        
        # TODO: Capture failure_flags from OpenMDAO
        return False, obj, con, grad, jac

#     def _solve_hybrid_residual_equations(self, x, guess_dict, tol_dict, force_rerun=False):
#         sim = self.options['simulator']
#         print('Solving for hybrid states >>>>>>>>>>')
#         self.check_if_warm_and_run_model(x,guess_dict, tol_dict, force_rerun, False)
#         print('---------Computed hybrid states---------')
#         return self._get_hybrid_state_vector()

#     def _get_hybrid_state_vector(self, ):
#         sim = self.options['simulator']
#         return np.concatenate([sim[key].flatten() for key in self.state_names])
    
#     def _get_pF_py(self, ):
#         sim = self.options['simulator']
#         pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
#         return np.concatenate([pFpy[key].flatten() for key in self.state_names])
    
#     def _get_pC_py(self, ):
#         sim = self.options['simulator']
#         pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
#         return np.concatenate([pCpy[key] for key in self.state_names], axis=1)
    
#     def _get_residuals(self,):
#         sim = self.options['simulator']
#         return np.concatenate([sim[key].flatten() for key in self.res_names])
    
#     def _get_objective_gradient(self,):
#         sim = self.options['simulator']
#         if not(self.SURF_mode):
#             return sim.objective_gradient()
#         else:
#             pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
#             return np.concatenate([pFpx]+[pFpy[key].flatten() for key in self.state_names])
    
#     def _get_constraints(self,):
#         sim = self.options['simulator']
#         if not(self.SURF_mode):
#             return sim.constraints()
#         else:
#             c = sim.constraints()
#             r = np.concatenate([sim[key].flatten() for key in self.res_names])
#             return np.concatenate([c, r])
#             # return np.concatenate([c, r, -r])
        
#     def _get_constraint_jacobian(self,):
#         sim = self.options['simulator']
#         if not(self.SURF_mode):
#             return sim.constraint_jacobian()
#         else:
#             pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
#             pCpy_jac = np.concatenate([pCpy[key] for key in self.state_names], axis=1)
#             pRpx_jac = np.concatenate([pRpx[key] for key in self.state_names], axis=0)
#             pRpy_jac = sp.linalg.block_diag(*[pRpy[key] for key in self.state_names])
#             jac = np.block([[pCpx, pCpy_jac], [pRpx_jac, pRpy_jac]])
#             # jac = np.block([[pCpx, pCpy_jac], [pRpx_jac, pRpy_jac], [-pRpx_jac, -pRpy_jac]])
#             return jac

#     # def _compute_surf_adjoint(self, x, lag, guess_dict, tol_dict):   # lag is constraint Lag.mults., only for SURF
#     #     sim = self.options['simulator']

#     #     print('Computing Derivatives for Adjoint >>>>>>>>>>')
#     #     self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict)
#     #     print('---------Computed Derivatives for Adjoint---------')

#     #     print('Computing Adjoint >>>>>>>>>>')
#     #     pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
#     #     # adj = {}
#     #     adj_vec = np.array([])
#     #     for state_name in self.state_names:
#     #         pCpy_SURF = 
#     #         rhs = pFpy[state_name].flatten() + pCpy[state_name].T @ lag
#     #         # adj[state_name] = -np.linalg.solve(pRpy[state_name].T, rhs)
#     #         adj_vec = np.append(adj_vec, -np.linalg.solve(pRpy[state_name].T, rhs))
#     #     print('---------Computed Adjoint---------')
        
#     #     return adj_vec
        
#         # prob_name = self.options['problem_name']
#         # if prob_name == 'WFopt':
#         #     adj = {}
#         #     for state_name in self.state_names:
#         #         adj[state_name] = -np.linalg.solve(pRpy[state_name].T, vec[state_name])
#         # elif prob_name == 'Trim':
#         #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
#         # elif prob_name == 'motor':
#         #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
#         # elif prob_name == 'BEM':
#         #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
#         # elif prob_name == 'Ozone':
#         #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
#         # else:
#         #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')

    
#     def _update_implicit_guess_and_tol(self, guess_dict, tol_dict):
#         sim = self.options['simulator']
#         for state_name in self.state_names:
#             sim.set_implicit_guess_and_tol(state_name, guess_dict[state_name], tol_dict[state_name])

#     def reset_eval_counts(self, ):
#         self.model_evals = 0
#         self.deriv_evals = 0