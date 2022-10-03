from array_manager.api import VectorComponentsDict, MatrixComponentsDict, Vector, Matrix, BlockMatrix
from array_manager.api import DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix

from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name

import numpy as np

from copy import deepcopy


class Problem(object):
    def __init__(self, **kwargs):

        self.options = OptionsDictionary()

        self.problem_name = 'unnamed_problem'
        self.options.declare('jac_format',
                             default='dense',
                             values=('dense', 'coo', 'csr', 'csc'))
        self.options.declare('hess_format',
                             default='dense',
                             values=('dense', 'coo', 'csr', 'csc'))

        self.x0 = None
        self.nx = 0
        self.nc = 0
        # TODO: Fix this
        self.obj = {}
        self.constrained = False
        self.second_order = False

        ###############################
        # Only for the SURF algorithm #
        ###############################

        self.y0 = None
        self.ny = 0
        self.nr = 0
        self.implicit_constrained = False

        ###########################

        # Numpy vectors for bounds
        self.x_lower = None
        self.x_upper = None
        self.c_lower = None
        self.c_upper = None

        # Abstract vectors for bounds
        self.x_l = None
        self.x_u = None
        self.c_l = None
        self.c_u = None

        self.initialize()
        self.options.update(kwargs)

        # Needed only if using the second method
        self.design_variables_dict = VectorComponentsDict()
        self.pF_px_dict = VectorComponentsDict()
        self.constraints_dict = VectorComponentsDict()

        ###############################
        # Only for the SURF algorithm #
        ###############################

        self.state_variables_dict = VectorComponentsDict()
        self.pF_py_dict = VectorComponentsDict()
        self.residuals_dict = VectorComponentsDict()

        ###########################

        self._setup()

    def __str__(self):
        """
        Print the details of the optimization problem.
        """
        name = self.problem_name
        obj  = self.obj
        dvs  = self.x
        x_l  = self.x_l; x_u  = self.x_u
        cons = self.con
        c_l  = self.c_l; c_u  = self.c_u

        # Print title : optimization problem name
        title   = f'Optimization problem : {name}'
        output  = f'\n\n{title}\n'
        output += '='*100
        
        # PROBLEM OVERVIEW BEGINS
        # >>>>>>>>>>>>>>>>>>>>>>>
        subtitle1 = 'Problem Overview :'
        output  = f'\n\n\t{subtitle1}\n'
        output += '-'*100
        
        # Print objective name
        if len(obj) > 1:
            raise TypeError(f'More than one objective defined for the optimization problem: {", ".join(obj)}.')
        output += f'\n\tObjective: {list(obj.keys())[0]}'
        
        # Print design variables list with their dimensions
        dv_list = ''
        for dv_name, dv in dvs.dict_.items():
            dv_list += f'{dv_name}{dv.shape}, '
        dv_list = dv_list[:-2]
        output += f'\n\tDesign variables:{dv_list}'
        
        #  Print constraints and their dimensions
        con_list = ''
        for con_name, con in cons.dict_.items():
            con_list += f'{con_name}{con.shape}, '
        con_list = con_list[:-2]
        output += f'\n\tConstraints: {con_list}'

        # <<<<<<<<<<<<<<<<<<<<<
        # PROBLEM OVERVIEW ENDS

        # PROBLEM DETAILS BEGINS
        # >>>>>>>>>>>>>>>>>>>>>>>
        subtitle2 = 'Problem Details :'
        output  = f'\n\n\n\t{subtitle2}\n'
        output += '-'*100
        
        # Print objective details
        output += f'\n\tObjectives:\n'
        header = "\t{0} | {1} | {2} ".format(
                                            pad_name('Index', 5),
                                            pad_name('Name', 12),
                                            pad_name('Value', 12),
                                        )
        output += header
        obj_template = "\t{idx:>{l_idx}} | {name:^{l_name}} | {value:<.6e} \n"
        for i, obj_name in enumerate(obj.keys()):
            obj_value = obj[obj_name]
            obj_name = obj_name[:12] if (len(obj_name)>12) else obj_name
            output += obj_template.format(idx=i, l_idx=5, name=obj_name, l_name=12, value=obj_value)
        
        # Print design variable details
        output += f'\n\n\tDesign Variables:'
        header = "\t{0} | {1} | {2} ".format(
                                            pad_name('Index', 8),
                                            pad_name('Name', 10),
                                            pad_name('Lower Limit', 10),
                                            pad_name('Value', 10),
                                            pad_name('Upper Limit', 10),
                                            )
        output += header
        for obj_name, obj_value in obj.items():
            output += obj_template.format(idx=i, l_idx=5, 
                                        )
        # for dv_name, dv in dvs.dict_.items():
        #     for i, x in enumerate(dv.flatten()):
        #         dv_info = .format()
        #         output += dv_info
        
        # #  Print constraint details
        # output += f'\n\n\tConstraints:'
        header = "{0} | {1} | {2} | {3} | {4} "\
                            .format(
                                pad_name('Derivative type', 8),
                                pad_name('Calc norm', 10),
                                pad_name('FD norm', 10),
                                pad_name('Abs error norm', 10),
                                pad_name('Rel error norm', 10),
                            )
        # for con_name, con in cons.dict_.items():
        #     for i, x in enumerate(con.flatten()):
        #         dv_info = .format()
        #         output += dv_info

        # # <<<<<<<<<<<<<<<<<<<<<
        # # PROBLEM DETAILS ENDS




    def _setup(self):
        # user setup() for the problem (call add_design_variables())
        self.setup()

        # CSDLProblem() overrides this method in Problem()
        self._setup_bounds()
        self._setup_gradient_vector()
        self._setup_jacobian_dict()
        self._setup_hessian_dict()

        # user setup() for the problem derivatives
        self.setup_derivatives()
        self._setup_vectors()
        self._setup_matrices()

        # When problem is not defined as CSDLProblem()
        if self.x0 is None:
            # array_manger puts np.zeros as the initial guess if no initial guess is provided
            self.x0 = self.x.get_data() * self.x.design_variables_dict.scaler

    # user defined (call add_design_variables() and add_state_variables() inside)
    def setup(self):
        pass

    def objective(self, x):
        self.x.set_data(x/self.x_scaler)
        self.compute_objective(self.x, self.obj)
        # print('obj', self.obj)
        return self.obj[0] * self.obj_scaler

    def objective_gradient(self, x):
        self.x.set_data(x/self.x_scaler)
        self.compute_objective_gradient(self.x, self.pF_px)
        # print('grad', self.pF_px.get_data())
        return self.pF_px.get_data() * self.obj_scaler / self.x_scaler

    def objective_hessian(self, x):
        self.x.set_data(x/self.x_scaler)
        self.compute_objective_hessian(self.x, self.p2F_pxx)
        self.hess.update_bottom_up()
        return self.hess.get_std_array() * self.obj_scaler / (np.outer(self.x_scaler, self.x_scaler))

    def constraints(self, x):
        self.x.set_data(x/self.x_scaler)
        self.compute_constraints(self.x, self.con)
        # print('con', self.con.get_data())
        return self.con.get_data() * self.c_scaler

    def constraint_jacobian(self, x):
        self.x.set_data(x)
        self.compute_constraint_jacobian(self.x, self.pC_px)
        self.jac.update_bottom_up()
        # print('jac', self.jac.get_std_array())
        return self.jac.get_std_array() * np.outer(self.c_scaler, 1/self.x_scaler)

    # Overridden in CSDLProblem()
    def _setup_bounds(self):
        self.x_scaler = self.design_variables_dict.scaler
        self.x_lower  = self.design_variables_dict.lower * self.x_scaler
        self.x_upper  = self.design_variables_dict.upper * self.x_scaler
        
        self.c_scaler = self.constraints_dict.scaler
        self.c_lower  = self.constraints_dict.lower * self.c_scaler
        self.c_upper  = self.constraints_dict.upper * self.c_scaler

        # Abstract vectors for bounds
        self.x_l = Vector(self.design_variables_dict)
        self.x_l.allocate(data=self.design_variables_dict.lower, setup_views=True)
        self.x_u = Vector(self.design_variables_dict)
        self.x_u.allocate(data=self.design_variables_dict.upper, setup_views=True)
        self.c_l = Vector(self.constraints_dict)
        self.c_l.allocate(data=self.constraints_dict.lower, setup_views=True)
        self.c_u = Vector(self.constraints_dict)
        self.c_u.allocate(data=self.constraints_dict.upper, setup_views=True)

    # user defined #(call declare_gradients() and declare_jacobians() inside,
    # can later include declare_hessians())
    def setup_derivatives(self, ):
        pass

    def _setup_gradient_vector(self):
        # self.pF_px_dict = deepcopy(self.design_variables_dict)
        # self.pF_px_dict.vals = np.zeros((self.nx, ))

        self.pF_px = Vector(self.design_variables_dict)
        self.pF_px.allocate(data=np.zeros((self.nx, )),
                            setup_views=True)

        if self.implicit_constrained:
            # self.pF_py_dict = deepcopy(self.state_variables_dict)
            # self.pF_py_dict.vals = np.zeros((self.nx, ))
            self.pF_py = Vector(self.design_variables_dict)
            self.pF_py.allocate(data=np.zeros((self.ny, )),
                                setup_views=True)

    def _setup_jacobian_dict(self):
        if self.constrained:
            self.pC_px_dict = MatrixComponentsDict(
                self.constraints_dict, self.design_variables_dict)

            if self.implicit_constrained:
                self.pC_py_dict = MatrixComponentsDict(
                    self.constraints_dict, self.state_variables_dict)

        if self.implicit_constrained:
            self.pR_px_dict = MatrixComponentsDict(
                self.residuals_dict, self.design_variables_dict)
            self.pR_py_dict = MatrixComponentsDict(
                self.residuals_dict, self.state_variables_dict)

    def _setup_hessian_dict(self):
        self.p2F_pxx_dict = MatrixComponentsDict(
            self.design_variables_dict, self.design_variables_dict)

        if self.constrained:
            self.p2L_pxx_dict = MatrixComponentsDict(
                self.design_variables_dict, self.design_variables_dict)
            if self.implicit_constrained:
                self.p2L_pxy_dict = MatrixComponentsDict(
                    self.design_variables_dict,
                    self.state_variables_dict)
                self.p2L_pyy_dict = MatrixComponentsDict(
                    self.state_variables_dict,
                    self.state_variables_dict)

        elif self.implicit_constrained:
            self.p2F_pxy_dict = MatrixComponentsDict(
                self.design_variables_dict, self.state_variables_dict)
            self.p2F_pyy_dict = MatrixComponentsDict(
                self.state_variables_dict, self.state_variables_dict)

    def _setup_vectors(self):
        self.x = Vector(self.design_variables_dict)
        self.x.allocate(setup_views=True)

        # self.pF_px = Vector(self.pF_px_dict)
        # self.pF_px.allocate(setup_views=True)

        if self.implicit_constrained:
            self.y = Vector(self.state_variables_dict)
            self.y.allocate(setup_views=True)

            self.residuals = Vector(self.residuals_dict)
            self.residuals.allocate(setup_views=True)

            # self.pF_py = Vector(self.pF_py_dict)
            # self.pF_py.allocate(setup_views=True)

        if self.constrained:
            self.con = Vector(self.constraints_dict)
            self.con.allocate(setup_views=True)

        # self.constraint_duals = Vector(self.constraints_dict)
        # self.residual_duals = Vector(self.residuals_dict)

    def _setup_matrices(self):
        self.p2F_pxx = Matrix(self.p2F_pxx_dict, setup_views=True)
        self.p2F_pxx.allocate()
        if self.options['hess_format'] == 'dense':
            self.hess = DenseMatrix(self.p2F_pxx)
        elif self.options['hess_format'] == 'coo':
            self.hess = COOMatrix(self.p2F_pxx)
        elif self.options['hess_format'] == 'csr':
            self.hess = CSRMatrix(self.p2F_pxx)
        else:
            self.hess = CSCMatrix(self.p2F_pxx)

        if self.constrained:
            self.pC_px = Matrix(self.pC_px_dict, setup_views=True)
            self.pC_px.allocate()
            # TODO: add standard matrices for all jac and hess
            if self.options['jac_format'] == 'dense':
                self.jac = DenseMatrix(self.pC_px)
            elif self.options['jac_format'] == 'coo':
                self.jac = COOMatrix(self.pC_px)
            elif self.options['jac_format'] == 'csr':
                self.jac = CSRMatrix(self.pC_px)
            else:
                self.jac = CSCMatrix(self.pC_px)

            self.p2L_pxx = Matrix(self.p2L_pxx_dict, setup_views=True)
            self.p2L_pxx.allocate()

            if self.implicit_constrained:
                self.p2L_pxy = Matrix(self.p2L_pxy_dict,
                                      setup_views=True)
                self.p2L_pxy.allocate()

                self.p2L_pyy = Matrix(self.p2L_pyy_dict,
                                      setup_views=True)
                self.p2L_pyy.allocate()

        elif self.implicit_constrained:
            self.p2F_pxy = Matrix(self.p2F_pxy_dict, setup_views=True)
            self.p2F_pxy.allocate()
            self.p2F_pyy = Matrix(self.p2F_pyy_dict, setup_views=True)
            self.p2F_pyy.allocate()

        if self.implicit_constrained:
            self.pC_py = Matrix(self.pC_py_dict, setup_views=True)
            self.pC_py.allocate()
            self.pR_px = Matrix(self.pR_px_dict, setup_views=True)
            self.pR_px.allocate()
            self.pR_py = Matrix(self.pR_py_dict, setup_views=True)
            self.pR_py.allocate()

    def add_design_variables(self,
                             name=None,
                             shape=(1, ),
                             lower=None,
                             upper=None,
                             equals=None,
                             vals=None,
                             scaler=None):
        # array_manager automatically sets vals = np.zeros(size) if vals is None
        # if vals is None:
        #     vals = np.zeros(shape)

        # Autonaming index starts from 0
        if name is None:
            name = f'dv{len(self.design_variables_dict)}'

        self.design_variables_dict[name] = dict(
            shape=shape,
            lower=lower,
            upper=upper,
            equals=equals,
            vals=vals,
            scaler=scaler,
        )

        self.nx += np.prod(shape)

    def add_objective(self, name='obj',scaler=1.0):
        # Setting objective name and initializing it with key=name and value=1.
        if len(self.obj)>0:
            raise KeyError('Only one objective is allowed for a problem.')

        print(f'Setting objective name as "{name}".')
        self.obj[name] = 1.
        self.obj_scaler = scaler

    def add_constraints(self,
                        name=None,
                        shape=(1, ),
                        lower=None,
                        upper=None,
                        equals=None,
                        scaler=None,):
        # Autonaming index starts from 0
        if name is None:
            name = len(self.constraints_dict)

        self.constraints_dict[name] = dict(
            shape=shape,
            lower=lower,
            upper=upper,
            equals=equals,
            scaler=scaler,
        )

        self.constrained = True
        self.nc += np.prod(shape)

    # def declare_objective_gradient(self, wrt, shape=(1, ), vals=None):
    def declare_objective_gradient(self, wrt, vals=None):

        if wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which gradient of F is declared'
                .format(wrt))

        else:
            # self.pF_px_dict[wrt] = dict(
            #     shape=shape,
            #     vals=vals,
            # )
            self.pF_px[wrt] = vals

    # if not declared, zero for partials and gradients. If declared with no values, method = cs, fd (user should provide the partials if using Problem() class since the functions must be fairly simple in order to use only the Problem() class or must be coupled through advanced frameworks like OpenMDAO/csdl for complex models and the frameworks already implement cs/fd appriximations)

    def declare_constraint_jacobian(self,
                                    of,
                                    wrt,
                                    shape=None,
                                    vals=None,
                                    rows=None,
                                    cols=None,
                                    ind_ptr=None):

        if of not in self.constraints_dict:
            raise Exception(
                'Undeclared constraint {} for which partial is declared'
                .format(of))
        elif wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pC_px_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def declare_objective_hessian(self,
                                  of,
                                  wrt,
                                  shape=None,
                                  vals=None,
                                  rows=None,
                                  cols=None,
                                  ind_ptr=None):

        if wrt not in self.design_variables_dict or of not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which hessian of F is declared'
                .format(wrt))

        else:
            self.p2F_pxx_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def compute_objective(self, dvs, obj):
        """
        Compute the objective function given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
        obj : ???np.ndarray
            Objective function value.

        # Returns
        # -------
        # f : float
        #     Objective function value.
        """
        pass

    def compute_constraints(self, dvs, con):
        """
        Compute the constraint vector given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
        con : array_manager.Vector
            Vector of constraints.

        # Returns
        # -------
        # con : np.ndarray
        #     Vector of constraints.
        """
        pass

    # def compute_objective_gradient(self, dvs, grad):
    # """
    # Compute the objective function gradient given the design variable vector.

    # Parameters
    # ----------
    # dvs : array_manager.Vector
    #     Design variable vector.
    # grad : array_manager.Vector
    #     Gradient vector of the objective function with respect to the design variable vector.

    # Returns
    # -------
    # pF_px : np.ndarray
    #     Gradient vector of the objective function with respect to the design vector.
    # pF_py : np.ndarray
    #     Gradient vector of the objective function with respect to the state vector.

    # """
    # pass

    # def compute_constraint_jacobian(self, dvs, jac):
    # """
    # Compute the constraint Jacobian with respect to the design variable vector.

    # Parameters
    # ----------
    # dvs : array_manager.Vector
    #     Design variable vector.
    # jac : array_manager.Matrix
    #     Jacobian matrix of the constraints with respect to the design vector.

    # Returns
    # -------
    # pC_px : np.ndarray
    #     Jacobian matrix of the constraints with respect to the design vector.
    # pC_py : np.ndarray
    #     Jacobian matrix of the constraints with respect to the state vector.
    # """
    # pass

    def compute_objective_hessian(self, dvs, hess):
        """
        Compute the objective Hessian given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
        hess : array_manager.Matrix
            Hessian matrix of the objective function with respect to the design variable vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the objective function with respect to the design and state vectors.
        """
        pass

    def compute_constraint_hessian(self, x, idx):
        pass

    def compute_lagrangian_hessian(self, dvs, lag_mult, lag_hess):
        """
        Compute the Lagrangian Hessian given the design variable and Lag. mult. vectors.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
        lag_mult : array_manager.Vector
            Lagrange multiplier vector.
        lag_hess : array_manager.Matrix
            Hessian matrix of the Lagrangian with respect to the design variable vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the objective function with respect to the design and state vectors.
        """
        pass

    def compute_objective_hvp(self, x, v):
        pass

    def compute_constraint_hvp(self, x, v):
        pass

    def compute_lagrangian_hvp(self, x, lag_mult, v):
        pass

    ###########################################################################
    # Everything below is applicable only for SURF algorithm for optimization #
    ###########################################################################

    def add_state_variables(self,
                            name,
                            shape=(1, ),
                            lower=None,
                            upper=None,
                            equals=None,
                            vals=None):
        if vals is None:
            vals == np.zeros(shape)
        self.state_variables_dict[name] = dict(shape=shape,
                                               lower=lower,
                                               upper=upper,
                                               equals=equals,
                                               vals=vals)

        self.ny += np.prod(shape)

    def add_residuals(self, name, shape=(1, )):
        self.residuals_dict[name] = dict(shape=shape,
                                         lower=None,
                                         upper=None,
                                         equals=np.zeros(shape))

        self.nr += np.prod(shape)

    def declare_pF_px_gradient(self, wrt, shape=(1, ), vals=None):

        if wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which gradient of F is declared'
                .format(wrt))

        else:
            self.pF_px_dict[wrt] = dict(
                shape=shape,
                vals=vals,
            )

    def declare_pF_py_gradient(self, wrt, shape=(1, ), vals=None):

        if wrt not in self.state_variables_dict:
            raise Exception(
                'Undeclared state variable {} with respect to which gradient of F is declared'
                .format(wrt))

        else:
            self.pF_py_dict[wrt] = dict(
                shape=shape,
                vals=vals,
            )

    def declare_pR_px_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.residuals_dict:
            raise Exception(
                'Undeclared residual {} for which partial is declared'.
                format(of))
        elif wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pR_px_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def declare_pR_py_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.residuals_dict:
            raise Exception(
                'Undeclared residual {} for which partial is declared'.
                format(of))
        elif wrt not in self.state_variables_dict:
            raise Exception(
                'Undeclared state variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pR_py_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def declare_pC_px_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.constraints_dict:
            raise Exception(
                'Undeclared constraint {} for which partial is declared'
                .format(of))
        elif wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pC_px_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def declare_pC_py_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.constraints_dict:
            raise Exception(
                'Undeclared constraint {} for which partial is declared'
                .format(of))
        elif wrt not in self.state_variables_dict:
            raise Exception(
                'Undeclared state variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pC_py_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def _run(self):
        # user evaluate_model() for the problem
        self.evaluate_model()  #(call compute_() and evaluate() inside)

    # # Override for specific problems if bounds are not available in this format, eg. csdl
    # def declare_variable_bounds(self, x_lower, x_upper):
    #     self.x_lower = x_lower
    #     self.x_upper = x_upper

    # # Override for specific problems if bounds are not available in this format, eg. csdl
    # def declare_constraint_bounds(self, c_lower, c_upper):
    #     self.c_lower = c_lower
    #     self.c_upper = c_upper

    def compute_functions(self, x):
        self.x = x
        self.y = self.solve_residual_equations(
            x)  # Note: assumes a single set of residual equations
        self.f_0 = self.evaluate_objective(x, self.y)
        self.c_0 = self.evaluate_constraints(x, self.y)

        return self.f_0, self.c_0

    def compute_direct_vector(self, x):
        if self.x.get_data() != x:
            self.x.set_data(x)
            self.y = self.solve_residual_equations(
                x)  # Note: assumes a single set of residual equations

        else:
            self.y = self.solve_residual_equations(
                x, self.y
            )  # uses the previous approximation of y to warm start the nonlinear solver

        pR_px, self.pR_py = self.evaluate_residual_jacobian(
            x,
            self.y)  # Note: assumes a single set of residual equations
        self.dy_dx_0 = np.linalg.solve(self.pR_py, pR_px)

        return self.dy_dx_0

    def compute_adjoint_vector(self, x, pFun_py):
        if self.x.get_data() != x:
            self.x.set_data(x)
            self.y = self.solve_residual_equations(
                x)  # Note: assumes a single set of residual equations

        else:
            self.y = self.solve_residual_equations(
                x, self.y
            )  # uses the previous approximation of y to warm start the nonlinear solver

        pR_px, self.pR_py = self.evaluate_residual_jacobian(
            x,
            self.y)  # Note: assumes a single set of residual equations
        self.df_dr_0 = np.linalg.solve(-self.pR_py.T, pFun_py).T

        return self.df_dr_0, pR_px

    def compute_objective_gradient(self, x):
        if self.x.get_data() != x:
            self.x.set_data(x)
            self.y = self.solve_residual_equations(
                x)  # Note: assumes a single set of residual equations

        else:
            self.y = self.solve_residual_equations(
                x, self.y
            )  # uses the previous approximation of y to warm start the nonlinear solver
            # Note: assumes a single set of residual equations
        pF_px_0, pF_py_0 = self.evaluate_objective_gradient(
            self, x, self.y)

        df_dr_0, pR_px = self.compute_adjoint_vector(x, pF_py_0)
        self.pF_px_0 = pF_px_0 + np.matmul(df_dr_0, pR_px)

        return self.pF_px_0

    def compute_constraint_jacobian(self, x):
        if self.x.get_data() != x:
            self.x.set_data(x)
            self.y = self.solve_residual_equations(
                x)  # Note: assumes a single set of residual equations

        else:
            self.y = self.solve_residual_equations(
                x, self.y
            )  # uses the previous approximation of y to warm start the nonlinear solver
            # Note: assumes a single set of residual equations
        pC_px_0, pC_py_0 = self.evaluate_constraint_jacobian(
            self, x, self.y)

        if self.nc <= self.nr:
            dc_dr_0, pR_px = self.compute_adjoint_vector(x, pC_py_0)
            self.pC_px_0 = pC_px_0 + np.matmul(dc_dr_0, pR_px)

        else:
            dy_dx = self.compute_direct_vector(x)
            self.pC_px_0 = pC_px_0 - np.matmul(pC_py_0, dy_dx)

        return self.pC_px_0

    def evaluate_objective(self, x, y):
        """
        Evaluate the objective function given the design and state vectors.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        f : float
            Objective function value.
        """
        pass

    def evaluate_constraints(self, x, y):
        """
        Evaluate the constraint vector given the design and state vectors.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        con : np.ndarray
            Vector of constraints.
        """
        pass

    def evaluate_residuals(self, x, y):
        """
        Evaluate the residual vector given the design and state vectors.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        res : np.ndarray
            Vector of residuals.
        """
        pass

    def evaluate_objective_gradient(self, x, y):
        """
        Evaluate the objective function gradient given the design and state vectors.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        pF_px : np.ndarray
            Gradient vector of the objective function with respect to the design vector.
        pF_py : np.ndarray
            Gradient vector of the objective function with respect to the state vector.
            
        """
        pass

    def solve_residual_equations(self, x, y=None, tol=None):
        """
        Solve the implicit functions that define the residuals to compute an inexact state vector with given tolerances.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        tol : np.ndarray
            Tolerance vector.

        Returns
        -------
        inexact_y : np.ndarray
            Inexact state variable vector.
        """
        pass

    def evaluate_constraint_jacobian(self, x, y):
        """
        Evaluate the constraint Jacobian with respect to the state and design variable vectors.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        pC_px : np.ndarray
            Jacobian matrix of the constraints with respect to the design vector.
        pC_py : np.ndarray
            Jacobian matrix of the constraints with respect to the state vector.
        """
        pass

    def evaluate_residual_jacobian(self, x, y):
        """
        Evaluate the residual Jacobian with respect to the state and design variable vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        pR_px : np.ndarray
            Jacobian matrix of the residual functions with respect to the design vector.
        pR_py : np.ndarray
            Jacobian matrix of the residual functions with respect to the state vector.
        """
        pass

    def evaluate_objective_hessian(self, x, y):
        """
        Evaluate the objective Hessian given the design and state vectors.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the objective function with respect to the design and state vectors.
        """
        pass

    # Note: Since Hessians are large, for constrained problems, we compute constraint Hessians as and when required, if available.
    def evaluate_constraint_hessians(self, x, y, idx):
        """
        Evaluate a specific constraint function given the design and state vectors along with the index of the constraint.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        idx : int
            Global constraint index.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the idx-th constraint function with respect to the design and state vectors.
        """
        pass

    def evaluate_lagrangian_hessian(self, x, y, lag_mult):
        """
        Evaluate the Lagrangian Hessian given the design and state vectors along with the Lagrange multiplier vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the Lagrangian function with respect to the design and state vectors.
        """
        hessian = self.evaluate_objective_hessian(x, y)

        # We cannot afford to store all the constraint Hessians for vectorizing the computation of Lagrangian Hessian so 'for loop' is unavoidable.
        for i in range(len(lag_mult)):
            hessian += lag_mult[i] * self.evaluate_constraint_hessian(
                x, y, i)

        return hessian

    def evaluate_adjoint_vector(self, ):
        pass

    def evaluate_lagrangian_hessian(self, x, y, lag_mult):
        """
        Evaluate the Lagrangian Hessian given the design and state vectors along with the Lagrange multiplier vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the Lagrangian function with respect to the design and state vectors.
        """
        hessian = self.evaluate_objective_hessian(x, y)

        # We cannot afford to store all the constraint Hessians for vectorizing the computation of Lagrangian Hessian so a 'for loop' is unavoidable.
        for i in range(len(lag_mult)):
            hessian += lag_mult[i] * self.evaluate_constraint_hessian(
                x, y, i)

        return hessian

    def evaluate_penalty_hessian(self, x, y, rho):
        """
        Evaluate the penalty Hessian given the design and state vectors along with the Lagrange multiplier vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        rho : np.ndarray
            Vector of penalty parameters.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the penalty function with respect to the design and state vectors.
        """
        hessian = self.evaluate_objective_hessian(x, y)

        # We cannot afford to store all the constraint Hessians for vectorizing the computation of Lagrangian Hessian so a 'for loop' is unavoidable.
        for i in range(len(rho)):
            hessian += rho[i] * self.evaluate_constraint_hessian(
                x, y, i)

        return hessian

    def evaluate_augmented_lagrangian_hessian(self, x, y, rho,
                                              lag_mult):
        """
        Evaluate the augmented Lagrangian Hessian given the design and state vectors along with the Lagrange multiplier vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        rho: np.ndarray
            Vector of penalty parameters
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the augmented Lagrangian function with respect to the design and state vectors.
        """
        hessian = self.evaluate_objective_hessian(x, y)

        # We cannot afford to store all the constraint Hessians for vectorizing the computation of Lagrangian Hessian so 'for loop' is unavoidable.
        for i in range(len(lag_mult)):
            hessian += lag_mult[i] * self.evaluate_constraint_hessian(
                x, y, i)

        return hessian

    # With Hessian-vector products, we can also compute products with the augmented Lagrangian Hessian
    def evaluate_hvp(self, x, y, lag_mult, rho, vx, vy):
        """
        Evaluate the Hessian-vector product along the direction vector specified, for given design and state vectors, for a given 
        Hessian (objective, penalty, Lagrangian, or Augmneted Lagrangian) specified by the Lagrange multipliers and/or penalty parameters.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        vx : np.ndarray
            x block of vector to be right-multiplied.
        vy : np.ndarray
            y block of vector to be right-multiplied.
        rho : np.ndarray
            Vector of penalty parameters.
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        wx : np.ndarray
            x block of the Hessian-vector product.
        wy : np.ndarray
            y block of the Hessian-vector product.
        """
        pass