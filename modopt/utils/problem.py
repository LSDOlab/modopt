from array_manager.api import VectorComponentsDict, MatrixComponentsDict, Vector, Matrix, BlockMatrix

from modopt.utils.options_dictionary import OptionsDictionary

import numpy as np

from copy import deepcopy


class Problem(object):
    def __init__(self, **kwargs):

        self.options = OptionsDictionary()

        self.options.declare('problem_name',
                             default='unnamed_problem',
                             types=str)
        self.options.declare('nx', default=0, types=(int, np.int64))
        self.options.declare('ny', default=0, types=(int, np.int64))
        self.options.declare('nc', default=0, types=(int, np.int64))
        self.options.declare('nr', default=0, types=(int, np.int64))
        self.options.declare('x0',
                             default=None,
                             types=(np.ndarray, float, type(None)))
        self.options.declare('y0',
                             default=None,
                             types=(np.ndarray, float, type(None)))

        self.nx = 0
        self.ny = 0
        self.nc = 0
        self.nr = 0
        self.constrained = False
        self.implicit_constrained = False
        self.second_order = False
        self.x_lower = None
        self.x_upper = None
        self.c_lower = None
        self.c_upper = None

        # self.initialize(**kwargs)
        self.initialize()
        self.options.update(kwargs)

        x0 = self.options['x0']
        y0 = self.options['y0']
        nx = self.options['nx']
        ny = self.options['ny']

        if nx != 0 and x0 is None:
            self.options['x0'] = np.full((nx, ), 0.)

        if ny != 0 and y0 is None:
            self.options['y0'] = np.full((ny, ), 0.)

        # Needed only if using the second method
        self.design_variables_dict = VectorComponentsDict()
        self.pF_px_dict = VectorComponentsDict()

        self.state_variables_dict = VectorComponentsDict()
        self.pF_py_dict = VectorComponentsDict()

        self.constraints_dict = VectorComponentsDict()
        self.residuals_dict = VectorComponentsDict()

    # def evaluate_constraints_and_residuals(self, x):
    #     pass

    # def compute_states(self, x):
    #     pass

    # def compute_adjoint_vector(self, x):
    #     pass

    # def compute_constraint_and_residual_jacobian(self, x):
    #     pass

    # Called before any partial declaration, after all variables and constraints are declared

    def _setup(self):
        # user setup() for the problem
        self.setup()
        self._setup_bounds()
        self._setup_gradient_vector()
        self._setup_jacobian_dict()
        self._setup_hessian_dict()

        # user setup() for the problem
        self.setup_derivatives()
        self._setup_vectors()
        self._setup_matrices()

    def setup(self):
        pass
        # self.declare_variable_bounds()
        # self.declare_constraint_bounds()

    # all compute() and evaluate() after _setup()

    def _run(self):
        # user evaluate_model() for the problem
        self.evaluate_model()  #(call compute_() and evaluate() inside)

    # user defined
    def setup(
        self,
    ):  #(call add_design_variables() and add_state_variables() inside)
        pass

    def _setup_bounds(self):
        self.x_lower = self.design_variables_dict.lower
        self.x_upper = self.design_variables_dict.upper
        self.c_lower = self.constraints_dict.lower
        self.c_upper = self.constraints_dict.upper

        print(self.x_lower)
        print(self.x_upper)
        print(self.c_lower)
        print(self.c_upper)

    # user defined
    def setup_derivatives(
        self,
    ):  #(call declare_gradients() and declare_jacobians() inside, can later include declare_hessians())
        pass

    def _setup_gradient_vector(self):
        # self.pF_px_dict = deepcopy(self.design_variables_dict)
        # self.pF_px_dict.vals = np.zeros((self.options['nx'], ))

        self.pF_px = Vector(self.design_variables_dict)
        self.pF_px.allocate(data=np.zeros((self.options['nx'], )),
                            setup_views=True)

        if self.implicit_constrained:
            # self.pF_py_dict = deepcopy(self.state_variables_dict)
            # self.pF_py_dict.vals = np.zeros((self.options['nx'], ))
            self.pF_py = Vector(self.design_variables_dict)
            self.pF_py.allocate(data=np.zeros((self.options['nx'], )),
                                setup_views=True)

    def _setup_jacobian_dict(self):
        if self.constrained:
            self.pC_px_dict = MatrixComponentsDict(
                self.constraints_dict, self.design_variables_dict)

            if self.implicit_constrained:
                self.pC_py_dict = MatrixComponentsDict(
                    self.constraints_dict, self.state_variables_dict)

        elif self.implicit_constrained:
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

    def _run(self):
        # user evaluate_model() for the problem
        self.evaluate_model()  #(call compute_() and evaluate() inside)

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
            self.constraints = Vector(self.constraints_dict)
            self.constraints.allocate(setup_views=True)

        # self.constraint_duals = Vector(self.constraints_dict)
        # self.residual_duals = Vector(self.residuals_dict)

    def _setup_matrices(self):
        self.p2F_pxx = Matrix(self.p2F_pxx_dict, setup_views=True)
        self.p2F_pxx.allocate()

        if self.constrained:
            self.pC_px = Matrix(self.pC_px_dict, setup_views=True)
            self.pC_px.allocate()

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
                             name,
                             shape=(1, ),
                             lower=None,
                             upper=None,
                             equals=None,
                             vals=None):
        if vals is None:
            vals == np.zeros(shape)
        self.design_variables_dict[name] = dict(
            shape=shape,
            lower=lower,
            upper=upper,
            equals=equals,
            vals=vals,
        )

        self.options['nx'] += np.prod(shape)

    def add_objective(self, name):
        pass

    def add_state_variables(self,
                            name,
                            shape=(1, ),
                            lower=None,
                            upper=None,
                            vals=None):
        if vals is None:
            vals == np.zeros(shape)
        self.state_variables_dict[name] = dict(shape=shape,
                                               lower=lower,
                                               upper=upper,
                                               vals=vals)

        self.options['ny'] += np.prod(shape)

    def add_constraints(self,
                        name,
                        shape=(1, ),
                        lower=None,
                        upper=None,
                        equals=None):
        self.constraints_dict[name] = dict(
            shape=shape,
            lower=lower,
            upper=upper,
            equals=equals,
        )

        self.constrained = True

        self.options['nc'] += np.prod(shape)

    def add_residuals(self, name, shape=(1, )):
        self.residuals_dict[name] = dict(shape=shape,
                                         lower=None,
                                         upper=None,
                                         equals=np.zeros(shape))

        self.options['nr'] += np.prod(shape)

    def declare_objective_gradient(self, wrt, shape=(1, ), vals=None):

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

    def declare_objective_hessian(self,
                                  of,
                                  wrt,
                                  shape=(1, 1),
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

    # if not declared, zero for partials and gradients. If declared with no values, method = cs, fd (user should provide the partials if using Problem() class since the functions must be fairly simple in order to use only the Problem() class or must be coupled through advanced frameworks like OpenMDAO/csdl for complex models and the frameworks already implement cs/fd appriximations)

    def declare_constraint_jacobian(self,
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

    def compute_objective(self, x):
        pass

    def compute_constraints(self, x):
        pass

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
        if self.x != x:
            self.x = x
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
        if self.x != x:
            self.x = x
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
        if self.x != x:
            self.x = x
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
        if self.x != x:
            self.x = x
            self.y = self.solve_residual_equations(
                x)  # Note: assumes a single set of residual equations

        else:
            self.y = self.solve_residual_equations(
                x, self.y
            )  # uses the previous approximation of y to warm start the nonlinear solver
            # Note: assumes a single set of residual equations
        pC_px_0, pC_py_0 = self.evaluate_constraint_jacobian(
            self, x, self.y)

        if self.options['nc'] <= self.options9['nr']:
            dc_dr_0, pR_px = self.compute_adjoint_vector(x, pC_py_0)
            self.pC_px_0 = pC_px_0 + np.matmul(dc_dr_0, pR_px)

        else:
            dy_dx = self.compute_direct_vector(x)
            self.pC_px_0 = pC_px_0 - np.matmul(pC_py_0, dy_dx)

        return self.pC_px_0

    def compute_objective_hessian(self, x):
        pass

    def compute_objective_hvp(self, x, v):
        pass

    def compute_constraint_hessians(self, x):
        pass

    def compute_lagrangian_hessian(self, x, lag_mult, v):
        pass

    def compute_lagrangian_hvp(self, x, lag_mult):
        pass

    def compute_hvp(self, x, coeffs, vx):
        pass

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