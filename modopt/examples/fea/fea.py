import numpy as np
from utils import get_bspline_mtx

from modopt.api import Problem

# dtype = complex
dtype = float


class FEA(Problem):
    def initialize(self):
        """
        Instantiate the FEA object.

        Parameters
        ----------
        num_elements : int
            Number of bar elements.
        num_design_variables : int
            Number of bar thickness design variables. 
            The bar thicknesses are mapped to the elements using a B-spline.
        E : float
            Elastic modulus of the bar elements.
        L : float
            Total length of the bar.
        b : float
            Width of the bar elements.
        """
        self.problem_name = 'fea'
        self.options.declare('num_elements', default=40, types=int)
        self.options.declare('nx', default=10, types=int)
        self.options.declare('E', default=1., types=float)
        self.options.declare('L', default=1., types=float)
        self.options.declare('b', default=1., types=float)
        self.options.declare('forces', types=np.ndarray)

    def setup(self):
        self.num_elements = self.options['num_elements']
        self.num_design_variables = self.options['nx']
        self.E = self.options['E']
        self.L = self.options['L'] / self.num_elements
        self.b = self.options['b']

        self.heights = np.ones(self.num_elements, dtype=dtype)
        self.forces = self.options['forces']
        # self.bsp = np.array(
        #     get_bspline_mtx(self.num_design_variables,
        #                     self.num_elements).todense())
        self.bsp = np.identity(self.num_elements)

        self.add_design_variables('heights',
                                  shape=(self.num_design_variables, ),
                                  lower=0.01 * np.ones(
                                      (self.num_design_variables, )),
                                  vals=.08 * np.ones(
                                      (self.num_design_variables, )))

        self.add_objective('compliance')

        self.add_constraints(
            'average_height',
            upper=np.array([0.5]),
        )

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='heights',)
        self.declare_constraint_jacobian(
            of='average_height',
            wrt='heights',
            shape=(1, self.num_design_variables),
            vals=1 / self.num_design_variables * np.ones(
                (1, self.num_design_variables)),
        )

    def _set_dvs(self, dvs):
        """
        Set the design variables (bar heights).

        Parameters
        ----------
        dvs : np.ndarray[num_design_variables]
            Design variables vector.
        """
        self.heights = dvs

    def _compute_K(self):
        """
        Compute and return the stiffness matrix.

        Returns
        -------
        np.ndarray[num_elements, num_elements]
            Stiffness matrix, with the first, fixed degree of freedom removed.
        """
        num_elements = self.num_elements
        K = np.zeros((num_elements + 1, num_elements + 1), dtype=dtype)
        A = self.b * self.heights

        EA_L = self.E * A / self.L

        arange = np.arange(num_elements)
        K[arange, arange] += EA_L
        K[arange + 1, arange + 1] += EA_L
        K[arange + 1, arange] -= EA_L
        K[arange, arange + 1] -= EA_L

        return K[1:, 1:]

    def _compute_pRph(self, u):
        """
        Compute and return the partial derivatives of R with respect to h.

        Parameters
        ----------
        u : np.ndarray[num_elements]
            Displacement vector with the first, fixed degree of freedom removed.

        Returns
        -------
        np.ndarray[num_elements, num_design_variables]
            Derivatives.
        """
        num_elements = self.num_elements
        pKph = np.zeros(
            (num_elements + 1, num_elements + 1, num_elements),
            dtype=dtype,
        )
        Eb_L = self.E * self.b / self.L

        arange = np.arange(num_elements)
        pKph[arange, arange, arange] += Eb_L
        pKph[arange + 1, arange + 1, arange] += Eb_L
        pKph[arange + 1, arange, arange] -= Eb_L
        pKph[arange, arange + 1, arange] -= Eb_L

        pKph = np.einsum('abi,ij->abj', pKph, self.bsp)[1:, 1:, :]
        pRph = np.einsum('ikj,k->ij', pKph, u)

        return pRph

    def _solve(self, K):
        """
        Compute and return the displacement vector.

        Parameters
        ----------
        K : np.ndarray[num_elements, num_elements]
            Stiffness matrix, with the first, fixed degree of freedom removed.

        Returns
        -------
        np.ndarray[num_elements]
            Displacement vector with the first, fixed degree of freedom removed.

        """
        u = np.linalg.solve(K, self.forces)
        return u

    def _compute_compliance(self, u):
        """
        Compute and return the compliance (dot product of displacements and forces).

        Parameters
        ----------
        u : np.ndarray[num_elements]
            Displacement vector with the first, fixed degree of freedom removed.

        Returns
        -------
        float
            Compliance.
        """
        return np.dot(u, self.forces)

    def _get_forces(self):
        """
        Return a copy of the forces vector.

        Returns
        -------
        np.ndarray[num_elements]
            Copy of the forces vector.
        """
        return np.array(self.forces)

    def compute_objective(self, dvs, obj):
        self._set_dvs(dvs['heights'])
        K = self._compute_K()
        u = self._solve(K)
        c = self._compute_compliance(u)

        print('objective:', c)

        obj['compliance'] = c
        

    def compute_constraints(self, dvs, con):
        con['average_height'] = np.array([np.sum(dvs['heights'] / self.num_design_variables)])

    def solve_residual_equations(self, dvs):
        self._set_dvs(dvs['heights'])
        pRpy = self._compute_K()
        u = self._solve(pRpy)
        return u

    def compute_objective_gradient(self, dvs, grad):
        self._set_dvs(dvs['heights'])
        K = self._compute_K()
        u = self._solve(K)
        pRpx = self._compute_pRph(u)
        pRpy = self._compute_K()
        pFpx = 0.
        pFpy = self._get_forces()

        dfdr = np.linalg.solve(pRpy.T, -pFpy)
        # dfdr = np.linalg.solve(pRpy.T, pFpy)
        grad_aj = pFpx + pRpx.T.dot(dfdr)
        grad_aj = np.real(grad_aj)

        grad['heights'] = grad_aj.reshape((1, self.num_design_variables))

    def evaluate_residual_jacobian(self, dvs, u):
        self._set_dvs(dvs['heights'])
        pRpx = self._compute_pRph(u)
        pRpy = self._compute_K()
        return pRpx, pRpy

    def evaluate_constraint_jacobian(self, dvs, u):
        pCpx = 0.1 * np.ones((1, self.num_design_variables))
        pCpy = np.zeros((1, self.num_elements))
        return pCpx, pCpy
    
    def compute_constraint_jacobian(self, dvs, jac):
        pass
    
    # def compute_constraint_jacobian(self, dvs, jac):
    #     u = self.solve_residual_equations(dvs)

    #     jac['average_height', 'heights'] = 0.1 * np.ones((1, self.num_design_variables))
    #     # The following also works

    #     # pRpx, pRpy = self.evaluate_residual_jacobian(dvs, u)
    #     # pCpx, pCpy = self.evaluate_constraint_jacobian(dvs, u)

    #     # dcdr = np.linalg.solve(pRpy.T, -pCpy.flatten())
    #     # jac_aj = pCpx + pRpx.T.dot(dcdr)
    #     # jac_aj = np.real(jac_aj)
        
    #     # jac['average_height', 'heights'] = jac_aj.reshape((1, self.num_design_variables))
