import numpy as np
from utils import get_bspline_mtx


dtype = complex

class FEA(Problem):

    def initialize(self, **kwargs):
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
        self.options.declare('num_elements', default=40, types=int)
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

        # self.K = K = np.zeros((num_elements + 1, num_elements + 1))
        # self.u = np.zeros(num_elements + 1)
        self.heights = np.ones(self.num_elements, dtype=dtype)
        self.forces = self.options['forces']
        self.bsp = np.array(
            get_bspline_mtx(self.num_design_variables, self.num_elements).todense()
        )


    def _set_dvs(self, dvs):
        """
        Set the design variables (bar heights).

        Parameters
        ----------
        dvs : np.ndarray[num_design_variables]
            Design variables vector.
        """
        self.heights = self.bsp.dot(dvs).flatten()

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

    def _evaluate(self, dvs):
        """
        Compute the compliance as a function of the design variables.

        Parameters
        ----------
        dvs : np.ndarray[num_design_variables]
            Design variables vector.

        Returns
        -------
        float
            Compliance.
        """
        self._set_dvs(dvs)
        K = self._compute_K()
        u = self._solve(K)
        c = self._compute_compliance(u)
        return c

    def compute_objective(self, x):

    def compute_constraints(self, x):
        
    def compute_objective_gradient(self, x):

    
    
    def declare_bounds(self, x_lower, x_upper):

    def declare_constraint_bounds(self, c_lower, c_upper):
