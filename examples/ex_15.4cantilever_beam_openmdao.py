'''Cantilever beam optimization with OpenMDAO'''

import numpy as np
import scipy.sparse as sp
import time
import openmdao.api as om
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from modopt import SLSQP, OpenMDAOProblem

E0, L0, b0, vol0, F0 = 1., 1., 0.1, 0.01, -1.

class VolumeComp(om.ExplicitComponent): # Total of 179 statements excluding comments.

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('b', default=1.)
        self.options.declare('L')

    def setup(self):
        num_elements = self.options['num_elements']
        b = self.options['b']
        L = self.options['L']
        L0 = L / num_elements

        self.add_input('h', shape=num_elements)
        self.add_output('volume')

        self.declare_partials('volume', 'h', val=b * L0)

    def compute(self, inputs, outputs):
        L0 = self.options['L'] / self.options['num_elements']

        outputs['volume'] = np.sum(inputs['h'] * self.options['b'] * L0)

class MomentOfInertiaComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('b')

    def setup(self):
        num_elements = self.options['num_elements']

        self.add_input('h', shape=num_elements)
        self.add_output('I', shape=num_elements)

    def setup_partials(self):
        rows = cols = np.arange(self.options['num_elements'])
        self.declare_partials('I', 'h', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['I'] = 1./12. * self.options['b'] * inputs['h'] ** 3

    def compute_partials(self, inputs, partials):
        partials['I', 'h'] = 1./4. * self.options['b'] * inputs['h'] ** 2

class LocalStiffnessMatrixComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('E')
        self.options.declare('L')

    def setup(self):
        num_elements = self.options['num_elements']
        E = self.options['E']
        L = self.options['L']

        self.add_input('I', shape=num_elements)
        self.add_output('K_local', shape=(num_elements, 4, 4))

        L0 = L / num_elements
        coeffs = np.empty((4, 4))
        coeffs[0, :] = [12, 6 * L0, -12, 6 * L0]
        coeffs[1, :] = [6 * L0, 4 * L0 ** 2, -6 * L0, 2 * L0 ** 2]
        coeffs[2, :] = [-12, -6 * L0, 12, -6 * L0]
        coeffs[3, :] = [6 * L0, 2 * L0 ** 2, -6 * L0, 4 * L0 ** 2]
        coeffs *= E / L0 ** 3

        self.mtx = mtx = np.zeros((num_elements, 4, 4, num_elements))
        for ind in range(num_elements):
            self.mtx[ind, :, :, ind] = coeffs

        self.declare_partials('K_local', 'I',
            val=self.mtx.reshape(16 * num_elements, num_elements))

    def compute(self, inputs, outputs):
        outputs['K_local'] = 0
        for ind in range(self.options['num_elements']):
            outputs['K_local'][ind, :, :] = self.mtx[ind, :, :, ind] * inputs['I'][ind]

class StatesComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('K_local', shape=(num_elements, 4, 4))
        self.add_output('d', shape=size)

        cols = np.arange(16*num_elements)
        rows = np.repeat(np.arange(4), 4)
        rows = np.tile(rows, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        self.declare_partials('d', 'K_local', rows=rows, cols=cols)
        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])

        self.K = self.assemble_CSC_K(inputs)
        residuals['d'] = self.K.dot(outputs['d'])  - force_vector

    def solve_nonlinear(self, inputs, outputs):
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])

        self.K = self.assemble_CSC_K(inputs)
        self.lu = splu(self.K)

        outputs['d'] = self.lu.solve(force_vector)

    def linearize(self, inputs, outputs, jacobian):
        num_elements = self.options['num_elements']

        self.K = self.assemble_CSC_K(inputs)
        self.lu = splu(self.K)

        i_elem = np.tile(np.arange(4), 4)
        i_d = np.tile(i_elem, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        jacobian['d', 'K_local'] = outputs['d'][i_d]

        jacobian['d', 'd'] = self.K.toarray()

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['d'] = self.lu.solve(d_residuals['d'])
        else:
            d_residuals['d'] = self.lu.solve(d_outputs['d'])

    def assemble_CSC_K(self, inputs):
        """
        Assemble the stiffness matrix in sparse CSC format.

        Returns
        -------
        ndarray
            Stiffness matrix as dense ndarray.
        """
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_entry = num_elements * 12 + 4
        ndim = num_entry + 4

        data = np.zeros((ndim, ), dtype=inputs._get_data().dtype)
        cols = np.empty((ndim, ))
        rows = np.empty((ndim, ))

        # First element.
        data[:16] = inputs['K_local'][0, :, :].flat
        cols[:16] = np.tile(np.arange(4), 4)
        rows[:16] = np.repeat(np.arange(4), 4)

        j = 16
        for ind in range(1, num_elements):
            ind1 = 2 * ind
            K = inputs['K_local'][ind, :, :]

            # NW quadrant gets summed with previous connected element.
            data[j-6:j-4] += K[0, :2]
            data[j-2:j] += K[1, :2]

            # NE quadrant
            data[j:j+4] = K[:2, 2:].flat
            rows[j:j+4] = np.array([ind1, ind1, ind1 + 1, ind1 + 1])
            cols[j:j+4] = np.array([ind1 + 2, ind1 + 3, ind1 + 2, ind1 + 3])

            # SE and SW quadrants together
            data[j+4:j+12] = K[2:, :].flat
            rows[j+4:j+12] = np.repeat(np.arange(ind1 + 2, ind1 + 4), 4)
            cols[j+4:j+12] = np.tile(np.arange(ind1, ind1 + 4), 2)

            j += 12

        data[-4:] = 1.0
        rows[-4] = 2 * num_nodes
        rows[-3] = 2 * num_nodes + 1
        rows[-2] = 0.0
        rows[-1] = 1.0
        cols[-4] = 0.0
        cols[-3] = 1.0
        cols[-2] = 2 * num_nodes
        cols[-1] = 2 * num_nodes + 1

        n_K = 2 * num_nodes + 2
        return coo_matrix((data, (rows, cols)), shape=(n_K, n_K)).tocsc()
    
class ComplianceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_nodes = self.options['num_elements'] + 1

        self.add_input('displacements', shape=2 * num_nodes)
        self.add_output('compliance')

    def setup_partials(self):
        num_nodes = self.options['num_elements'] + 1
        force_vector = self.options['force_vector']
        self.declare_partials('compliance', 'displacements',
                              val=force_vector.reshape((1, 2 * num_nodes)))

    def compute(self, inputs, outputs):
        outputs['compliance'] = np.dot(self.options['force_vector'], inputs['displacements'])

class BeamGroup(om.Group):

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('volume')
        self.options.declare('num_elements', int)
        self.options.declare('tip_force', types=float, default=-1.)

    def setup(self):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        volume = self.options['volume']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        tip_force = self.options['tip_force']

        force_vector = np.zeros(2 * num_nodes)
        force_vector[-2] = tip_force

        I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
        self.add_subsystem('I_comp', I_comp, promotes_inputs=['h'])

        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        comp = StatesComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('states_comp', comp)

        comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('compliance_comp', comp)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp, promotes_inputs=['h'])

        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect('local_stiffness_matrix_comp.K_local', 'states_comp.K_local')
        self.connect('states_comp.d', 'compliance_comp.displacements',
                     src_indices=np.arange(2 *num_nodes))

        self.add_design_var('h', lower=1e-2, upper=10.)
        self.add_objective('compliance_comp.compliance')
        self.add_constraint('volume_comp.volume', equals=volume)

# Method 1 - Using OpenMDAO-modOpt interface
n_el = 50
om_prob = om.Problem(model=BeamGroup(E=E0, L=L0, b=b0, volume=vol0, num_elements=n_el, tip_force=F0))
om_prob.setup()

prob = OpenMDAOProblem(problem_name=f'Cantilever beam {n_el} elements OpenMDAO', om_problem=om_prob)

# SLSQP
print('\tSLSQP \n\t-----')
optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1e-9})
start_time = time.time()
optimizer.solve()
opt_time = time.time() - start_time
success = optimizer.results['success']
optimizer.print_results()

print('\tTime:', opt_time)
print('\tSuccess:', success)
print('\tOptimized vars:', optimizer.results['x'])
print('\tOptimized obj:', optimizer.results['fun'])

assert np.allclose(optimizer.results['x'],
                    [0.14915754,  0.14764328,  0.14611321,  0.14456715,  0.14300421,  0.14142417,
                    0.13982611,  0.13820976,  0.13657406,  0.13491866,  0.13324268,  0.13154528,
                    0.12982575,  0.12808305,  0.12631658,  0.12452477,  0.12270701,  0.12086183,
                    0.11898809,  0.11708424,  0.11514904,  0.11318072,  0.11117762,  0.10913764,
                    0.10705891,  0.10493903,  0.10277539,  0.10056526,  0.09830546,  0.09599246,
                    0.09362243,  0.09119084,  0.08869265,  0.08612198,  0.08347229,  0.08073573,
                    0.07790323,  0.07496382,  0.07190453,  0.06870925,  0.0653583,   0.06182632,
                    0.05808044,  0.05407658,  0.04975295,  0.0450185,   0.03972912,  0.03363155,
                    0.02620192,  0.01610863], rtol=0, atol=1e-5)


# # Method 2 - Using OpenMDAO directly and its ScipyOptimizeDriver
# prob = om.Problem(model=BeamGroup(E=E0, L=L0, b=b0, volume=vol0, num_elements=n_el, tip_force=F0))

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-9
# prob.driver.options['disp'] = True

# prob.setup()

# start = time.time()
# prob.run_driver()
# print('Time:', time.time() - start)
# print('\tOptimized vars:', prob['h'])
# print('\tOptimized obj:', prob['compliance_comp.compliance'])

# assert np.allclose(prob['h'],
#                     [0.14915754,  0.14764328,  0.14611321,  0.14456715,  0.14300421,  0.14142417,
#                     0.13982611,  0.13820976,  0.13657406,  0.13491866,  0.13324268,  0.13154528,
#                     0.12982575,  0.12808305,  0.12631658,  0.12452477,  0.12270701,  0.12086183,
#                     0.11898809,  0.11708424,  0.11514904,  0.11318072,  0.11117762,  0.10913764,
#                     0.10705891,  0.10493903,  0.10277539,  0.10056526,  0.09830546,  0.09599246,
#                     0.09362243,  0.09119084,  0.08869265,  0.08612198,  0.08347229,  0.08073573,
#                     0.07790323,  0.07496382,  0.07190453,  0.06870925,  0.0653583,   0.06182632,
#                     0.05808044,  0.05407658,  0.04975295,  0.0450185,   0.03972912,  0.03363155,
#                     0.02620192,  0.01610863], rtol=0, atol=1e-5)