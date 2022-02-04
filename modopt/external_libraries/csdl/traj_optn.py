# csdl flight dynamics model
from ozone.api import ODEProblem, Wrap, NativeSystem
import openmdao.api as om
import csdl
from csdl_om import Simulator
from modopt.csdl_library import CSDLProblem
from modopt.scipy_library import SLSQP
from modopt.optimization_algorithms import SQP
from modopt.snopt_library import SNOPT
import csdl_om
import numpy as np
import matplotlib.pyplot as plt


# STEP 1: ODE Model
class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.parameters['num_nodes']
        # State y. All ODE states must have shape = (n, .. shape of state ...)
        x = self.create_input('x', shape=n)
        z = self.create_input('z', shape=n)
        u = self.create_input('u', shape=n)
        w = self.create_input('w', shape=n)
        o = self.create_input('o', shape=n)  # objective state
        # Parameters are now inputs
        p1 = self.create_input('p1', shape=(n))
        p3 = self.create_input('p3', shape=(n))
        # compute rotor normal velocities
        u1 = 1 * w
        u3 = 1 * u
        m = 3724.  # mass (kg) 3724
        A1 = 58.4  # total lift disk area (m^2) 58.4
        A3 = 7.3  # total cruise disk area (m^2) 7.3
        rho = 1.225
        # rotor surrogate model with momentum theory
        fm1 = ((p1 *
                ((2 * rho * A1)**0.5))**(2 / 3)) * csdl.exp(-0.05 * u1)
        fm3 = ((p3 *
                ((2 * rho * A3)**0.5))**(2 / 3)) * csdl.exp(-0.05 * u3)
        # lift slope
        a0 = 5.733
        # Oswald's efficiency factor
        e = 0.8
        # aspect ratio
        AR = 11.2
        # zero lift drag coefficient
        cd0 = 0.0151
        # wing area (m^2)
        s = 18.1
        # compute angle of attack
        alpha = csdl.arctan(w / u)
        # wing angle of incidence
        aefs = 0.06  # rad
        vel = ((u**2) + (w**2))**0.5
        cl = (alpha + aefs) * a0
        cd = cd0 + (cl**2) / (np.pi * e * AR)
        lift = 0.5 * rho * (vel**2) * s * cl
        drag = 0.5 * rho * (vel**2) * s * cd
        g = 9.81
        fx = 1 * fm3 - lift * csdl.sin(alpha) - drag * csdl.cos(alpha)
        fz = 1 * fm1 + lift * csdl.cos(alpha) - drag * csdl.sin(
            alpha) - m * g
        # compute system of ode's
        dx = 1 * u
        dz = 1 * w
        du = (fx / m)
        dw = (fz / m)
        do = (p1 + p3) / 100000.
        # Register output
        self.register_output('dx', dx)
        self.register_output('dz', dz)
        self.register_output('du', du)
        self.register_output('dw', dw)
        self.register_output('do', do)


# STEP 2: ODEProblem class
class ODEProblemTest(ODEProblem):
    def setup(self):
        # User needs to define setup method
        # Define ODE from Step 2.
        self.ode_system = Wrap(ODESystemModel)
        self.profile_outputs_system = Wrap(ODESystemModel)
        self.add_parameter('p1', dynamic=True, shape=(num, 1))
        self.add_parameter('p3', dynamic=True, shape=(num, 1))
        # State names and timevector correspond to respective upstream CSDL variables
        self.add_state('x', 'dx', initial_condition_name='x_0')
        self.add_state('z', 'dz', initial_condition_name='z_0')
        self.add_state('u', 'du', initial_condition_name='u_0')
        self.add_state('w', 'dw', initial_condition_name='w_0')
        self.add_state('o', 'do', initial_condition_name='o_0')
        self.add_times(step_vector='h')
        # Profile Output Variables
        self.add_profile_output('x', state_name='x')
        self.add_profile_output('z', state_name='z')
        self.add_profile_output('u', state_name='u')
        self.add_profile_output('w', state_name='w')
        self.add_profile_output('o', state_name='o')


# csdl model specification containing the ODE integrator
class RunModel(csdl.Model):
    def define(self):
        dt = 4.0
        # Timestep vector
        h_vec = np.ones(num) * dt
        self.create_input('h', h_vec)
        # Create given inputs
        # Coefficients for field output
        #self.create_input('coefficients', np.ones(num+1)/(num+1))
        # Initial condition for states
        self.create_input('x_0', 0.)
        self.create_input('z_0', 0.)
        self.create_input(
            'u_0',
            0.01)  # so angle of attack can be defined at time zero
        self.create_input('w_0', 0.)
        self.create_input('o_0', 0.)
        p1 = np.zeros(
            (num, 1))  # dynamic parameter defined at every timestep
        p3 = np.zeros((num, 1))
        for t in range(num):
            p1[t] = 800000  #dynamic parameter defined at every timestep
            p3[t] = 40000
        self.create_input('p1', p1)
        self.create_input('p3', p3)
        # Create Model containing integrator
        self.add(ODEProblem.create_solver_model(), 'subgroup', ['*'])
        # Read an output from integrator
        # profile outputs
        p_x = self.declare_variable('x', shape=(num + 1, ))
        p_z = self.declare_variable('z', shape=(num + 1, ))
        p_u = self.declare_variable('u', shape=(num + 1, ))
        p_w = self.declare_variable('w', shape=(num + 1, ))
        p_o = self.declare_variable('o', shape=(num + 1, ))
        obj = 1 * p_o[-1]
        self.register_output('obj', obj)
        #self.add_design_variable('y')
        self.add_design_variable('p1')
        self.add_design_variable('p3')
        final_x = p_x[-1]
        final_z = p_z[-1]
        final_u = p_u[-1]
        final_w = p_w[-1]
        #profile constraints
        constraint_x = 1 * p_x
        constraint_z = 1 * p_z
        constraint_u = 1 * p_u
        constraint_w = 1 * p_w
        # time step constraint
        self.register_output('final_x', final_x)
        self.register_output('final_z', final_z)
        self.register_output('final_u', final_u)
        self.register_output('final_w', final_w)
        self.register_output('constraint_x', constraint_x)
        self.register_output('constraint_z', constraint_z)
        self.register_output('constraint_u', constraint_u)
        self.register_output('constraint_w', constraint_w)
        # final position constraints
        self.add_constraint('final_x', lower=500.0,
                            upper=2500.)  # 100-3000
        self.add_constraint('final_z', lower=300.,
                            upper=1000.)  # 500-1200
        # final velocity constraints
        self.add_constraint(
            'final_u',
            lower=45.3)  # 88 knots (45.2 m/s) cruise velocity
        #self.add_constraint('final_w', equals=0.)
        self.add_constraint('final_w', lower=-0.01, upper=0.01)
        # profile constraints
        self.add_constraint('constraint_x', lower=0.)
        self.add_constraint('constraint_z', lower=0.)
        self.add_constraint('constraint_u',
                            lower=0.)  # positive u keeps aoa defined
        self.add_constraint('constraint_w', lower=0.)
        # power constraints
        self.add_constraint('p1', lower=10., upper=829218.)  # 829218
        self.add_constraint(
            'p3', lower=10.,
            upper=468299.)  # 468299 force issue by raising bottom limit
        self.add_objective('obj')


# ODEProblem_instance
num = 20
# Integration approach: Timeamarching or Checkpointing
approach = 'time-marching'
ODEProblem = ODEProblemTest('RK4',
                            approach,
                            num_times=num,
                            display='default',
                            visualization='end')
# Simulator Object:
sim = csdl_om.Simulator(RunModel(), mode='rev')

# Setup your optimization problem
prob = CSDLProblem(
    problem_name='traj_optn',
    simulator=sim,
)

# Setup your optimizer with the problem
# optimizer = SLSQP(prob, maxiter=20)
# optimizer = SQP(prob, max_itr=20)
optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3)

# Check first derivatives at the initial guess, if needed
optimizer.check_first_derivatives(prob.x0)

# Solve your optimization problem
optimizer.solve()

# Print results of optimization
# (summary_table contains information from each iteration)
optimizer.print_results(summary_table=True)

# sim.prob.driver = om.ScipyOptimizeDriver()
# sim.prob.driver.options['optimizer'] = 'SLSQP'
# sim.prob.driver.options['tol'] = 1.0#e-1
# sim.prob.driver.options['maxiter'] = 150
# sim.prob.run_driver()
#sim.run()
plt.show()
p1 = sim['p1']
p3 = sim['p3']
plt.plot(p3)
#print('x: ', sim['x'])
print('z: ', sim['z'])
print('p1: ', sim['p1'])
print('p3: ', sim['p3'])
#print('u: ', sim['u'])
print('w: ', sim['w'])