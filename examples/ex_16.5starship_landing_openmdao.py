'''Starship 2D trajectory optimization with OpenMDAO'''

import numpy as np
import scipy.sparse as sp
import time
import openmdao.api as om
from modopt import SLSQP, OpenMDAOProblem

g = 9.80665  # gravity (m/s^2)
m = 100000.  # mass (kg)
L = 50.      # length (m)
W = 10.      # width (m)
I = (1/12) * m * L**2   # moment of inertia (kg*m^2)

duration = 16.   # duration (s)

min_gimbal = -20 * np.pi / 180  # (rad)
max_gimbal =  20 * np.pi / 180  # (rad)

min_thrust =  880 * 1000.   # (N)
max_thrust = 2210 * 1000.   # (N)

x_init = np.array([0, 0, 1000, -80, np.pi/2, 0])
x_final = np.array([0., 0., 0., 0., 0., 0.])

class CostComp(om.ExplicitComponent): # Total of 170 statements excluding comments.

    def initialize(self):
        self.options.declare('nt', types=int)

    def setup(self):
        nt = self.options['nt']

        self.add_input('x', shape=(6,nt))
        self.add_input('u', shape=(2,nt))
        self.add_output('cost')

        self.declare_partials('cost', 'x', rows=np.zeros(nt), cols=np.arange(5*nt, 6*nt))
        self.declare_partials('cost', 'u')

    def compute(self, inputs, outputs):
        x = inputs['x']
        u = inputs['u']

        outputs['cost'] = np.sum(u[0,:]**2) + np.sum(u[1,:]**2) + 2 * np.sum(x[5,:]**2)
    
    def compute_partials(self, inputs, partials):
        x = inputs['x']
        u = inputs['u']

        partials['cost', 'x'] = 4 * x[5,:]
        partials['cost', 'u'] = 2 * u

class DynamicsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nt', types=int)
        self.options.declare('max_thrust', types=float)
        self.options.declare('g', types=float)
        self.options.declare('m', types=float)
        self.options.declare('L', types=float)
        self.options.declare('I', types=float)

    def setup(self):
        nt = self.options['nt']

        self.add_input('x', shape=(6,nt))
        self.add_input('u', shape=(2,nt))
        self.add_output('f', shape=(6,nt-1))

        rows = np.arange(5*(nt-1))
        cols = np.concatenate([np.arange(nt, 2*nt-1), np.arange(4*nt,5*nt-1), np.arange(3*nt,4*nt-1), np.arange(4*nt,5*nt-1), np.arange(5*nt,6*nt-1)])
        val  = np.ones(5*(nt-1))
        self.declare_partials('f', 'x', rows=rows, cols=cols, val=val)

        rows = np.concatenate([np.arange(nt-1, 2*(nt-1)), np.arange(3*(nt-1), 4*(nt-1)), np.arange(5*(nt-1), 6*(nt-1))])
        rows = np.concatenate([rows, rows])
        cols1 = np.concatenate([np.arange(0, nt-1), np.arange(0,nt-1), np.arange(0,nt-1)])
        cols2 = np.concatenate([np.arange(nt, 2*nt-1), np.arange(nt, 2*nt-1), np.arange(nt, 2*nt-1)])
        cols = np.concatenate([cols1, cols2])
        self.declare_partials('f', 'u', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        nt = self.options['nt']

        max_thrust = self.options['max_thrust']
        g = self.options['g']
        m = self.options['m']
        L = self.options['L']
        I = self.options['I']

        x = inputs['x']
        u = inputs['u']

        f = np.zeros((6, nt-1))
        thrust = max_thrust * u[0, :-1] # thrust magnitude (N)
        theta  = x[4, :-1]              # rocket angle (rad)
        beta   = u[1, :-1]              # thrust angle / gimbal (rad)

        # Dynamics: xdot = f(x,u) = [xdot, xdotdot, ydot, ydotdot, thetadot, thetadotdot]
        f[0, :] = x[1, :-1]
        f[1, :] = -thrust * np.sin(beta + theta) / m
        f[2, :] = x[3, :-1]
        f[3, :] = thrust * np.cos(beta + theta) / m - g
        f[4, :] = x[5, :-1]
        f[5, :] = -0.5 * L * thrust * np.sin(beta) / I

        outputs['f'] = f

    def compute_partials(self, inputs, partials):
        nt = self.options['nt']

        max_thrust = self.options['max_thrust']
        m = self.options['m']
        L = self.options['L']
        I = self.options['I']

        x = inputs['x']
        u = inputs['u']

        thrust = max_thrust * u[0, :-1] # thrust magnitude (N)
        theta  = x[4, :-1]              # rocket angle (rad)
        beta   = u[1, :-1]              # thrust angle / gimbal (rad)

        partials['f', 'x'][nt-1 : 2*(nt-1)]     = -thrust * np.cos(beta + theta) / m
        partials['f', 'x'][3*(nt-1) : 4*(nt-1)] = -thrust * np.sin(beta + theta) / m

        partials['f', 'u'][0: (nt-1)]           = -max_thrust * np.sin(beta + theta) / m
        partials['f', 'u'][1*(nt-1) : 2*(nt-1)] =  max_thrust * np.cos(beta + theta) / m
        partials['f', 'u'][2*(nt-1) : 3*(nt-1)] = -0.5 * L * max_thrust * np.sin(beta) / I

        partials['f', 'u'][3*(nt-1) : 4*(nt-1)] = -thrust * np.cos(beta + theta) / m
        partials['f', 'u'][4*(nt-1) : 5*(nt-1)] = -thrust * np.sin(beta + theta) / m
        partials['f', 'u'][5*(nt-1) : 6*(nt-1)] = -0.5 * L * thrust * np.cos(beta) / I

class ConstraintsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nt', types=int)
        self.options.declare('duration', types=float)

    def setup(self):
        nt = self.options['nt']
        dt = self.options['duration'] / nt

        self.add_input('x', shape=(6,nt))
        self.add_input('f', shape=(6,nt-1))
        self.add_output('constraints', shape=(6,nt-1))

        rows    = np.arange(6*(nt-1))
        rows    = np.concatenate([rows, rows])
        x_cols  = np.arange(6*nt).reshape((6, nt))
        cols    = np.concatenate([x_cols[:, 1:].flatten(), x_cols[:, :-1].flatten()])
        val     = np.concatenate([np.ones(6*(nt-1)), -np.ones(6*(nt-1))])
        self.declare_partials('constraints', 'x', rows=rows, cols=cols, val=val)

        rows =  np.arange(6*(nt-1))
        cols =  np.arange(6*(nt-1))
        val  = -dt * np.ones(6*(nt-1))
        self.declare_partials('constraints', 'f', rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        nt = self.options['nt']
        duration = self.options['duration']
        dt = duration / nt

        x = inputs['x']
        f = inputs['f']

        outputs['constraints'] = x[:, 1:] - x[:, :-1] - f * dt


class StarshipGroup(om.Group):

    def initialize(self):
        self.options.declare('g', default=9.80665, types=float)
        self.options.declare('m', default=100000., types=float)
        self.options.declare('L', default=50., types=float)
        self.options.declare('W', default=10., types=float)
        self.options.declare('min_gimbal', default=-20 * np.pi / 180, types=float)
        self.options.declare('max_gimbal', default= 20 * np.pi / 180, types=float)
        self.options.declare('min_thrust', default= 880 * 1000., types=float)
        self.options.declare('max_thrust', default=2210 * 1000., types=float)
        self.options.declare('duration', default=16., types=float)
        self.options.declare('nt', default=20, types=int)
        self.options.declare('x_init', types=np.ndarray)    # Initial state
        self.options.declare('x_final', types=np.ndarray)   # Final state

    def setup(self):
        g = self.options['g']
        m = self.options['m']
        L = self.options['L']
        W = self.options['W']

        I = (1/12) * m * L**2

        min_gimbal = self.options['min_gimbal']
        max_gimbal = self.options['max_gimbal']
        min_thrust = self.options['min_thrust']
        max_thrust = self.options['max_thrust']

        nt = self.options['nt']
        dt = 16 / nt

        xl = np.full((6, nt), -np.inf)
        xu = np.full((6, nt),  np.inf)

        xl[:, 0] = self.options['x_init']
        xu[:, 0] = self.options['x_init']
        xl[:, -1] = self.options['x_final']
        xu[:, -1] = self.options['x_final']

        ul = np.full((2, nt), -np.inf)
        uu = np.full((2, nt),  np.inf)

        ul[0, :] = min_thrust / max_thrust
        uu[0, :] = 1.0
        ul[1, :] = min_gimbal
        uu[1, :] = max_gimbal

        comp = DynamicsComp(nt=nt, max_thrust=max_thrust, g=g, m=m, L=L, I=I)
        self.add_subsystem('dynamics_comp', comp, promotes_inputs=['x', 'u'])

        comp = ConstraintsComp(nt=nt, duration=duration)
        self.add_subsystem('constraints_comp', comp, promotes_inputs=['x'])

        comp = CostComp(nt=nt,)
        self.add_subsystem('cost_comp', comp, promotes_inputs=['x', 'u'])

        self.connect('dynamics_comp.f', 'constraints_comp.f')

        self.add_design_var('x', lower=xl, upper=xu)
        self.add_design_var('u', lower=ul, upper=uu)
        self.add_objective('cost_comp.cost')
        self.add_constraint('constraints_comp.constraints', equals=0.)

# # Test to see if the problem is correctly defined
# nt = 4
# om_prob = om.Problem(model=StarshipGroup(g=g, m=m, L=L, W=W, nt=nt, duration=duration,
#                                          min_gimbal=min_gimbal, max_gimbal=max_gimbal, 
#                                          min_thrust=min_thrust, max_thrust=max_thrust,
#                                          x_init=x_init, x_final=x_final))
# om_prob.setup()
# om_prob.check_partials(compact_print=True)
# # om_prob.set_val('x', np.arange(6*nt).reshape((6, nt)))
# # om_prob.set_val('u', np.arange(6*nt, 8*nt).reshape((2, nt)))
# # print(prob.x0)
# prob = OpenMDAOProblem(problem_name=f'Starship {nt} timesteps OpenMDAO', om_problem=om_prob)
# print(prob._compute_objective(np.arange(nt*8)))         # 9800.0
# print(prob._compute_constraints(np.arange(nt*8)))       # [  -15.   -19.    -23.     38.5564043   1993.9522483   -1764.75651359   
#                                                         #    -47.   -51.    -55.  -2081.04096363   995.3398582    1511.53434984
#                                                         #    -79.   -83.    -87.     69.97044646  -174.99570609   -271.50702618]
# print(np.linalg.norm(prob._compute_objective_gradient(np.arange(nt*8))))    # 232.44784361228218
# print(np.linalg.norm(prob._compute_constraint_jacobian(np.arange(nt*8))))   # 5427.787420199876
# exit()

# Method 1 - Using OpenMDAO-modOpt interface
def get_problem(nt):
    om_prob = om.Problem(model=StarshipGroup(g=g, m=m, L=L, W=W, nt=nt, duration=duration,
                                             min_gimbal=min_gimbal, max_gimbal=max_gimbal, 
                                             min_thrust=min_thrust, max_thrust=max_thrust,
                                             x_init=x_init, x_final=x_final))
    om_prob.setup()
    prob = OpenMDAOProblem(problem_name=f'Starship {nt} timesteps OpenMDAO', om_problem=om_prob)
    prob.x0 = np.ones(nt*8)
    return prob

# SLSQP
print('\tSLSQP \n\t-----')
nt =20
optimizer = SLSQP(get_problem(nt), solver_options={'maxiter': 1000, 'ftol': 1e-9})
start_time = time.time()
optimizer.solve()
opt_time = time.time() - start_time
success = optimizer.results['success']

print('\tTime:', opt_time)
print('\tSuccess:', success)
print('\tOptimized vars:', optimizer.results['x'])
print('\tOptimized obj:', optimizer.results['fun'])

optimizer.print_results()

v = optimizer.results['x']
x = v[:nt*6].reshape((6, nt)) 
u = v[nt*6:].reshape((2, nt))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x[0], label='x')
plt.plot(x[1], label='xdot')
plt.plot(x[2], label='y')
plt.plot(x[3], label='ydot')
plt.plot(x[4], label='theta')
plt.plot(x[5], label='thetadot')
plt.legend()
plt.show()

plt.figure()
plt.plot(u[0], label='thrust (percent)')
plt.plot(u[1], label='gimbal (rad)')
plt.legend()
plt.show()

assert np.allclose(optimizer.results['x'],
                   [0.00000000e+00, -1.14981167e-16, -6.95601904e+00, -1.92043869e+01,
                    -3.69467385e+01, -5.72041701e+01, -7.68762633e+01, -9.18344882e+01,
                    -1.00474022e+02, -1.02464419e+02, -9.81901512e+01, -8.88349269e+01,
                    -7.58386118e+01, -6.06761211e+01, -4.47973328e+01, -2.96063775e+01,
                    -1.64487202e+01, -6.56680867e+00, -9.68113862e-01, 0.00000000e+00,
                    0.00000000e+00, -8.69502380e+00, -1.53104599e+01, -2.21779395e+01,
                    -2.53217895e+01, -2.45901165e+01, -1.86977810e+01, -1.07994179e+01,
                    -2.48799537e+00, 5.34283450e+00, 1.16940304e+01, 1.62453938e+01,
                    1.89531134e+01, 1.98484854e+01, 1.89886941e+01, 1.64470716e+01,
                    1.23523894e+01, 6.99836850e+00, 1.21014233e+00, 0.00000000e+00,
                    1.00000000e+03, 9.36000000e+02, 8.63191960e+02, 7.82181407e+02,
                    6.96133777e+02, 6.08849118e+02, 5.26067836e+02, 4.50345671e+02,
                    3.81001376e+02, 3.17864482e+02, 2.61132291e+02, 2.10475608e+02,
                    1.65510527e+02, 1.25951986e+02, 9.16146746e+01, 6.24261656e+01,
                    3.84366059e+01, 1.98213206e+01, 6.87840791e+00, 0.00000000e+00,
                    -8.00000000e+01, -9.10100499e+01, -1.01263192e+02, -1.07559537e+02,
                    -1.09105825e+02, -1.03476602e+02, -9.46527057e+01, -8.66803691e+01,
                    -7.89211175e+01, -7.09152387e+01, -6.33208533e+01, -5.62063523e+01,
                    -4.94481758e+01, -4.29216392e+01, -3.64856364e+01, -2.99869496e+01,
                    -2.32691065e+01, -1.61786409e+01, -8.59800988e+00, 0.00000000e+00,
                    1.57079633e+00, 1.57079633e+00, 1.26698226e+00, 7.32017302e-01,
                    1.41712972e-01, -2.68916666e-01, -4.27309112e-01, -4.65532270e-01,
                    -4.42990402e-01, -3.79929318e-01, -2.89945718e-01, -1.83410931e-01,
                    -6.87765502e-02, 4.57478865e-02, 1.51208726e-01, 2.36914749e-01,
                    2.88834676e-01, 2.86777467e-01, 2.00432413e-01, 0.00000000e+00,
                    0.00000000e+00, -3.79767582e-01, -6.68706199e-01, -7.37880413e-01,
                    -5.13287047e-01, -1.97990557e-01, -4.77789481e-02, 2.81773345e-02,
                    7.88263550e-02, 1.12479500e-01, 1.33168484e-01, 1.43292976e-01,
                    1.43155546e-01, 1.31826050e-01, 1.07132529e-01, 6.48999088e-02,
                    -2.57151162e-03, -1.07931318e-01, -2.50540516e-01, 0.00000000e+00,
                    5.23362617e-01, 3.98190045e-01, 3.98190045e-01, 3.98190045e-01,
                    7.63257503e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                    1.00000000e+00, 9.44285918e-01, 8.84437355e-01, 8.40068008e-01,
                    8.14463832e-01, 8.09229686e-01, 8.23950353e-01, 8.55648178e-01,
                    8.97421853e-01, 9.31908952e-01, 9.32567862e-01, 3.98190045e-01,
                    3.49065850e-01, 3.49065850e-01, 8.19741320e-02, -2.69089579e-01,
                    -1.95960087e-01, -7.08603834e-02, -3.58090677e-02, -2.38752982e-02,
                    -1.58628188e-02, -1.03271354e-02, -5.39566195e-03, 7.71086315e-05,
                    6.55659561e-03, 1.43834373e-02, 2.41616021e-02, 3.71758714e-02,
                    5.53651794e-02, 7.21917959e-02, -1.26970126e-01, 8.19444515e-09],
                    rtol=0, atol=1e-6)

# # Method 2 - Using OpenMDAO directly and its ScipyOptimizeDriver
prob = om.Problem(model=StarshipGroup(g=g, m=m, L=L, W=W, nt=nt, duration=duration,
                                      min_gimbal=min_gimbal, max_gimbal=max_gimbal, 
                                      min_thrust=min_thrust, max_thrust=max_thrust,
                                      x_init=x_init, x_final=x_final))
prob.setup()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

prob.setup()

start = time.time()
prob.run_driver()
print('Time:', time.time() - start)
print('\tOptimized states:', prob['x'])
print('\tOptimized controls:', prob['u'])
print('\tOptimized obj:', prob['cost_comp.cost'])

assert np.allclose(prob['x'].flatten(),
                   [0.00000000e+00, -1.14981167e-16, -6.95601904e+00, -1.92043869e+01,
                    -3.69467385e+01, -5.72041701e+01, -7.68762633e+01, -9.18344882e+01,
                    -1.00474022e+02, -1.02464419e+02, -9.81901512e+01, -8.88349269e+01,
                    -7.58386118e+01, -6.06761211e+01, -4.47973328e+01, -2.96063775e+01,
                    -1.64487202e+01, -6.56680867e+00, -9.68113862e-01, 0.00000000e+00,
                    0.00000000e+00, -8.69502380e+00, -1.53104599e+01, -2.21779395e+01,
                    -2.53217895e+01, -2.45901165e+01, -1.86977810e+01, -1.07994179e+01,
                    -2.48799537e+00, 5.34283450e+00, 1.16940304e+01, 1.62453938e+01,
                    1.89531134e+01, 1.98484854e+01, 1.89886941e+01, 1.64470716e+01,
                    1.23523894e+01, 6.99836850e+00, 1.21014233e+00, 0.00000000e+00,
                    1.00000000e+03, 9.36000000e+02, 8.63191960e+02, 7.82181407e+02,
                    6.96133777e+02, 6.08849118e+02, 5.26067836e+02, 4.50345671e+02,
                    3.81001376e+02, 3.17864482e+02, 2.61132291e+02, 2.10475608e+02,
                    1.65510527e+02, 1.25951986e+02, 9.16146746e+01, 6.24261656e+01,
                    3.84366059e+01, 1.98213206e+01, 6.87840791e+00, 0.00000000e+00,
                    -8.00000000e+01, -9.10100499e+01, -1.01263192e+02, -1.07559537e+02,
                    -1.09105825e+02, -1.03476602e+02, -9.46527057e+01, -8.66803691e+01,
                    -7.89211175e+01, -7.09152387e+01, -6.33208533e+01, -5.62063523e+01,
                    -4.94481758e+01, -4.29216392e+01, -3.64856364e+01, -2.99869496e+01,
                    -2.32691065e+01, -1.61786409e+01, -8.59800988e+00, 0.00000000e+00,
                    1.57079633e+00, 1.57079633e+00, 1.26698226e+00, 7.32017302e-01,
                    1.41712972e-01, -2.68916666e-01, -4.27309112e-01, -4.65532270e-01,
                    -4.42990402e-01, -3.79929318e-01, -2.89945718e-01, -1.83410931e-01,
                    -6.87765502e-02, 4.57478865e-02, 1.51208726e-01, 2.36914749e-01,
                    2.88834676e-01, 2.86777467e-01, 2.00432413e-01, 0.00000000e+00,
                    0.00000000e+00, -3.79767582e-01, -6.68706199e-01, -7.37880413e-01,
                    -5.13287047e-01, -1.97990557e-01, -4.77789481e-02, 2.81773345e-02,
                    7.88263550e-02, 1.12479500e-01, 1.33168484e-01, 1.43292976e-01,
                    1.43155546e-01, 1.31826050e-01, 1.07132529e-01, 6.48999088e-02,
                    -2.57151162e-03, -1.07931318e-01, -2.50540516e-01, 0.00000000e+00],
                    rtol=0, atol=1e-6)

assert np.allclose(prob['u'].flatten(),
                    [5.23362617e-01, 3.98190045e-01, 3.98190045e-01, 3.98190045e-01,
                    7.63257503e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                    1.00000000e+00, 9.44285918e-01, 8.84437355e-01, 8.40068008e-01,
                    8.14463832e-01, 8.09229686e-01, 8.23950353e-01, 8.55648178e-01,
                    8.97421853e-01, 9.31908952e-01, 9.32567862e-01, 3.98190045e-01,
                    3.49065850e-01, 3.49065850e-01, 8.19741320e-02, -2.69089579e-01,
                    -1.95960087e-01, -7.08603834e-02, -3.58090677e-02, -2.38752982e-02,
                    -1.58628188e-02, -1.03271354e-02, -5.39566195e-03, 7.71086315e-05,
                    6.55659561e-03, 1.43834373e-02, 2.41616021e-02, 3.71758714e-02,
                    5.53651794e-02, 7.21917959e-02, -1.26970126e-01, 8.19444515e-09],
                    rtol=0, atol=1e-6)