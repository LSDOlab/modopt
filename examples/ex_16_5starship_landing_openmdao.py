'''Starship 2D trajectory optimization with OpenMDAO'''

import numpy as np
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

min_thrust =  884 * 1000.   # (N)
max_thrust = 2210 * 1000.   # (N)

x_init  = np.array([0, 0, 1000, -80, np.pi/2, 0])
x_final = np.array([0., 0., 0., 0., 0., 0.])

class CostComp(om.ExplicitComponent): # Total of ~170 statements excluding comments.

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
        self.options.declare('min_thrust', default= 884 * 1000., types=float)
        self.options.declare('max_thrust', default=2210 * 1000., types=float)
        self.options.declare('duration', default=16., types=float)
        self.options.declare('nt', default=20, types=int)
        self.options.declare('x_init', types=np.ndarray)    # Initial state
        self.options.declare('x_final', types=np.ndarray)   # Final state

    def setup(self):
        g = self.options['g']
        m = self.options['m']
        L = self.options['L']
        I = (1/12) * m * L**2

        min_gimbal = self.options['min_gimbal']
        max_gimbal = self.options['max_gimbal']
        min_thrust = self.options['min_thrust']
        max_thrust = self.options['max_thrust']

        nt = self.options['nt']

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

# Method 1 - Using OpenMDAO-modOpt interface
def get_problem(nt):
    om_prob = om.Problem(model=StarshipGroup(g=g, m=m, L=L, W=W, nt=nt, duration=duration,
                                             min_gimbal=min_gimbal, max_gimbal=max_gimbal, 
                                             min_thrust=min_thrust, max_thrust=max_thrust,
                                             x_init=x_init, x_final=x_final))
    om_prob.setup()
    prob = OpenMDAOProblem(problem_name=f'starship_{nt}_om', om_problem=om_prob)
    prob.x0 = np.ones(nt*8)
    
    return prob

if __name__ == '__main__':
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
    # prob = OpenMDAOProblem(problem_name=f'starship_{nt}_om', om_problem=om_prob)
    # print(prob._compute_objective(np.arange(nt*8)))         # 9800.0
    # print(prob._compute_constraints(np.arange(nt*8)))       # [  -15.   -19.    -23.     38.5564043   1993.9522483   -1764.75651359   
    #                                                         #    -47.   -51.    -55.  -2081.04096363   995.3398582    1511.53434984
    #                                                         #    -79.   -83.    -87.     69.97044646  -174.99570609   -271.50702618]
    # print(np.linalg.norm(prob._compute_objective_gradient(np.arange(nt*8))))    # 232.44784361228218
    # print(np.linalg.norm(prob._compute_constraint_jacobian(np.arange(nt*8))))   # 5427.787420199876
    # exit()

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
                        [0.00000000e+00, 2.03759954e-16, -6.95543787e+00, -1.92272807e+01, -3.70171262e+01,
                        -5.73251315e+01, -7.70321155e+01, -9.20093998e+01, -1.00655176e+02, -1.02640831e+02,
                        -9.83525602e+01, -8.89790015e+01, -7.59615262e+01, -6.07761859e+01, -4.48738711e+01,
                        -2.96598578e+01, -1.64810765e+01, -6.58179904e+00, -9.71561432e-01, 0.00000000e+00,
                        0.00000000e+00, -8.69429734e+00, -1.53398036e+01, -2.22373069e+01, -2.53850066e+01,
                        -2.46337300e+01, -1.87216053e+01, -1.08072201e+01, -2.48206896e+00, 5.36033849e+00,
                        1.17169484e+01, 1.62718441e+01, 1.89816753e+01, 1.98778935e+01, 1.90175166e+01,
                        1.64734767e+01, 1.23740968e+01, 7.01279701e+00, 1.21445179e+00, 0.00000000e+00,
                        1.00000000e+03, 9.36000000e+02, 8.63192172e+02, 7.82173074e+02, 6.96126754e+02,
                        6.08870469e+02, 5.26122516e+02, 4.50428073e+02, 3.81105091e+02, 3.17983654e+02,
                        2.61262341e+02, 2.10606910e+02, 1.65634822e+02, 1.26062551e+02, 9.17065742e+01,
                        6.24965119e+01, 3.84847192e+01, 1.98487678e+01, 6.88900882e+00, 0.00000000e+00,
                        -8.00000000e+01, -9.10097854e+01, -1.01273872e+02, -1.07557900e+02, -1.09070356e+02,
                        -1.03434942e+02, -9.46180535e+01, -8.66537274e+01, -7.89017958e+01, -7.09016419e+01,
                        -6.33192886e+01, -5.62151096e+01, -4.94653393e+01, -4.29449709e+01, -3.65125779e+01,
                        -3.00147408e+01, -2.32949392e+01, -1.61996988e+01, -8.61126103e+00, 0.00000000e+00,
                        1.57079633e+00, 1.57079633e+00, 1.26700764e+00, 7.31017383e-01, 1.39970913e-01,
                        -2.70159195e-01, -4.28319588e-01, -4.66403998e-01, -4.43718800e-01, -3.80500367e-01,
                        -2.90355810e-01, -1.83663691e-01, -6.88789737e-02, 4.57820842e-02, 1.51359763e-01,
                        2.37157574e-01, 2.89138727e-01, 2.87100964e-01, 2.00698707e-01, 0.00000000e+00,
                        0.00000000e+00, -3.79735853e-01, -6.69987827e-01, -7.38808088e-01, -5.12662636e-01,
                        -1.97700490e-01, -4.76055127e-02, 2.83564967e-02, 7.90230420e-02, 1.12680696e-01,
                        1.33365149e-01, 1.43480896e-01, 1.43326322e-01, 1.31972098e-01, 1.07247264e-01,
                        6.49764412e-02, -2.54720412e-03, -1.08002821e-01, -2.50873384e-01, 0.00000000e+00,
                        5.23318890e-01, 4.00000000e-01, 4.00000000e-01, 4.00000000e-01, 7.63668116e-01,
                        1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 9.43773192e-01,
                        8.83937030e-01, 8.39622310e-01, 8.14118607e-01, 8.09027871e-01, 8.23926904e-01,
                        8.55826765e-01, 8.97815057e-01, 9.32523511e-01, 9.33333248e-01, 4.00000000e-01,
                        3.49065850e-01, 3.49065850e-01, 8.11839128e-02, -2.69738955e-01, -1.95642976e-01,
                        -7.08052720e-02, -3.58117687e-02, -2.38835608e-02, -1.58649445e-02, -1.03304826e-02,
                        -5.39405342e-03, 8.67737450e-05, 6.57369288e-03, 1.44052702e-02, 2.41841512e-02,
                        3.71968927e-02, 5.53912837e-02, 7.22765885e-02, -1.27034906e-01, 2.61331412e-10],
                        rtol=0, atol=1e-6)

    # # Method 2 - Using OpenMDAO directly and its ScipyOptimizeDriver
    # prob = om.Problem(model=StarshipGroup(g=g, m=m, L=L, W=W, nt=nt, duration=duration,
    #                                       min_gimbal=min_gimbal, max_gimbal=max_gimbal, 
    #                                       min_thrust=min_thrust, max_thrust=max_thrust,
    #                                       x_init=x_init, x_final=x_final))
    # prob.setup()

    # prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['tol'] = 1e-9
    # prob.driver.options['disp'] = True

    # prob.setup()

    # start = time.time()
    # prob.run_driver()
    # print('Time:', time.time() - start)
    # print('\tOptimized states:', prob['x'])
    # print('\tOptimized controls:', prob['u'])
    # print('\tOptimized obj:', prob['cost_comp.cost'])

    # solution = np.concatenate((prob['x'].flatten(), prob['u'].flatten()))
    # assert np.allclose(optimizer.results['x'],
    #                    [0.00000000e+00, 2.03759954e-16, -6.95543787e+00, -1.92272807e+01, -3.70171262e+01,
    #                     -5.73251315e+01, -7.70321155e+01, -9.20093998e+01, -1.00655176e+02, -1.02640831e+02,
    #                     -9.83525602e+01, -8.89790015e+01, -7.59615262e+01, -6.07761859e+01, -4.48738711e+01,
    #                     -2.96598578e+01, -1.64810765e+01, -6.58179904e+00, -9.71561432e-01, 0.00000000e+00,
    #                     0.00000000e+00, -8.69429734e+00, -1.53398036e+01, -2.22373069e+01, -2.53850066e+01,
    #                     -2.46337300e+01, -1.87216053e+01, -1.08072201e+01, -2.48206896e+00, 5.36033849e+00,
    #                     1.17169484e+01, 1.62718441e+01, 1.89816753e+01, 1.98778935e+01, 1.90175166e+01,
    #                     1.64734767e+01, 1.23740968e+01, 7.01279701e+00, 1.21445179e+00, 0.00000000e+00,
    #                     1.00000000e+03, 9.36000000e+02, 8.63192172e+02, 7.82173074e+02, 6.96126754e+02,
    #                     6.08870469e+02, 5.26122516e+02, 4.50428073e+02, 3.81105091e+02, 3.17983654e+02,
    #                     2.61262341e+02, 2.10606910e+02, 1.65634822e+02, 1.26062551e+02, 9.17065742e+01,
    #                     6.24965119e+01, 3.84847192e+01, 1.98487678e+01, 6.88900882e+00, 0.00000000e+00,
    #                     -8.00000000e+01, -9.10097854e+01, -1.01273872e+02, -1.07557900e+02, -1.09070356e+02,
    #                     -1.03434942e+02, -9.46180535e+01, -8.66537274e+01, -7.89017958e+01, -7.09016419e+01,
    #                     -6.33192886e+01, -5.62151096e+01, -4.94653393e+01, -4.29449709e+01, -3.65125779e+01,
    #                     -3.00147408e+01, -2.32949392e+01, -1.61996988e+01, -8.61126103e+00, 0.00000000e+00,
    #                     1.57079633e+00, 1.57079633e+00, 1.26700764e+00, 7.31017383e-01, 1.39970913e-01,
    #                     -2.70159195e-01, -4.28319588e-01, -4.66403998e-01, -4.43718800e-01, -3.80500367e-01,
    #                     -2.90355810e-01, -1.83663691e-01, -6.88789737e-02, 4.57820842e-02, 1.51359763e-01,
    #                     2.37157574e-01, 2.89138727e-01, 2.87100964e-01, 2.00698707e-01, 0.00000000e+00,
    #                     0.00000000e+00, -3.79735853e-01, -6.69987827e-01, -7.38808088e-01, -5.12662636e-01,
    #                     -1.97700490e-01, -4.76055127e-02, 2.83564967e-02, 7.90230420e-02, 1.12680696e-01,
    #                     1.33365149e-01, 1.43480896e-01, 1.43326322e-01, 1.31972098e-01, 1.07247264e-01,
    #                     6.49764412e-02, -2.54720412e-03, -1.08002821e-01, -2.50873384e-01, 0.00000000e+00,
    #                     5.23318890e-01, 4.00000000e-01, 4.00000000e-01, 4.00000000e-01, 7.63668116e-01,
    #                     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 9.43773192e-01,
    #                     8.83937030e-01, 8.39622310e-01, 8.14118607e-01, 8.09027871e-01, 8.23926904e-01,
    #                     8.55826765e-01, 8.97815057e-01, 9.32523511e-01, 9.33333248e-01, 4.00000000e-01,
    #                     3.49065850e-01, 3.49065850e-01, 8.11839128e-02, -2.69738955e-01, -1.95642976e-01,
    #                     -7.08052720e-02, -3.58117687e-02, -2.38835608e-02, -1.58649445e-02, -1.03304826e-02,
    #                     -5.39405342e-03, 8.67737450e-05, 6.57369288e-03, 1.44052702e-02, 2.41841512e-02,
    #                     3.71968927e-02, 5.53912837e-02, 7.22765885e-02, -1.27034906e-01, 2.61331412e-10],
    #                     rtol=0, atol=1e-6)