'''Starship 2D trajectory optimization with CSDL'''

import numpy as np
from modopt import CSDLAlphaProblem, SLSQP
import time
import csdl_alpha as csdl

def get_problem(nt): # 47 statements excluding comments, naming, and returns.
    
    g = 9.80665 # gravity (m/s^2)
    m = 100000  # mass (kg)
    L = 50      # length (m)
    W = 10      # width (m)
    I = (1/12) * m * L**2   # moment of inertia (kg*m^2)

    min_gimbal = -20 * np.pi / 180  # (rad)
    max_gimbal =  20 * np.pi / 180  # (rad)

    min_thrust =  884 * 1000   # (N)
    max_thrust = 2210 * 1000   # (N)

    dt = 16 / nt # timestep (s)

    x_init  = np.array([0, 0, 1000, -80, np.pi/2, 0])
    x_final = np.array([0., 0., 0., 0., 0., 0.])

    # x[0] = x position (m)
    # x[1] = x velocity (m/s)
    # x[2] = y position (m)
    # x[3] = y velocity (m/s)
    # x[4] = angle (rad)
    # x[5] = angular velocity (rad/s)

    # u[0] = thrust (percent)
    # u[1] = thrust angle (rad)

    # v = [x, u]

    # State variable bounds
    xl = np.full((6, nt), -np.inf)
    xu = np.full((6, nt),  np.inf)

    # Initial condition
    xl[:, 0] = x_init
    xu[:, 0] = x_init
    # Final condition
    xl[:, -1] = x_final
    xu[:, -1] = x_final

    # Control variable bounds
    ul = np.full((2, nt), -np.inf)
    uu = np.full((2, nt),  np.inf)

    # Thrust limits
    ul[0, :] = min_thrust / max_thrust
    uu[0, :] = 1.0
    # TVC gimbal angle limits
    ul[1, :] = min_gimbal
    uu[1, :] = max_gimbal

    rec = csdl.Recorder()
    rec.start()

    # add state variables
    x = csdl.Variable(name = 'x', shape=(6, nt), value=1.0)
    x.set_as_design_variable(lower=xl, upper=xu)

    # add control variables
    u = csdl.Variable(name = 'u', shape=(2, nt), value=1.0)
    u.set_as_design_variable(lower=ul, upper=uu)

    cost = csdl.sum(u[0]**2) + csdl.sum(u[1]**2) + 2*csdl.sum(x[5]**2)
    cost.add_name('cost')
    cost.set_as_objective()

    f = csdl.Variable(name='f', value=np.zeros((6, nt-1)))

    thrust = max_thrust * u[0, :-1] # thrust magnitude (N)
    theta  = x[4, :-1]              # rocket angle (rad)
    beta   = u[1, :-1]              # thrust angle / gimbal (rad)

    # Dynamics: xdot = f(x,u) = [xdot, xdotdot, ydot, ydotdot, thetadot, thetadotdot]
    f = f.set(csdl.slice[0, :], x[1, :-1])
    f = f.set(csdl.slice[1, :], -thrust * csdl.sin(beta + theta) / m)
    f = f.set(csdl.slice[2, :], x[3, :-1])
    f = f.set(csdl.slice[3, :], thrust * csdl.cos(beta + theta) / m - g)
    f = f.set(csdl.slice[4, :], x[5, :-1])
    f = f.set(csdl.slice[5, :], -0.5 * L * thrust * csdl.sin(beta) / I)

    # Dynamics constraint: x[i+1] = x[i] + dt * f(x[i], u[i]) 
    c = x[:, 1:] - x[:, :-1] - f * dt
    c.add_name('dynamics')
    c.set_as_constraint(equals=0.0)

    rec.stop()

    # Create a Simulator object from the Recorder object
    sim = csdl.experimental.JaxSimulator(rec, gpu=False)

    return CSDLAlphaProblem(problem_name=f'starship_{nt}_csdl', simulator=sim)

if __name__ == '__main__':
    # # Test to see if the problem is correctly defined
    # nt = 4
    # prob = get_problem(nt)
    # print(prob._compute_objective(np.arange(nt*8)))         # 9800.0
    # print(prob._compute_constraints(np.arange(nt*8)))       # [  -15.   -19.    -23.     38.5564043   1993.9522483   -1764.75651359   
    #                                                         #    -47.   -51.    -55.  -2081.04096363   995.3398582    1511.53434984
    #                                                         #    -79.   -83.    -87.     69.97044646  -174.99570609   -271.50702618]
    # print(np.linalg.norm(prob._compute_objective_gradient(np.arange(nt*8))))    # 232.44784361228218
    # print(np.linalg.norm(prob._compute_constraint_jacobian(np.arange(nt*8))))   # 5427.787420199876
    # exit()

    # SLSQP
    nt = 20
    print('\tSLSQP \n\t-----')
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