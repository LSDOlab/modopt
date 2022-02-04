"""
Numerical lifting line theory and optimization model implemented in CSDL
Nicholas Orndorff
1/2/2022
"""
import openmdao.api as om
import csdl
import csdl_om
import numpy as np
import matplotlib.pyplot as plt


class Linear(csdl.Model):
    def initialize(self):
        self.parameters.declare('N', types=int)

    def define(self):
        N = self.parameters['N']
        # vector of odd indices
        m = self.declare_variable('m',
                                  val=(np.arange(1, 2 * N, 2)).reshape(
                                      (N, 1)),
                                  shape=(N, 1))  # np.arange(1,2*N-1,2)
        # declare variables without values
        phi = self.declare_variable('phi', shape=(N, 1))
        aoa = self.declare_variable('aoa')
        #aoa = self.declare_variable('aoa', shape=(N,1))
        aoa = csdl.expand(aoa, (N, 1))
        aoa_zl = self.declare_variable('aoa_zl', shape=(N, 1))
        b = self.declare_variable('b')
        Cla = self.declare_variable('Cla')
        c = self.declare_variable('c')
        #C_of_Phi = self.declare_variable('C_of_Phi', shape=(N,1))
        C_of_Phi = csdl.expand(c, (N, 1))
        # compute mu
        coef = csdl.expand(0.25 * Cla / b, (N, 1))
        mu = C_of_Phi * coef
        # create B matrix from component matrices
        # calculate sin(m*phi) component matrix
        mphi = csdl.transpose(csdl.matmat(m, csdl.transpose(phi)))
        sin_mphi = csdl.sin(mphi)
        # calculate sin(phi) component matrix
        one = self.declare_variable('one', val=1, shape=(N, 1))
        sin_phi = csdl.transpose(
            csdl.matmat(one, csdl.transpose(csdl.sin(phi))))
        # calculate m*mu component matrix
        mmu = csdl.transpose(csdl.matmat(m, csdl.transpose(mu)))
        # calculate secondary matrix term
        sec = mmu + sin_phi
        # calculate entire matrix
        B = sin_mphi * sec  # must be elemet-wise haddemard product
        # create C vector
        # compute middle term
        mid = aoa - aoa_zl
        # create C
        C = mu * mid * csdl.sin(phi)
        # create expression for linear system
        A = self.declare_variable('A', shape=(N, 1))
        y = csdl.matmat(B, A) - C
        # register residual as output
        self.register_output('y', y)


class LiftingLine(csdl.Model):
    def initialize(self):
        self.parameters.declare('N', types=int)

    def define(self):
        N = self.parameters['N']
        # define linear implicit solver
        solve_linear = self.create_implicit_operation(Linear(N=N))
        solve_linear.declare_state('A', residual='y')
        solve_linear.nonlinear_solver = csdl.NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_linear.linear_solver = csdl.ScipyKrylov()
        # declare variables and values
        # create stations
        dPhi = 0.5 * np.pi / N
        p = np.arange(dPhi, (np.pi / 2) + dPhi, dPhi).reshape((N, 1))
        phi = self.declare_variable('phi', val=p, shape=(N, 1))
        #aoa = self.declare_variable('aoa')
        aoa = self.declare_variable('aoa', val=0.08727)
        #aoa = self.declare_variable('aoa', val=0.08727, shape=(N,1))
        aoa_zl = self.declare_variable('aoa_zl',
                                       val=-0.04712,
                                       shape=(N, 1))
        c = self.declare_variable('c')
        C_of_Phi = csdl.expand(c, (N, 1))
        #C_of_Phi = self.declare_variable('C_of_Phi', val=(np.ones(N)*3).reshape(N,1), shape=(N,1))
        b = self.declare_variable('b')
        Cla = self.declare_variable('Cla', val=6.2832)
        # solve linear system
        A = solve_linear(phi, aoa, aoa_zl, b, Cla, c)
        # calculate total lift coefficient
        AR = csdl.expand((b / c), (1, 1))
        Cl = A[0, 0] * csdl.reshape(np.pi * AR, new_shape=(1, 1))
        self.register_output('Cl', Cl)
        self.register_output('AR', AR)
        # calculate total induced drag coefficient
        sumcdi = 0.
        for i in range(2, N + 1):
            n = 2 * (i) - 1
            sumcdi = sumcdi + n * A[i - 1, 0]**2
        delcdi = sumcdi / (A[0, 0]**2)
        coef = (1 + delcdi) / (np.pi * AR)
        CDi = ((A[0, 0] * csdl.reshape(np.pi * AR, new_shape=(1, 1)))**
               2) * coef
        self.register_output('CDi', CDi)
        # calculate lift coefficient distribution
        k = self.declare_variable('k',
                                  val=(np.arange(1, 2 * N, 2)).reshape(
                                      (N, 1)),
                                  shape=(N, 1))  # np.arange(1,2*N-1,2)
        kphi = csdl.transpose(csdl.matmat(k, csdl.transpose(phi)))
        sin_kphi = csdl.sin(kphi)
        one = self.declare_variable('one', val=1, shape=(N, 1))
        oa = csdl.matmat(one, csdl.transpose(A))
        oap = oa * sin_kphi  # element wise haddemard product
        row_sum = csdl.matmat(csdl.transpose(one), csdl.transpose(oap))
        sum_cl = csdl.transpose(row_sum)
        coef = csdl.expand(4 * b, (N, 1))
        cl_dist = (coef * sum_cl) / C_of_Phi
        self.register_output('cl_dist', cl_dist)


# optimization model
class opt(csdl.Model):
    def initialize(self):
        self.parameters.declare('N', types=int)

    def define(self):
        N = self.parameters['N']
        c = 3
        self.create_input('c', c)
        b = 38.3
        self.create_input('b', b)
        self.add(LiftingLine(N=N))
        # calculate lift constraint
        Cl = self.declare_variable('Cl')
        rho = 1.225
        v = 98
        s = b * c
        lift = 0.5 * rho * (v**2) * s * Cl
        self.register_output('lift', lift)
        self.add_constraint('lift', equals=560000.)
        # calculate induced drag
        CDi = self.declare_variable('CDi')
        drag = 0.5 * rho * (v**2) * s * CDi
        self.register_output('drag', drag)
        # calculate thrust and power required
        w = 560000
        tr = w / (Cl / CDi)
        pr = tr * v
        self.register_output('pr', pr)
        #self.add_constraint('pr', upper=500000)
        # design variables
        self.add_design_variable('c')
        self.add_design_variable('b')
        #self.add_constraint('Cl', equals=0.8)
        self.add_constraint('c', lower=0.1)
        #self.add_constraint('b', lower=0.1)
        #self.add_constraint('AR', lower=1)
        #self.add_objective('CDi')
        self.add_objective('drag')


# run optimization problem
N = 20  # define number of vortices
sim = csdl_om.Simulator(opt(N=N), mode='auto')  # mode='rev'
sim.prob.driver = om.ScipyOptimizeDriver()
sim.prob.driver.options['optimizer'] = 'SLSQP'
sim.prob.driver.options['tol'] = 1.0e-10
sim.prob.driver.options['maxiter'] = 100
sim.prob.run_driver()
print('chord: ', sim['c'])
print('span: ', sim['b'])
print('CL: ', sim['Cl'])
print('CDi: ', sim['CDi'])
print('AR: ', sim['AR'])
print('lift: ', sim['lift'])
print('drag: ', sim['drag'])
print('power required: ', sim['pr'])
# run model
"""
N = 20 # define number of vortices
sim = csdl_om.Simulator(LiftingLine(N=N))
sim.run()
print('CL: ', sim['Cl'])
print('CDi: ', sim['CDi'])
print('AR: ', sim['AR'])
"""
# for plotting lift coefficient vs half-span
A = sim['A']
cl_dist = sim['cl_dist']
b = sim['b']
zero = np.zeros((1, 1))
Cl2 = np.concatenate((zero, cl_dist))
dy = (b / 2) / N
ys = np.arange(0, (b / 2), dy).reshape((N, 1))
ys = np.concatenate((ys, [ys[-1, 0] + dy]))
plt.plot(np.flip(ys), Cl2)
