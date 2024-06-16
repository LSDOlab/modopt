import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt.utils.options_dictionary import OptionsDictionary


class TrustRegion(object):
    # TODO: Currently checks only Armijo condition

    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        # Truat region radius
        self.options.declare('delta',
                             default=1.,
                             types=float,
                             lower=eps)

        # Armijo parameter
        self.options.declare('eta_a',
                             default=1e-4,
                             types=float,
                             upper=(1 - eps),
                             lower=eps)
        # Good correlation reference parameter
        self.options.declare('eta_ref',
                             default=1e-2,
                             types=float,
                             upper=(1 - eps),
                             lower=eps)

        # Trust region contraction factor
        self.options.declare('gamma_c',
                             default=0.6,
                             types=float,
                             upper=(1.0 - eps),
                             lower=eps)
        # Trust region expansion factor
        self.options.declare('gamma_e',
                             default=2.0,
                             types=float,
                             upper=(1.0 - eps),
                             lower=eps)

        self.options.declare('f', types=type(lambda: None))
        self.options.declare('g', types=type(lambda: None))

        self.initialize()
        self.options.update(kwargs)

    def update(self, x, f0=None, g0=None, H0=None):
        # Note: H0 is needed for formulating subproblems for Newton-based methods, not required for steepest descent

        # TODO: Pass in a merit function object
        delta = self.options['delta']

        eta_a = self.options['eta_a']
        eta_ref = self.options['eta_ref']

        maxiter = self.options['maxiter']

        gamma_c = self.options['gamma_c']
        gamma_e = self.options['gamma_e']

        f = self.options['f']
        g = self.options['g']

        if f0 is None:
            self.f1 = f1 = f(x)
        else:
            self.f1 = f1 = f0 * 1.

        if g0 is None:
            self.g1 = g1 = g(x)
        else:
            self.g1 = g1 = g0 * 1.

        self.itr = 0

        d = self.subproblem_solve()
        norm_d = np.linalg.norm(d)

        slope = np.inner(g1, d)

        f2 = f(x + d)
        rho = (f1 - f2) / slope

        # If there is no sufficient reduction, x remains unchanged, decrease trust region radius, and solve the subproblem with the reduced radius
        while (rho <= eta_a) and (self.itr <= maxiter):
            self.itr += 1

            delta = gamma_c * norm_d

            # Step from solving new subproblem with reduced trust region radius
            d = self.subproblem_solve()
            norm_d = np.linalg.norm(d)

            f2 = f(x + d)
            rho = (f1 - f2) / slope

        # Update to the new design point (TODO: gets updated even when the solution does not satisfy sufficient decrease, if max_it is reached)
        x1 = x + d

        if (rho >= eta_a):
            converged = True
            # Check for good correlation and if positive, increase trust region radius
            if (rho >= eta_ref):
                delta = np.maximum(delta, gamma_e * norm_d)

            # else: delta remains unchanged

        # Note: delta is not returned but updated inside the TrustRegion object

        return x1, f2, converged

    def subbproblem_solve(self):
        pass
