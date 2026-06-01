import numpy as np
from numpy.testing import assert_allclose
import pytest

pytest.importorskip("egobox")

from modopt import Egor, Problem, ProblemLite, optimize


pytestmark = [pytest.mark.interfaces, pytest.mark.egobox]


class FiniteBoundsProblem(Problem):
    def initialize(self):
        self.problem_name = "finite_bounds_problem"

    def setup(self):
        self.add_design_variables(
            "x",
            shape=(1,),
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            vals=np.array([0.8]),
        )
        self.add_objective("f")

    def setup_derivatives(self):
        pass

    def compute_objective(self, dvs, obj):
        x = dvs["x"]
        obj["f"] = (x[0] - 0.25) ** 2


def finite_bounds_lite():
    x0 = np.array([0.8])

    def obj(x):
        return (x[0] - 0.25) ** 2

    return ProblemLite(
        x0, obj=obj, xl=np.array([-1.0]), xu=np.array([1.0]), name="finite_bounds_lite"
    )


class DoubleSidedIneqProblem(Problem):
    def initialize(self):
        self.problem_name = "double_sided_ineq_problem"

    def setup(self):
        self.add_design_variables(
            "x",
            shape=(1,),
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            vals=np.array([0.0]),
        )
        self.add_objective("f")
        self.add_constraints(
            "c", shape=(1,), lower=np.array([0.2]), upper=np.array([0.6])
        )

    def setup_derivatives(self):
        self.declare_constraint_jacobian(of="c", wrt="x", vals=np.array([[1.0]]))

    def compute_objective(self, dvs, obj):
        x = dvs["x"]
        obj["f"] = (x[0] - 0.8) ** 2

    def compute_constraints(self, dvs, cons):
        cons["c"][0] = dvs["x"][0]

    def compute_constraint_jacobian(self, dvs, jac):
        pass


def double_sided_ineq_lite():
    x0 = np.array([0.0])

    def obj(x):
        return (x[0] - 0.8) ** 2

    def con(x):
        return np.array([x[0]])

    def jac(x):
        return np.array([[1.0]])

    return ProblemLite(
        x0,
        obj=obj,
        con=con,
        jac=jac,
        xl=np.array([-1.0]),
        xu=np.array([1.0]),
        cl=np.array([0.2]),
        cu=np.array([0.6]),
        name="double_sided_ineq_lite",
    )


class EqualityConstraintProblem(Problem):
    def initialize(self):
        self.problem_name = "equality_constraint_problem"

    def setup(self):
        self.add_design_variables(
            "x",
            shape=(1,),
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            vals=np.array([0.0]),
        )
        self.add_objective("f")
        self.add_constraints("c", shape=(1,), equals=np.array([0.5]))

    def setup_derivatives(self):
        self.declare_constraint_jacobian(of="c", wrt="x", vals=np.array([[1.0]]))

    def compute_objective(self, dvs, obj):
        obj["f"] = dvs["x"][0] ** 2

    def compute_constraints(self, dvs, cons):
        cons["c"][0] = dvs["x"][0]

    def compute_constraint_jacobian(self, dvs, jac):
        pass


def test_egor_direct_interface():
    probs = [FiniteBoundsProblem(), finite_bounds_lite()]

    for prob in probs:
        optimizer = Egor(
            prob,
            solver_options={"max_iters": 12, "n_doe": 4, "seed": 3},
            turn_off_outputs=True,
        )
        results = optimizer.solve()
        assert results["success"] is True
        assert_allclose(results["x"], [0.25], atol=8e-2)
        assert abs(results["fun"]) <= 1e-2


def test_egor_with_inequality_constraints():
    probs = [DoubleSidedIneqProblem(), double_sided_ineq_lite()]

    for prob in probs:
        optimizer = Egor(
            prob,
            solver_options={"max_iters": 15, "n_doe": 5, "seed": 7},
            turn_off_outputs=True,
        )
        results = optimizer.solve()
        assert results["success"] is True
        assert_allclose(results["x"], [0.6], atol=1e-1)
        assert results["constraints"][0] <= 0.6 + 1e-3
        assert results["constraints"][0] >= 0.2 - 1e-3


def test_optimize_with_egor():
    prob = finite_bounds_lite()
    results = optimize(
        prob,
        solver="Egor",
        solver_options={"max_iters": 12, "n_doe": 4, "seed": 11},
        turn_off_outputs=True,
    )
    assert results["success"] is True
    assert_allclose(results["x"], [0.25], atol=8e-2)


def test_egor_requires_finite_bounds():
    prob = ProblemLite(
        np.array([0.0]), obj=lambda x: x[0] ** 2, name="unbounded_problem"
    )

    with pytest.raises(ValueError) as exc_info:
        Egor(prob, turn_off_outputs=True)

    assert (
        str(exc_info.value)
        == "Egor requires finite lower and upper bounds on all design variables."
    )


def test_egor_supports_equality_constraints():
    optimizer = Egor(
        EqualityConstraintProblem(),
        solver_options={"max_iters": 15, "n_doe": 5, "seed": 13},
        turn_off_outputs=True,
    )
    results = optimizer.solve()
    assert results["success"] is True
    assert_allclose(results["x"], [0.5], atol=1e-1)
    assert abs(results["constraints"][0] - 0.5) <= 1e-1


class MixedBoundIneqProblem(Problem):
    def initialize(self):
        self.problem_name = "mixed_bound_ineq_problem"

    def setup(self):
        self.add_design_variables(
            "x",
            shape=(1,),
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            vals=np.array([0.0]),
        )
        self.add_objective("f")
        self.add_constraints(
            "c", shape=(2,), lower=np.array([-np.inf, 0.0]), upper=np.array([0.2, np.inf])
        )

    def setup_derivatives(self):
        self.declare_constraint_jacobian(of="c", wrt="x", vals=np.array([[1.0], [1.0]]))

    def compute_objective(self, dvs, obj):
        obj["f"] = (dvs["x"][0] - 0.1) ** 2

    def compute_constraints(self, dvs, cons):
        x = dvs["x"][0]
        cons["c"][0] = x  # x <= 0.2
        cons["c"][1] = x  # x >= 0.0

    def compute_constraint_jacobian(self, dvs, jac):
        pass


def test_egor_with_upper_and_lower_ineq_constraints():
    optimizer = Egor(
        MixedBoundIneqProblem(),
        solver_options={"max_iters": 8, "n_doe": 4, "seed": 5},
        turn_off_outputs=True,
    )
    results = optimizer.solve()
    assert results["success"] is True
    # Ensure wrapper accepts both <= and >= forms and returns finite result.
    assert np.isfinite(results["x"]).all()
    assert np.isfinite(results["fun"])
