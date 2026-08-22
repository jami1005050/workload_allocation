import numpy as np

from greenload.scheduler import mpc


def _tiny_instance(num_loc=3, horizon=5, seed=0):
    rng = np.random.default_rng(seed)
    price = rng.uniform(20, 80, size=(num_loc, horizon))
    water = rng.uniform(0.2, 3.0, size=(num_loc, horizon))
    carbon = rng.uniform(50, 700, size=(num_loc, horizon))
    workload = rng.uniform(0.1, num_loc * 0.5, size=(1, horizon))
    return price, water, carbon, workload


def test_offline_oracle_solves():
    price, water, carbon, workload = _tiny_instance()
    cost, allocation = mpc.offline_oracle(price, water, carbon, workload, l1=60, l2=1500)
    assert np.isfinite(cost)
    assert cost >= 0
    assert (allocation >= -1e-6).all()
    np.testing.assert_allclose(allocation.sum(), workload.sum(), atol=1e-4)


def test_mpc_solver_one_step():
    price, water, carbon, workload = _tiny_instance(horizon=3)
    cost, y = mpc.mpc_solver(
        price,
        water,
        carbon,
        workload,
        historical_cost=(0.0, np.zeros(3), np.zeros(3)),
        l1=60,
        l2=1500,
    )
    assert np.isfinite(cost)
    assert cost >= 0
    assert (y >= -1e-6).all()


def test_evaluate_allocation_matches_manual_computation():
    price, water, carbon, workload = _tiny_instance(num_loc=2, horizon=2)
    allocation = np.ones((2, 2)) * 0.5
    result = mpc.evaluate_allocation(allocation, price, water, carbon, l1=60, l2=1500)
    expected_price = np.sum(price * allocation)
    assert np.isclose(result["price_cost"], expected_price)
    assert (
        result["total_cost"] == result["price_cost"] + result["water_cost"] + result["carbon_cost"]
    )


def _run_perfect_foresight_mpc(num_loc: int, horizon: int, num_hours: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    price = rng.uniform(20, 80, size=(num_loc, num_hours))
    water = rng.uniform(0.2, 3.0, size=(num_loc, num_hours))
    carbon = rng.uniform(50, 700, size=(num_loc, num_hours))
    workload = rng.uniform(0.1, num_loc * 0.5, size=(1, num_hours))

    def perfect_predictions(true_signal):
        return np.stack(
            [
                np.pad(
                    true_signal[:, i + 1 : i + 1 + horizon],
                    ((0, 0), (0, max(0, horizon - (num_hours - i - 1)))),
                )
                for i in range(num_hours)
            ],
            axis=0,
        )

    allocation = mpc.run_receding_horizon(
        perfect_predictions(price),
        perfect_predictions(water),
        perfect_predictions(carbon),
        perfect_predictions(workload),
        price,
        water,
        carbon,
        workload,
        l1=60,
        l2=1500,
    )
    return allocation, workload


def test_receding_horizon_backlog_stays_bounded():
    """Regression test for a real bug: deferring workload to a *forecast*
    column within the lookahead window is a legitimate MPC planning choice,
    but only column 0 is ever realized -- anything planned into later columns
    is thrown away once the window rolls forward. Unless that deferred amount
    is explicitly carried forward as backlog (see `run_receding_horizon`'s
    docstring), it silently vanishes: on a 121-hour real evaluation this
    dropped ~16% of all workload, unboundedly, and would have dropped a
    *larger* fraction on a longer trace since the leak was never re-injected.

    A correctly bounded scheduler instead always has a small, roughly
    constant amount of workload "in flight" at any snapshot (normal for a
    receding-horizon delay-tolerant scheduler) -- so the *unserved fraction*
    should not grow as the trace gets longer; it should if anything shrink,
    since a fixed absolute backlog is a smaller share of a bigger total.
    """

    def unserved_fraction(num_hours: int) -> float:
        allocation, workload = _run_perfect_foresight_mpc(num_loc=3, horizon=4, num_hours=num_hours)
        return 1 - allocation.sum() / workload.sum()

    short_gap = unserved_fraction(40)
    long_gap = unserved_fraction(120)
    assert long_gap <= short_gap + 0.05, (
        f"unserved fraction grew with trace length ({short_gap:.3f} -> {long_gap:.3f}); "
        "looks like backlog is leaking rather than being carried forward"
    )


def test_num_loc_is_derived_not_hardcoded():
    """Regression test: the original code hardcoded a loop over range(10)."""
    price, water, carbon, workload = _tiny_instance(num_loc=4, horizon=3)
    cost, allocation = mpc.offline_oracle(price, water, carbon, workload, l1=60, l2=1500)
    assert allocation.shape == (4, 3)
