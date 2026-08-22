import numpy as np

from greenload.config import DEFAULT
from greenload.data.generate_synthetic_data import generate_raw_series


def test_shapes_and_reproducibility():
    series_a = generate_raw_series(DEFAULT)
    series_b = generate_raw_series(DEFAULT)
    assert series_a["workload"].shape == (DEFAULT.total_hours, 1)
    for signal in ("price", "water", "carbon"):
        assert series_a[signal].shape == (DEFAULT.total_hours, DEFAULT.num_loc)
    for signal in series_a:
        np.testing.assert_array_equal(series_a[signal], series_b[signal])


def test_non_negative():
    series = generate_raw_series(DEFAULT)
    for name, arr in series.items():
        assert (arr >= 0).all(), f"{name} has negative values"


def test_locations_are_distinct():
    series = generate_raw_series(DEFAULT)
    for signal in ("price", "water", "carbon"):
        means = series[signal].mean(axis=0)
        # locations should not collapse to near-identical baselines
        assert means.std() / means.mean() > 0.1, f"{signal}: locations too similar"


def test_workload_utilization_feasible():
    series = generate_raw_series(DEFAULT)
    workload = series["workload"][:, 0]
    fleet_capacity_per_hour = DEFAULT.num_loc * DEFAULT.max_cap
    rolling_24h = np.convolve(workload, np.ones(24), mode="valid")
    utilization = rolling_24h / (fleet_capacity_per_hour * 24)
    assert utilization.min() > 0, "degenerate all-zero window found"
    assert utilization.max() < 1.0, "workload exceeds fleet capacity within a 24h window"
    assert 0.2 < utilization.mean() < 0.9, (
        f"utilization {utilization.mean():.2f} outside a sane band"
    )
