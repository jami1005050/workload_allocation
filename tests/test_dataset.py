import numpy as np

from greenload.config import DEFAULT
from greenload.data.dataset import build_all_splits
from greenload.data.generate_synthetic_data import generate_raw_series


def test_split_shapes():
    raw = generate_raw_series(DEFAULT)
    splits = build_all_splits(raw, DEFAULT)

    for signal, num_loc in (("workload", 1), ("price", 10), ("water", 10), ("carbon", 10)):
        s = splits[signal]
        assert s.train.num_windows == DEFAULT.num_train_windows
        assert s.val.num_windows == DEFAULT.num_val_windows
        assert s.train.source.shape == (DEFAULT.num_train_windows * num_loc, 1, 1)
        assert s.train.target.shape == (DEFAULT.num_train_windows * num_loc, 1, DEFAULT.window_size)
        expected_test_windows = (
            DEFAULT.total_hours - DEFAULT.train_block_hours
        ) - DEFAULT.window_size
        assert s.test.num_windows == expected_test_windows


def test_no_train_val_leakage():
    """Train/val windows must be split by chronological order, not shuffled first.

    The original data pipeline built all overlapping windows, shuffled, then
    sliced -- letting near-duplicate overlapping windows land in both splits.
    This asserts window i's (denormalized) source value is exactly
    raw_series[i] for i in the train range, and raw_series[n_train + i] for i
    in the val range: i.e. the split boundary follows time order, so windows
    are only ever adjacent across the boundary, never duplicated across it.
    """
    raw = generate_raw_series(DEFAULT)
    splits = build_all_splits(raw, DEFAULT)
    n_train = DEFAULT.num_train_windows
    scaler = splits["price"].scaler

    train_source = scaler.denormalize(
        splits["price"].train.source.numpy().reshape(n_train, DEFAULT.num_loc)
    )
    val_source = scaler.denormalize(
        splits["price"].val.source.numpy().reshape(DEFAULT.num_val_windows, DEFAULT.num_loc)
    )

    np.testing.assert_allclose(train_source, raw["price"][:n_train], atol=1e-4)
    np.testing.assert_allclose(
        val_source, raw["price"][n_train : n_train + DEFAULT.num_val_windows], atol=1e-4
    )


def test_scaler_uses_only_training_statistics():
    """The scaler must be computed from the train block, not from test data
    (otherwise the model would implicitly see test-set statistics)."""
    raw = generate_raw_series(DEFAULT)
    splits = build_all_splits(raw, DEFAULT)
    scaler = splits["price"].scaler
    train_block = raw["price"][: DEFAULT.train_block_hours]
    assert np.isclose(scaler.mean, train_block.mean())
    assert np.isclose(scaler.std, train_block.std())


def test_target_is_one_step_ahead_of_source():
    raw = generate_raw_series(DEFAULT)
    splits = build_all_splits(raw, DEFAULT)
    train = splits["workload"].train
    # target[:, :, 0] should equal the *next* hour after source, not source itself
    source = train.source.numpy().reshape(-1)
    first_target_step = train.target.numpy()[:, 0, 0]
    assert not np.allclose(source, first_target_step)
