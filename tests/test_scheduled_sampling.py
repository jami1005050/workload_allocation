import dataclasses

import torch

from greenload.config import DEFAULT
from greenload.models.gru_forecaster import GRUForecaster, scheduled_sampling_prob


def test_schedule_is_pure_teacher_forcing_during_warmup():
    for epoch in range(DEFAULT.ss_warmup_epochs):
        assert scheduled_sampling_prob(epoch, DEFAULT, total_epochs=25) == 0.0


def test_schedule_is_monotonically_non_decreasing():
    probs = [scheduled_sampling_prob(e, DEFAULT, total_epochs=25) for e in range(25)]
    assert all(b >= a for a, b in zip(probs, probs[1:], strict=False))


def test_schedule_never_exceeds_cap():
    probs = [scheduled_sampling_prob(e, DEFAULT, total_epochs=25) for e in range(25)]
    assert max(probs) <= DEFAULT.ss_max_sampling_prob + 1e-9
    assert max(probs) > 0.0  # cap should actually be reached/approached by the last epoch


def test_zero_max_prob_config_is_pure_teacher_forcing():
    cfg = dataclasses.replace(DEFAULT, ss_max_sampling_prob=0.0)
    probs = [scheduled_sampling_prob(e, cfg, total_epochs=25) for e in range(25)]
    assert all(p == 0.0 for p in probs)


def test_forward_target_none_is_closed_loop_and_ignores_sampling_prob():
    model = GRUForecaster(hidden_size=4, num_layers=1, horizon=6, dropout=0.0)
    x = torch.randn(2, 1, 1)
    out_a, _ = model(x, target=None, sampling_prob=0.0)
    out_b, _ = model(x, target=None, sampling_prob=0.9)
    assert out_a.shape == (2, 1, 6)
    torch.testing.assert_close(out_a, out_b)


def test_forward_teacher_forcing_uses_target_shape():
    model = GRUForecaster(hidden_size=4, num_layers=1, horizon=6, dropout=0.0)
    x = torch.randn(3, 1, 1)
    target = torch.randn(3, 1, 6)
    out, _ = model(x, target=target, sampling_prob=0.0)
    assert out.shape == target.shape
