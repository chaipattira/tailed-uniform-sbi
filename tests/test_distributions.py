import torch
import pytest
from toolbox.distributions import (
    GaussianTailed, LinearTailed, ExponentialTailed, UniformTailed,
)

CLASSES = [GaussianTailed, LinearTailed, ExponentialTailed, UniformTailed]


def _make_dist(cls, d=2):
    a = torch.tensor([-1.0] * d)
    b = torch.tensor([ 1.0] * d)
    sigma = torch.tensor([0.2] * d)
    return cls(a=a, b=b, sigma=sigma)


@pytest.mark.parametrize("cls", CLASSES, ids=[c.__name__ for c in CLASSES])
def test_sample_lhs_shape(cls):
    dist = _make_dist(cls, d=2)
    samples = dist.sample_lhs(100)
    assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"


@pytest.mark.parametrize("cls", CLASSES, ids=[c.__name__ for c in CLASSES])
def test_sample_lhs_mean_near_zero(cls):
    dist = _make_dist(cls, d=2)
    samples = dist.sample_lhs(2000)
    mean = samples.mean(dim=0)
    assert torch.all(mean.abs() < 0.1), f"Expected mean near 0 for symmetric dist, got {mean}"
