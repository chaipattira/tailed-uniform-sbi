import numpy as np
import pytest
import torch
from toolbox.evaluators import c2st, Evaluator


def _same_gaussian(n=1000, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X, X.copy()


def _separated_gaussians(n=1000, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((n, d))
    X2 = rng.standard_normal((n, d)) + 50.0
    return X1, X2


def test_c2st_identical_distributions():
    X1, X2 = _same_gaussian()
    score = c2st(X1, X2)
    assert 0.4 <= score <= 0.6, f"Expected ~0.5 for identical distributions, got {score}"


def test_c2st_separated_distributions():
    X1, X2 = _separated_gaussians()
    score = c2st(X1, X2)
    assert score < 0.1, f"Expected near 0 for well-separated distributions, got {score}"


def test_evaluator_accepts_reference_sampler():
    calls = []
    def ref_sampler(x_obs, n_samples):
        calls.append(n_samples)
        return np.zeros((n_samples, 2))

    param_ranges = [(-1.0, 1.0), (-1.0, 1.0)]
    ev = Evaluator(simulator=None, param_ranges=param_ranges, reference_sampler=ref_sampler)
    assert ev.reference_sampler is ref_sampler
    assert ev.dim == 2


def test_evaluator_calls_reference_sampler_in_evaluate_all():
    n_samples = 10
    calls = []
    def ref_sampler(x_obs, n):
        calls.append(n)
        return np.zeros((n, 2))

    def simulator(theta):
        return torch.zeros(4)

    param_ranges = [(-1.0, 1.0), (-1.0, 1.0)]
    ev = Evaluator(simulator=simulator, param_ranges=param_ranges, reference_sampler=ref_sampler)
    test_points = np.array([[0.0, 0.0], [0.5, 0.5]])

    class FakePosterior:
        def sample(self, shape, x_obs):
            return torch.zeros(*shape, 2)

    results = ev.evaluate_all({'p': FakePosterior()}, test_points, n_samples)
    assert len(calls) == len(test_points)
    assert all(c == n_samples for c in calls)
    assert 'Reference' in results
    assert results['Reference'][0].shape == (n_samples, 2)
