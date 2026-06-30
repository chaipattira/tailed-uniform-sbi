import numpy as np
import pytest
from toolbox.evaluators import c2st


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
