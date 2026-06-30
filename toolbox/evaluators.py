# ABOUTME: Evaluation classes for SBI posterior quality assessment
# ABOUTME: Includes C2ST metric and Evaluator with injected reference sampler

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split


def c2st(X1, X2):
    """C2ST-Hamming: misclassification rate. 0.5 = indistinguishable, 0 = perfect separation."""
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return hamming_loss(y_test, LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))


class Evaluator:
    """Evaluate posteriors at spherical distance bins.

    reference_sampler: callable (x_obs: Tensor, n_samples: int) -> np.ndarray of shape (n_samples, dim)
    """

    def __init__(self, simulator, param_ranges, reference_sampler):
        self.simulator = simulator
        self.param_ranges = param_ranges
        self.reference_sampler = reference_sampler
        self.dim = len(param_ranges)
        self.prior_center = np.array([(low + high) / 2 for low, high in param_ranges])
        self.sigma = np.mean([(high - low) / np.sqrt(12) for low, high in param_ranges])
        self.max_radius = min([(high - low) / 2 for low, high in param_ranges])

    def sample_on_sphere(self, n_points, radius, dim, filter_bounds=True):
        """Sample points uniformly on a sphere of given radius in d dimensions."""
        if radius == 0:
            return np.array([self.prior_center])

        points = np.random.randn(n_points, dim)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms * radius + self.prior_center

        if filter_bounds:
            valid_points = [p for p in points
                            if all(self.param_ranges[i][0] <= p[i] <= self.param_ranges[i][1]
                                   for i in range(self.dim))]
            return np.array(valid_points)
        return points

    def create_test_points(self, n_points_per_radius=50):
        """Create test points at distance bins: center, r=0.25/0.5/0.75/1.0, 2σ-extrap."""
        test_points = []
        distance_bins = []
        distance_labels = ['center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0', '2sigma-extrap']
        radii = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0 * self.sigma]

        for radius, label in zip(radii, distance_labels):
            if radius == 0:
                test_points.append(self.prior_center.copy())
                distance_bins.append(label)
            else:
                use_filter = (label != '2sigma-extrap')
                points_at_radius = []
                for _ in range(20):
                    if len(points_at_radius) >= n_points_per_radius:
                        break
                    sample_count = n_points_per_radius * 3 if use_filter else n_points_per_radius
                    new_pts = self.sample_on_sphere(sample_count, radius, self.dim, filter_bounds=use_filter)
                    if len(new_pts) > 0:
                        points_at_radius.extend(new_pts)

                points_at_radius = points_at_radius[:n_points_per_radius]
                if len(points_at_radius) < n_points_per_radius:
                    print(f"Warning: Only got {len(points_at_radius)}/{n_points_per_radius} points for {label}")

                for point in points_at_radius:
                    test_points.append(point)
                    distance_bins.append(label)

        return np.array(test_points), np.array(distance_bins)

    def evaluate_all(self, posterior_dict, test_points, n_samples):
        """Evaluate posteriors and reference at each test point.

        Returns dict with 'test_points', 'observations', 'Reference', and one key per posterior.
        """
        results = {'test_points': test_points}

        observations = []
        for theta in tqdm(test_points, desc="Generating observations"):
            observations.append(self.simulator(torch.tensor(theta, dtype=torch.float32)))
        results['observations'] = observations

        results['Reference'] = [self.reference_sampler(x_obs, n_samples)
                                 for x_obs in tqdm(observations, desc="Reference")]

        for name, posterior in posterior_dict.items():
            results[name] = [posterior.sample((n_samples,), x_obs).cpu().numpy()
                             for x_obs in tqdm(observations, desc=name)]

        return results
