# ABOUTME: Evaluation classes for SBI posterior quality assessment
# ABOUTME: Includes C2ST, TARP, and spatial evaluation methods

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split


def c2st(X1, X2):
    """C2ST-Hamming: 0.5 = identical distributions, 0 = fully separable."""
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return hamming_loss(y_test, LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))


class SBIEvaluator:
    """Compare SBI posteriors using C2ST and TARP metrics"""

    def __init__(self, param_names=['θ₁', 'θ₂']):
        self.param_names = param_names

    def tarp_score(self, samples, true_theta):
        """TARP calibration score - lower is better"""
        alpha_levels = np.linspace(0.05, 0.95, 19)
        empirical_coverage = []

        for alpha in alpha_levels:
            coverage = all(
                np.percentile(samples[:, j], 100*alpha/2) <= true_theta[j] <=
                np.percentile(samples[:, j], 100*(1-alpha/2))
                for j in range(len(true_theta))
            )
            empirical_coverage.append(coverage)

        expected_coverage = 1 - alpha_levels
        return np.mean(np.abs(expected_coverage - empirical_coverage))

    def compare(self, samples_dict, true_theta):
        """Compare multiple methods"""
        results = {}
        methods = list(samples_dict.keys())

        for method, samples in samples_dict.items():
            results[method] = {
                'tarp': self.tarp_score(samples, true_theta),
                'mean': np.mean(samples, axis=0),
                'std': np.std(samples, axis=0),
                'c2st_vs_others': {}
            }

        # C2ST comparisons
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                score = c2st(samples_dict[method1], samples_dict[method2])
                results[method1]['c2st_vs_others'][method2] = score
                results[method2]['c2st_vs_others'][method1] = score

        return results

    def print_results(self, results, true_theta):
        """Print concise comparison"""
        methods = list(results.keys())

        print("Method Comparison")
        print("-" * 50)

        for method in methods:
            stats = results[method]
            print(f"\n{method}:")
            print(f"  TARP: {stats['tarp']:.4f}")
            for i, param in enumerate(self.param_names):
                print(f"  {param}: Mean={stats['mean'][i]:.3f}, Std={stats['std'][i]:.3f}")

            if stats['c2st_vs_others']:
                c2st_str = ", ".join([f"{k}={v:.3f}" for k, v in stats['c2st_vs_others'].items()])
                print(f"  C2ST: {c2st_str}")


class GridEvaluator:
    """Evaluate posteriors on rectangular parameter grid"""

    def __init__(self, simulator, param_ranges, task):
        self.simulator = simulator
        self.param_ranges = param_ranges
        self.task = task

    def create_test_points(self, n_points_per_dim):
        """Create test points on rectangular grid"""
        x_points = np.linspace(self.param_ranges[0][0], self.param_ranges[0][1], n_points_per_dim)
        y_points = np.linspace(self.param_ranges[1][0], self.param_ranges[1][1], n_points_per_dim)

        test_points = []
        for x in x_points:
            for y in y_points:
                test_points.append([x, y])

        return np.array(test_points)

    def evaluate_all(self, posterior_dict, test_points, n_samples):
        """Evaluate all posteriors including reference"""
        observations = []
        results = {'test_points': test_points}

        for theta in tqdm(test_points):
            x_obs = self.simulator(torch.tensor(theta, dtype=torch.float32))
            observations.append(x_obs)

        results['observations'] = observations

        ref_samples = []
        for x_obs in tqdm(observations, desc="Reference"):
            ref_post = self.task._get_reference_posterior(observation=x_obs.unsqueeze(0))
            ref_samples.append(ref_post.sample((n_samples,)).cpu().numpy())
        results['Reference'] = ref_samples

        for name, posterior in posterior_dict.items():
            samples = []
            for x_obs in tqdm(observations, desc=name):
                samples.append(posterior.sample((n_samples,), x_obs, ).cpu().numpy())
            results[name] = samples

        return results

    def compute_c2st_grid(self, results, n_points_per_dim):
        """Compute C2ST for grid points"""
        methods = ['Uniform', 'Tailed-Uniform', 'Reference']
        c2st_grid = {}

        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                comparison_name = f"{m1}_vs_{m2}"
                c2st_values = [c2st(results[m1][idx], results[m2][idx])
                               for idx in range(len(results['test_points']))]
                c2st_grid[comparison_name] = np.array(c2st_values).reshape(n_points_per_dim, n_points_per_dim)

        return c2st_grid


class CircleEvaluator:
    """Evaluate posteriors at varying radial distances"""

    def __init__(self, simulator, param_ranges, task):
        self.simulator = simulator
        self.param_ranges = param_ranges
        self.task = task
        self.prior_center = np.array([(low + high) / 2 for low, high in param_ranges])
        self.max_radius = min([(high - low) / 2 for low, high in param_ranges])

    def create_test_points(self, n_radii=12, n_angles=30):
        """Create test points on concentric circles"""
        test_points = [self.prior_center.copy()]
        radii = [0.0]

        for radius in np.linspace(0.1 * self.max_radius, self.max_radius, n_radii):
            for angle in np.linspace(0, 2*np.pi, n_angles, endpoint=False):
                x = self.prior_center[0] + radius * np.cos(angle)
                y = self.prior_center[1] + radius * np.sin(angle)

                if (self.param_ranges[0][0] <= x <= self.param_ranges[0][1] and
                    self.param_ranges[1][0] <= y <= self.param_ranges[1][1]):
                    test_points.append([x, y])
                    radii.append(radius)

        return np.array(test_points), np.array(radii)

    def evaluate_all(self, posterior_dict, test_points, n_samples):
        """Evaluate all posteriors including reference"""
        observations = []
        results = {'test_points': test_points}

        # Generate observations and reference posteriors
        for theta in tqdm(test_points):
            x_obs = self.simulator(torch.tensor(theta, dtype=torch.float32))
            observations.append(x_obs)

        results['observations'] = observations

        # Reference posteriors
        ref_samples = []
        for x_obs in tqdm(observations, desc="Reference"):
            ref_post = self.task._get_reference_posterior(observation=x_obs.unsqueeze(0))
            ref_samples.append(ref_post.sample((n_samples,)).cpu().numpy())
        results['Reference'] = ref_samples

        # Learned posteriors
        for name, posterior in posterior_dict.items():
            samples = []
            for x_obs in tqdm(observations, desc=name):
                samples.append(posterior.sample((n_samples,), x_obs, ).cpu().numpy())
            results[name] = samples

        return results

    def compute_c2st_by_radius(self, results, radii):
        """Compute C2ST organized by radius"""
        methods = ['Uniform', 'GaussianTailed', 'Reference']
        unique_radii = np.unique(np.round(radii, 3))
        c2st_data = {}

        for radius in unique_radii:
            indices = np.where(np.abs(radii - radius) < 1e-3)[0]
            c2st_data[f'{radius:.3f}'] = {}

            for i, m1 in enumerate(methods):
                for m2 in methods[i+1:]:
                    c2st_vals = [c2st(results[m1][idx], results[m2][idx]) for idx in indices]
                    c2st_data[f'{radius:.3f}'][f"{m1} vs {m2}"] = c2st_vals

        return c2st_data


class RectGridEvaluator:
    """Evaluate NPE posteriors on scientific simulation grids"""

    def __init__(self, param_ranges):
        self.param_ranges = param_ranges

    def load_grid_data(self, mcmc_folder, n_points_per_dim=20):
        """
        Load Reference grid data from disk

        Returns:
            test_points: (n_points, 2) array of test points
            reference_samples: (n_points, n_samples, 2) array of Reference samples
            observations: (n_points, n_k_bins) array of observations
        """
        try:
            test_points = np.load(f'{mcmc_folder}/test_points.npy')
            reference_samples = np.load(f'{mcmc_folder}/mcmc_samples.npy')
            observations = np.load(f'{mcmc_folder}/observations.npy')
            print(f"Loaded grid data:")
            return test_points, reference_samples, observations
        except:
            raise FileNotFoundError(f"No grid data found in {mcmc_folder}")

    def evaluate_npe_on_grid(self, posterior_dict, observations, n_samples=8000):
        """
        Sample from NPE posteriors for all grid observations

        Args:
            posterior_dict: Dictionary of {name: posterior_ensemble}
            observations: Array of observations (n_points, n_features)
            n_samples: Number of posterior samples per point

        Returns:
            results_dict: {method_name: samples_array} where samples_array
                         has shape (n_points, n_samples, 2)
        """
        results = {}

        for name, posterior in posterior_dict.items():
            print(f"\nSampling from {name}...")
            samples_list = []

            for i, x_obs in enumerate(tqdm(observations, desc=f"  {name}")):
                # Convert to log10 if needed
                if not np.all(x_obs < 10):
                    x_obs_log = np.log10(x_obs)
                    mask = np.isnan(x_obs_log)
                    if np.any(mask):
                        x_obs_log[mask] = np.nanmean(x_obs_log)
                    x_obs = x_obs_log

                # Sample from posterior
                samples = posterior.sample((n_samples,), x_obs, ).cpu().numpy()
                samples_list.append(samples)

            results[name] = np.array(samples_list)
            print(f"  Shape: {results[name].shape}")

        return results

    def save_npe_samples(self, npe_results, save_folder):
        """
        Save NPE samples to disk

        Args:
            npe_results: Dictionary of NPE samples {name: (n_points, n_samples, 2)}
            save_folder: Path to save the samples
        """
        print("\nSaving NPE samples to disk...")
        for method_name, samples in npe_results.items():
            filename = f'{save_folder}/{method_name.lower()}_samples.npy'
            np.save(filename, samples)
            file_size_mb = samples.nbytes / 1e6
            print(f"  Saved {filename}")
            print(f"    Shape: {samples.shape}, Size: {file_size_mb:.2f} MB")

        print("\nAll NPE samples saved successfully!")

    def compute_radial_distance(self, test_points, normalize=True):
        """
        Compute radial distance of test points from parameter space center

        Args:
            test_points: (n_points, 2) array of test points
            normalize: If True, normalize by parameter ranges before computing distance

        Returns:
            distances: (n_points,) array of distances from center
            center: (2,) array of parameter space center
        """
        # Compute center of parameter space
        center = np.array([
            (self.param_ranges[0][0] + self.param_ranges[0][1]) / 2,
            (self.param_ranges[1][0] + self.param_ranges[1][1]) / 2
        ])

        if normalize:
            # Normalize by parameter ranges to give equal weight to both dimensions
            param_widths = np.array([
                self.param_ranges[0][1] - self.param_ranges[0][0],
                self.param_ranges[1][1] - self.param_ranges[1][0]
            ])
            normalized_points = (test_points - center) / param_widths
            distances = np.linalg.norm(normalized_points, axis=1)
        else:
            distances = np.linalg.norm(test_points - center, axis=1)

        return distances, center

    def compute_c2st_grid(self, reference_samples, npe_results, n_points_per_dim):
        """
        Compute C2ST scores across the grid

        Args:
            reference_samples: (n_points, n_reference_samples, 2) Reference samples
            npe_results: Dictionary of NPE samples {name: (n_points, n_samples, 2)}
            n_points_per_dim: Grid dimension

        Returns:
            c2st_grid: Dictionary of C2ST heatmaps
        """
        methods = list(npe_results.keys())
        c2st_grid = {}

        # Reference vs each NPE
        for method in methods:
            comparison_name = f"Reference_vs_{method}"
            c2st_values = []

            print(f"\nComputing C2ST for {comparison_name}...")
            for idx in tqdm(range(len(reference_samples))):
                c2st_values.append(c2st(reference_samples[idx], npe_results[method][idx]))

            c2st_values = np.array(c2st_values)
            c2st_grid[comparison_name] = c2st_values.reshape(n_points_per_dim,
                                                             n_points_per_dim)

        # NPE vs NPE comparisons
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                comparison_name = f"{m1}_vs_{m2}"
                c2st_values = []

                print(f"\nComputing C2ST for {comparison_name}...")
                for idx in tqdm(range(len(reference_samples))):
                    c2st_values.append(c2st(npe_results[m1][idx], npe_results[m2][idx]))

                c2st_values = np.array(c2st_values)
                c2st_grid[comparison_name] = c2st_values.reshape(n_points_per_dim,
                                                                 n_points_per_dim)

        return c2st_grid


class DistanceEvaluator:
    """Evaluate posteriors at spherical distance bins"""

    def __init__(self, simulator, param_ranges, task):
        self.simulator = simulator
        self.param_ranges = param_ranges
        self.task = task
        self.dim = len(param_ranges)
        self.prior_center = np.array([(low + high) / 2 for low, high in param_ranges])
        self.sigma = np.mean([(high - low) / np.sqrt(12) for low, high in param_ranges])
        self.max_radius = min([(high - low) / 2 for low, high in param_ranges])

    def sample_on_sphere(self, n_points, radius, dim, filter_bounds=True):
        """Sample points uniformly on a sphere of given radius in d dimensions"""
        if radius == 0:
            return np.array([self.prior_center])

        # Generate points from standard normal distribution
        points = np.random.randn(n_points, dim)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        # Scale to desired radius
        points = points * radius
        # Translate to prior center
        points = points + self.prior_center

        # Conditionally filter points based on flag
        if filter_bounds:
            # Filter points that fall within prior bounds
            valid_points = []
            for point in points:
                if all(self.param_ranges[i][0] <= point[i] <= self.param_ranges[i][1]
                       for i in range(self.dim)):
                    valid_points.append(point)
            return np.array(valid_points)
        else:
            # Return all points, even those outside prior
            return points

    def create_test_points(self, n_points_per_radius=50):
        """Create test points at different distance bins"""
        test_points = []
        distance_bins = []
        distance_labels = ['center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0', '2sigma-extrap']
        radii = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0 * self.sigma]

        for radius, label in zip(radii, distance_labels):
            if radius == 0:
                # Center point
                test_points.append(self.prior_center.copy())
                distance_bins.append(label)
            else:
                # Sample points on sphere at this radius
                attempts = 0
                max_attempts = 20
                points_at_radius = []

                # For extrapolation point, disable filtering
                use_filter = (label != '2sigma-extrap')

                while len(points_at_radius) < n_points_per_radius and attempts < max_attempts:
                    sample_count = n_points_per_radius * 3 if use_filter else n_points_per_radius
                    new_points = self.sample_on_sphere(
                        sample_count, radius, self.dim, filter_bounds=use_filter
                    )
                    if len(new_points) > 0:
                        points_at_radius.extend(new_points)
                    attempts += 1

                # Take only the requested number of points
                points_at_radius = points_at_radius[:n_points_per_radius]

                if len(points_at_radius) < n_points_per_radius:
                    print(f"Warning: Only got {len(points_at_radius)}/{n_points_per_radius} points for {label} (radius={radius:.3f})")

                for point in points_at_radius:
                    test_points.append(point)
                    distance_bins.append(label)

        return np.array(test_points), np.array(distance_bins)

    def get_reference_samples(self, x_obs, n_samples):
        """Sample from the reference posterior for a given observation.
        Handles both GaussianLinear (_get_reference_posterior) and
        GaussianLinearUniform (_sample_reference_posterior) task APIs."""
        if hasattr(self.task, '_get_reference_posterior'):
            ref_post = self.task._get_reference_posterior(observation=x_obs.unsqueeze(0))
            return ref_post.sample((n_samples,)).cpu().numpy()
        # _sample_reference_posterior uses Pyro batched sampling which can produce
        # shape (n_samples, 1, dim) or (n_samples*dim,) depending on Pyro version;
        # reshape to canonical (n_samples, dim).
        samples = self.task._sample_reference_posterior(n_samples, observation=x_obs.unsqueeze(0))
        return samples.reshape(n_samples, self.dim).cpu().numpy()

    def evaluate_all(self, posterior_dict, test_points, n_samples):
        """Evaluate all posteriors including reference"""
        observations = []
        results = {'test_points': test_points}

        # Generate observations
        for theta in tqdm(test_points, desc="Generating observations"):
            x_obs = self.simulator(torch.tensor(theta, dtype=torch.float32))
            observations.append(x_obs)

        results['observations'] = observations

        # Reference posteriors
        ref_samples = []
        for x_obs in tqdm(observations, desc="Reference"):
            ref_samples.append(self.get_reference_samples(x_obs, n_samples))
        results['Reference'] = ref_samples

        # Learned posteriors
        for name, posterior in posterior_dict.items():
            samples = []
            for x_obs in tqdm(observations, desc=name):
                samples.append(posterior.sample((n_samples,), x_obs, ).cpu().numpy())
            results[name] = samples

        return results

