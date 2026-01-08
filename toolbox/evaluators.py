# ABOUTME: Evaluation classes for SBI posterior quality assessment
# ABOUTME: Includes C2ST, TARP, and spatial evaluation methods

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split


class SBIEvaluator:
    """Compare SBI posteriors using C2ST and TARP metrics"""

    def __init__(self, param_names=['θ₁', 'θ₂']):
        self.param_names = param_names

    def c2st(self, X1, X2):
        """C2ST score - lower is better (0.5 = identical distributions)"""
        X = np.vstack([X1, X2])
        y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        return hamming_loss(y_test, y_pred)

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
                score = self.c2st(samples_dict[method1], samples_dict[method2])
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
                samples.append(posterior.sample((n_samples,), x_obs).cpu().numpy())
            results[name] = samples

        return results

    def c2st(self, X1, X2):
        """C2ST score"""
        X = np.vstack([X1, X2])
        y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return hamming_loss(y_test, LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))

    def compute_c2st_grid(self, results, n_points_per_dim):
        """Compute C2ST for grid points"""
        methods = ['Uniform', 'Tailed-Uniform', 'Reference']
        c2st_grid = {}

        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                comparison_name = f"{m1}_vs_{m2}"
                c2st_values = []

                for idx in range(len(results['test_points'])):
                    c2st_val = self.c2st(results[m1][idx], results[m2][idx])
                    c2st_values.append(c2st_val)

                c2st_grid[comparison_name] = np.array(c2st_values).reshape(n_points_per_dim, n_points_per_dim)

        return c2st_grid

    def plot_c2st_grid(self, c2st_grid, n_points_per_dim):
        """Plot C2ST scores as heatmaps"""
        sns.set(style="white")
        comparisons = list(c2st_grid.keys())
        fig, axes = plt.subplots(len(comparisons), 1, figsize=(10, 8*len(comparisons)))
        if len(comparisons) == 1: axes = [axes]

        for i, comp in enumerate(comparisons):
            im = axes[i].imshow(c2st_grid[comp], cmap='RdYlBu_r', vmin=0.3, vmax=0.6,
                               extent=[self.param_ranges[0][0], self.param_ranges[0][1],
                                      self.param_ranges[1][0], self.param_ranges[1][1]],
                               origin='lower')

            # Add text annotations
            for j in range(n_points_per_dim):
                for k in range(n_points_per_dim):
                    value = c2st_grid[comp][j, k]
                    color = 'white' if value > 0.5 else 'black'

                    # Calculate pixel coordinates for centering
                    x_extent = self.param_ranges[0][1] - self.param_ranges[0][0]
                    y_extent = self.param_ranges[1][1] - self.param_ranges[1][0]
                    x_pos = self.param_ranges[0][0] + (k + 0.5) * x_extent / n_points_per_dim
                    y_pos = self.param_ranges[1][0] + (j + 0.5) * y_extent / n_points_per_dim

                    axes[i].text(x_pos, y_pos, f'{value:.2f}', ha='center', va='center',
                               color=color, fontweight='bold', fontsize=6)

            axes[i].set_title(comp.replace('_', ' '), fontsize=18)
            axes[i].set_xlabel('Parameter 1', fontsize=16)
            axes[i].set_ylabel('Parameter 2', fontsize=16)
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('C2ST', fontsize=16)
            cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()
        plt.show()


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
                samples.append(posterior.sample((n_samples,), x_obs).cpu().numpy())
            results[name] = samples

        return results

    def c2st(self, X1, X2):
        """C2ST score"""
        X = np.vstack([X1, X2])
        y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return hamming_loss(y_test, LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))

    def compute_c2st_by_radius(self, results, radii):
        """Compute C2ST organized by radius"""
        methods = ['Uniform', 'TailedUniform', 'Reference']
        unique_radii = np.unique(np.round(radii, 3))
        c2st_data = {}

        for radius in unique_radii:
            indices = np.where(np.abs(radii - radius) < 1e-3)[0]
            c2st_data[f'{radius:.3f}'] = {}

            for i, m1 in enumerate(methods):
                for m2 in methods[i+1:]:
                    c2st_vals = [self.c2st(results[m1][idx], results[m2][idx]) for idx in indices]
                    c2st_data[f'{radius:.3f}'][f"{m1} vs {m2}"] = c2st_vals

        return c2st_data

    def plot_c2st_by_radius(self, c2st_data):
        """Plot median C2ST scores by radius with percentile error bars"""
        comparisons = list(next(iter(c2st_data.values())).keys())
        radii = [float(r) for r in c2st_data.keys()]

        plt.figure(figsize=(10, 6))

        # Define horizontal offsets for each comparison
        offset_amount = 0.01
        offsets = {
            "Uniform vs TailedUniform": -offset_amount,
            "Uniform vs Reference": 0,
            "TailedUniform vs Reference": offset_amount
        }

        for comp in comparisons:
            median_c2st = []
            lower_err = []
            upper_err = []

            for radius_str in c2st_data.keys():
                vals = np.array(c2st_data[radius_str][comp])
                median = np.median(vals)
                p16 = np.percentile(vals, 16)
                p84 = np.percentile(vals, 84)

                median_c2st.append(median)
                lower_err.append(median - p16)
                upper_err.append(p84 - median)

            # Set color and style based on comparison
            if "Uniform vs TailedUniform" in comp:
                linestyle = '--'
                alpha = 0.5
                color = 'gray'
            elif "TailedUniform vs Reference" in comp:
                linestyle = '-'
                alpha = 0.85
                color = 'green'
            elif "Uniform vs Reference" in comp:
                linestyle = '-'
                alpha = 0.85
                color = "#0096FF"
            else:
                linestyle = '-'
                alpha = 0.85
                color = 'green'

            # Apply horizontal offset
            offset = offsets.get(comp, 0)
            radii_offset = [r + offset for r in radii]

            # Remove connecting lines by using 'o' instead of 'o-'
            plt.errorbar(radii_offset, median_c2st,
                        yerr=[lower_err, upper_err],
                        fmt='o', label=comp,
                        alpha=alpha, color=color,
                        capsize=2, capthick=1.5, markersize=6, barsabove=True, elinewidth=1)

        plt.axhline(0.5, color='gray', linestyle='--', linewidth=2, label='Ideal C2ST=0.5')
        plt.xlabel('Radius', fontsize=16)
        plt.ylabel('C2ST', fontsize=16)
        plt.title('C2ST Performance by Distance from Center', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_c2st_tables(self, c2st_data):
        """Plot C2ST as colored tables"""
        sns.set(style="white")
        comparisons = list(next(iter(c2st_data.values())).keys())
        fig, axes = plt.subplots(len(comparisons), 1, figsize=(20, 10*len(comparisons)))
        if len(comparisons) == 1: axes = [axes]

        for i, comp in enumerate(comparisons):
            radii = list(c2st_data.keys())
            max_pts = max(len(c2st_data[r][comp]) for r in radii)

            # Create data matrix
            data = np.full((len(radii), max_pts), np.nan)
            for j, r in enumerate(radii):
                vals = c2st_data[r][comp]
                data[j, :len(vals)] = vals

            # Plot
            im = axes[i].imshow(data, cmap='RdYlBu_r', vmin=0.3, vmax=0.6)

            # Add text
            for j in range(len(radii)):
                for k in range(max_pts):
                    if not np.isnan(data[j, k]):
                        color = 'white' if data[j, k] > 0.5 else 'black'
                        axes[i].text(k, j, f'{data[j, k]:.2f}', ha='center', va='center',
                                     color=color, fontweight='bold')

            axes[i].set_yticks(range(len(radii)))
            axes[i].set_yticklabels([f'r={r}' for r in radii])
            axes[i].set_title(comp.replace('_', ' '))
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.show()


class RectGridEvaluator:
    """Evaluate NPE posteriors on scientific simulation grids"""

    def __init__(self, param_ranges):
        self.param_ranges = param_ranges

    def c2st(self, X1, X2):
        """C2ST score - measures distributional similarity"""
        X = np.vstack([X1, X2])
        y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = LogisticRegression(max_iter=1000)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        return hamming_loss(y_test, y_pred)

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
                samples = posterior.sample((n_samples,), x_obs).cpu().numpy()
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
                c2st_val = self.c2st(reference_samples[idx], npe_results[method][idx])
                c2st_values.append(c2st_val)

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
                    c2st_val = self.c2st(npe_results[m1][idx], npe_results[m2][idx])
                    c2st_values.append(c2st_val)

                c2st_values = np.array(c2st_values)
                c2st_grid[comparison_name] = c2st_values.reshape(n_points_per_dim,
                                                                 n_points_per_dim)

        return c2st_grid

    def plot_c2st_heatmaps(self, c2st_grid, n_points_per_dim, save_path=None):
        """
        Plot C2ST scores as heatmaps

        Args:
            c2st_grid: Dictionary of C2ST arrays
            n_points_per_dim: Grid dimension
            save_path: Optional path to save figure
        """
        comparisons = list(c2st_grid.keys())
        n_comparisons = len(comparisons)

        fig, axes = plt.subplots(n_comparisons, 1,
                                figsize=(12, 8*n_comparisons))
        if n_comparisons == 1:
            axes = [axes]

        for i, comp in enumerate(comparisons):
            im = axes[i].imshow(c2st_grid[comp], cmap='RdYlBu_r',
                               vmin=0.2, vmax=0.5,
                               extent=[self.param_ranges[0][0],
                                      self.param_ranges[0][1],
                                      self.param_ranges[1][0],
                                      self.param_ranges[1][1]],
                               origin='lower', aspect='auto')

            # Add text annotations
            for j in range(n_points_per_dim):
                for k in range(n_points_per_dim):
                    value = c2st_grid[comp][j, k]
                    color = 'white' if value > 0.5 else 'black'

                    # Calculate pixel coordinates
                    x_extent = self.param_ranges[0][1] - self.param_ranges[0][0]
                    y_extent = self.param_ranges[1][1] - self.param_ranges[1][0]
                    x_pos = self.param_ranges[0][0] + (k + 0.5) * x_extent / n_points_per_dim
                    y_pos = self.param_ranges[1][0] + (j + 0.5) * y_extent / n_points_per_dim

                    axes[i].text(x_pos, y_pos, f'{value:.2f}',
                               ha='center', va='center',
                               color=color, fontsize=8)

            axes[i].set_title(comp.replace('_', ' '), fontsize=18)
            axes[i].set_xlabel('$\\Omega_m$', fontsize=14)
            axes[i].set_ylabel('$h$', fontsize=14)
            cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('C2ST Score', fontsize=14)
            cbar.ax.axhline(0.5, color='black', linestyle='--', linewidth=2)
            axes[i].grid(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"\nSaved heatmap to: {save_path}")

        plt.show()


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
            ref_post = self.task._get_reference_posterior(observation=x_obs.unsqueeze(0))
            ref_samples.append(ref_post.sample((n_samples,)).cpu().numpy())
        results['Reference'] = ref_samples

        # Learned posteriors
        for name, posterior in posterior_dict.items():
            samples = []
            for x_obs in tqdm(observations, desc=name):
                samples.append(posterior.sample((n_samples,), x_obs).cpu().numpy())
            results[name] = samples

        return results

    def c2st(self, X1, X2):
        """C2ST score"""
        X = np.vstack([X1, X2])
        y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return hamming_loss(y_test, LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))
