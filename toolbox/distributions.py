# ABOUTME: Custom probability distributions for simulation-based inference
# ABOUTME: Boundary-extrapolation priors with equal-volume tail parameterization for SBI comparison
# ABOUTME: All distributions map the same sigma to the same total tail probability mass as GaussianTailed

import torch
from torch.distributions import Distribution, HalfNormal
from torch.distributions.utils import broadcast_all
import math
from scipy.stats import qmc
from ili.utils.distributions_pt import CustomIndependent


class GaussianTailed(Distribution):
    arg_constraints = {
        'a': torch.distributions.constraints.real,
        'b': torch.distributions.constraints.dependent,
        'sigma': torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, a, b, sigma, validate_args=None):
        self.a, self.b, self.sigma = broadcast_all(a, b, sigma)
        if torch.any(self.a >= self.b):
            raise ValueError("`a` must be less than `b`.")

        self.Z = math.sqrt(2 * math.pi) * self.sigma + (self.b - self.a)
        self.A = math.sqrt(2 * math.pi) * self.sigma / self.Z
        self.B = (self.b - self.a) / self.Z

        self.halfnormal = HalfNormal(self.sigma)

        super().__init__(batch_shape=self.a.size(),
                         validate_args=validate_args)

    def log_prob(self, x):
        x, a, b, sigma = broadcast_all(x, self.a, self.b, self.sigma)

        logA = torch.log(self.A.to(dtype=x.dtype, device=x.device))
        logB = torch.log(self.B.to(dtype=x.dtype, device=x.device))
        log_uniform = logB - torch.log(b - a)

        # left: x <= a => z = a - x
        z_left = torch.abs(a - x)
        log_halfnorm_left = self.halfnormal.log_prob(
            z_left) + logA - math.log(2.0)

        # right: x >= b => z = x - b
        z_right = torch.abs(x - b)
        log_halfnorm_right = self.halfnormal.log_prob(
            z_right) + logA - math.log(2.0)

        return torch.where(x <= a, log_halfnorm_left,
                           torch.where(x >= b, log_halfnorm_right,
                                       log_uniform))

    def cdf(self, x):
        x, a, b, sigma = broadcast_all(x, self.a, self.b, self.sigma)
        sqrt2 = math.sqrt(2.0)

        def Phi(z):  # Standard Normal CDF
            return 0.5 * (1 + torch.erf(z / sqrt2))

        left_cdf = self.A * Phi((x - a) / sigma)
        center_cdf = 0.5 * self.A + self.B * (x - a) / (b - a)
        right_cdf = self.B + self.A * Phi((x - b) / sigma)

        return torch.where(x <= a, left_cdf,
                           torch.where(x >= b, right_cdf,
                                       center_cdf))

    def icdf(self, u):
        # Helper function for the Inverse Standard Normal CDF
        def inv_Phi(p):
            # Clamping p to avoid NaNs from erfinv at the boundaries 0 and 1
            p_clamped = torch.clamp(p, 1e-9, 1.0 - 1e-9)
            return math.sqrt(2.0) * torch.erfinv(2.0 * p_clamped - 1.0)

        # Thresholds dividing the distribution regions
        thresh_left = 0.5 * self.A
        thresh_right = 1.0 - 0.5 * self.A

        u_left_norm = u / self.A
        left_tail = self.a + self.sigma * inv_Phi(u_left_norm)

        u_right_norm = (u - self.B) / self.A
        right_tail = self.b + self.sigma * inv_Phi(u_right_norm)

        u_middle_norm = (u - thresh_left) / self.B
        middle = self.a + u_middle_norm * (self.b - self.a)

        return torch.where(u < thresh_left, left_tail,
                           torch.where(u > thresh_right, right_tail, middle))

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.a.shape, device=self.a.device)
        thresh_left = 0.5 * self.A
        thresh_right = 1.0 - 0.5 * self.A

        left_tail = self.a - self.halfnormal.sample(sample_shape)
        right_tail = self.b + self.halfnormal.sample(sample_shape)

        x_middle = self.a + (u - thresh_left) * (self.b - self.a) / self.B

        return torch.where(u < thresh_left, left_tail,
                           torch.where(u > thresh_right, right_tail,
                                       x_middle))

    def sample_lhs(self, n_samples):
        """Sample using Latin Hypercube Sampling"""
        # Generate LHS samples in [0,1]^d
        sampler = qmc.LatinHypercube(d=len(self.a.flatten()), seed=42)
        u_samples = sampler.random(n_samples)
        u_tensor = torch.tensor(
            u_samples, dtype=torch.float32, device=self.a.device)

        # Transform using inverse CDF
        return self.icdf(u_tensor)

    def mean(self):
        return 0.5 * (self.a + self.b)


class IndependentGaussianTailed(CustomIndependent):
    Distribution = GaussianTailed

    def sample_lhs(self, n_samples):
        return self.base_dist.sample_lhs(n_samples)


class LinearTailed(Distribution):
    """Uniform box with linearly-decreasing tails beyond the boundaries.

    Given sigma, the total tail probability mass A matches GaussianTailed(a, b, sigma),
    so all boundary-extrapolation variants can be compared on equal footing.

    Tail shape: density ramps linearly from box_density at the boundary down to 0
    at distance L = A*W/B on each side, where W=b-a, A=tail mass, B=box mass.
    The distribution is continuous at [a, b] boundaries.
    """
    arg_constraints = {
        'a': torch.distributions.constraints.real,
        'b': torch.distributions.constraints.dependent,
        'sigma': torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, a, b, sigma, validate_args=None):
        self.a, self.b, self.sigma = broadcast_all(a, b, sigma)
        if torch.any(self.a >= self.b):
            raise ValueError("`a` must be less than `b`.")
        W = self.b - self.a
        Z = math.sqrt(2 * math.pi) * self.sigma + W
        self.A = math.sqrt(2 * math.pi) * self.sigma / Z   # total tail probability
        self.B = W / Z                                       # box probability
        self.L = self.A * W / self.B                         # linear tail extent each side
        self.box_density = self.B / W                        # density in box (= density at boundary)
        super().__init__(batch_shape=self.a.size(), validate_args=validate_args)

    def log_prob(self, x):
        x, a, b = broadcast_all(x, self.a, self.b)
        L = self.L.to(dtype=x.dtype, device=x.device)
        log_box = torch.log(self.box_density.to(dtype=x.dtype, device=x.device))
        a_ext, b_ext = a - L, b + L
        neg_inf = torch.full_like(x, float('-inf'))
        # Linear ramp: density = box_density * (dist_from_far_edge / L); zero at exact tip
        frac_left  = ((x - a_ext) / L).clamp(min=0.0)
        frac_right = ((b_ext - x) / L).clamp(min=0.0)
        log_left  = torch.where(frac_left  > 0, log_box + torch.log(frac_left.clamp(min=1e-38)),  neg_inf)
        log_right = torch.where(frac_right > 0, log_box + torch.log(frac_right.clamp(min=1e-38)), neg_inf)
        return torch.where(
            x <= a,
            torch.where(x >= a_ext, log_left, neg_inf),
            torch.where(x >= b,
                        torch.where(x <= b_ext, log_right, neg_inf),
                        log_box)
        )

    def cdf(self, x):
        x, a, b = broadcast_all(x, self.a, self.b)
        L  = self.L.to(dtype=x.dtype, device=x.device)
        A  = self.A.to(dtype=x.dtype, device=x.device)
        B  = self.B.to(dtype=x.dtype, device=x.device)
        W  = b - a
        a_ext, b_ext = a - L, b + L
        # Left tail: (A/2) * ((x - a_ext) / L)^2
        cdf_left  = (A / 2.0) * ((x - a_ext) / L) ** 2
        cdf_box   = A / 2.0 + (B / W) * (x - a)
        # Right tail: 1 - (A/2) * ((b_ext - x) / L)^2
        cdf_right = 1.0 - (A / 2.0) * ((b_ext - x) / L) ** 2
        return torch.where(x <= a_ext, torch.zeros_like(x),
               torch.where(x <= a,    cdf_left,
               torch.where(x <= b,    cdf_box,
               torch.where(x <= b_ext, cdf_right,
                                       torch.ones_like(x)))))

    def icdf(self, u):
        u, a, b = broadcast_all(u, self.a, self.b)
        L  = self.L.to(dtype=u.dtype, device=u.device)
        A  = self.A.to(dtype=u.dtype, device=u.device)
        B  = self.B.to(dtype=u.dtype, device=u.device)
        W  = b - a
        half_A = A / 2.0
        # Left tail: (a-L) + L * sqrt(2u/A)
        left_x  = (a - L) + L * torch.sqrt((2.0 * u / A).clamp(min=0.0))
        box_x   = a + (u - half_A) * W / B
        # Right tail: (b+L) - L * sqrt(2*(1-u)/A)
        right_x = (b + L) - L * torch.sqrt((2.0 * (1.0 - u) / A).clamp(min=0.0))
        return torch.where(u < half_A, left_x,
                           torch.where(u > 1.0 - half_A, right_x, box_x))

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.a.shape, device=self.a.device)
        return self.icdf(u)

    def sample_lhs(self, n_samples):
        sampler = qmc.LatinHypercube(d=len(self.a.flatten()), seed=42)
        u_samples = sampler.random(n_samples)
        u_tensor = torch.tensor(u_samples, dtype=torch.float32, device=self.a.device)
        return self.icdf(u_tensor)

    def mean(self):
        return 0.5 * (self.a + self.b)


class IndependentLinearTailed(CustomIndependent):
    Distribution = LinearTailed

    def sample_lhs(self, n_samples):
        return self.base_dist.sample_lhs(n_samples)


class ExponentialTailed(Distribution):
    """Uniform box with exponentially-decaying tails beyond the boundaries.

    Given sigma, the total tail probability mass A matches GaussianTailed(a, b, sigma),
    so all boundary-extrapolation variants can be compared on equal footing.

    Tail shape: density decays as exp(-d/tau) where d is distance from boundary and
    tau = A*W/(2*B). The distribution is continuous at [a, b] boundaries.
    Unlike LinearTailed, support extends to ±infinity.
    """
    arg_constraints = {
        'a': torch.distributions.constraints.real,
        'b': torch.distributions.constraints.dependent,
        'sigma': torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, a, b, sigma, validate_args=None):
        self.a, self.b, self.sigma = broadcast_all(a, b, sigma)
        if torch.any(self.a >= self.b):
            raise ValueError("`a` must be less than `b`.")
        W = self.b - self.a
        Z = math.sqrt(2 * math.pi) * self.sigma + W
        self.A = math.sqrt(2 * math.pi) * self.sigma / Z
        self.B = W / Z
        self.tau = self.A * W / (2.0 * self.B)   # exp scale = half the linear tail length
        self.box_density = self.B / W
        super().__init__(batch_shape=self.a.size(), validate_args=validate_args)

    def log_prob(self, x):
        x, a, b = broadcast_all(x, self.a, self.b)
        tau     = self.tau.to(dtype=x.dtype, device=x.device)
        log_box = torch.log(self.box_density.to(dtype=x.dtype, device=x.device))
        log_left  = log_box - (a - x) / tau
        log_right = log_box - (x - b) / tau
        return torch.where(x <= a, log_left,
                           torch.where(x >= b, log_right, log_box))

    def cdf(self, x):
        x, a, b = broadcast_all(x, self.a, self.b)
        tau = self.tau.to(dtype=x.dtype, device=x.device)
        A   = self.A.to(dtype=x.dtype, device=x.device)
        B   = self.B.to(dtype=x.dtype, device=x.device)
        W   = b - a
        # Left tail: (A/2) * exp((x-a)/tau)  -> 0 as x->-inf, A/2 at x=a
        cdf_left  = (A / 2.0) * torch.exp((x - a) / tau)
        cdf_box   = A / 2.0 + (B / W) * (x - a)
        # Right tail: 1 - (A/2) * exp(-(x-b)/tau)
        cdf_right = 1.0 - (A / 2.0) * torch.exp(-(x - b) / tau)
        return torch.where(x <= a, cdf_left,
                           torch.where(x >= b, cdf_right, cdf_box))

    def icdf(self, u):
        u, a, b = broadcast_all(u, self.a, self.b)
        tau    = self.tau.to(dtype=u.dtype, device=u.device)
        A      = self.A.to(dtype=u.dtype, device=u.device)
        B      = self.B.to(dtype=u.dtype, device=u.device)
        W      = b - a
        half_A = A / 2.0
        # Left tail: a + tau * log(2u/A)
        left_x  = a + tau * torch.log((2.0 * u / A).clamp(min=1e-38))
        box_x   = a + (u - half_A) * W / B
        # Right tail: b - tau * log(2*(1-u)/A)
        right_x = b - tau * torch.log((2.0 * (1.0 - u) / A).clamp(min=1e-38))
        return torch.where(u < half_A, left_x,
                           torch.where(u > 1.0 - half_A, right_x, box_x))

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.a.shape, device=self.a.device)
        return self.icdf(u)

    def sample_lhs(self, n_samples):
        sampler = qmc.LatinHypercube(d=len(self.a.flatten()), seed=42)
        u_samples = sampler.random(n_samples)
        u_tensor = torch.tensor(u_samples, dtype=torch.float32, device=self.a.device)
        return self.icdf(u_tensor)

    def mean(self):
        return 0.5 * (self.a + self.b)


class IndependentExponentialTailed(CustomIndependent):
    Distribution = ExponentialTailed

    def sample_lhs(self, n_samples):
        return self.base_dist.sample_lhs(n_samples)


class UniformTailed(Distribution):
    """Uniform distribution extended symmetrically beyond the original box.

    Effectively Uniform([a-delta, b+delta]) where delta = A*W/(2*B) is chosen so
    that the tail probability mass matches GaussianTailed(a, b, sigma). This is the
    simplest possible boundary extrapolation: flat density everywhere including tails.

    Note: the density is continuous at [a, b] boundaries (same flat density inside
    and outside the box), so there is no boundary discontinuity whatsoever.
    """
    arg_constraints = {
        'a': torch.distributions.constraints.real,
        'b': torch.distributions.constraints.dependent,
        'sigma': torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, a, b, sigma, validate_args=None):
        self.a, self.b, self.sigma = broadcast_all(a, b, sigma)
        if torch.any(self.a >= self.b):
            raise ValueError("`a` must be less than `b`.")
        W = self.b - self.a
        Z = math.sqrt(2 * math.pi) * self.sigma + W
        self.A = math.sqrt(2 * math.pi) * self.sigma / Z
        self.B = W / Z
        self.delta = self.A * W / (2.0 * self.B)      # extension = tau from ExponentialTailed
        self.box_density = self.B / W                # = 1 / (W + 2*delta) = 1/Z
        super().__init__(batch_shape=self.a.size(), validate_args=validate_args)

    def log_prob(self, x):
        x, a, b = broadcast_all(x, self.a, self.b)
        delta       = self.delta.to(dtype=x.dtype, device=x.device)
        log_density = torch.log(self.box_density.to(dtype=x.dtype, device=x.device))
        neg_inf     = torch.full_like(x, float('-inf'))
        in_support  = (x >= a - delta) & (x <= b + delta)
        return torch.where(in_support, log_density, neg_inf)

    def cdf(self, x):
        x, a, b = broadcast_all(x, self.a, self.b)
        delta   = self.delta.to(dtype=x.dtype, device=x.device)
        density = self.box_density.to(dtype=x.dtype, device=x.device)
        cdf_val = density * (x - (a - delta))
        return torch.clamp(cdf_val, min=0.0, max=1.0)

    def icdf(self, u):
        u, a, b = broadcast_all(u, self.a, self.b)
        delta   = self.delta.to(dtype=u.dtype, device=u.device)
        density = self.box_density.to(dtype=u.dtype, device=u.device)
        return (a - delta) + u / density

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.a.shape, device=self.a.device)
        return self.icdf(u)

    def sample_lhs(self, n_samples):
        sampler = qmc.LatinHypercube(d=len(self.a.flatten()), seed=42)
        u_samples = sampler.random(n_samples)
        u_tensor = torch.tensor(u_samples, dtype=torch.float32, device=self.a.device)
        return self.icdf(u_tensor)

    def mean(self):
        return 0.5 * (self.a + self.b)


class IndependentUniformTailed(CustomIndependent):
    Distribution = UniformTailed

    def sample_lhs(self, n_samples):
        return self.base_dist.sample_lhs(n_samples)
