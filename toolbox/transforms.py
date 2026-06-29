# ABOUTME: Degeneracy-aligned coordinate rotation for the (Omega_m, h) cosmology task.
# ABOUTME: Rotates log-space to align with Omega_m * h^2.563 = const (from degen_detector).

import numpy as np
import torch

# From degen_detector on pilot MCMC (uniform prior, theta=(0.32, 0.67)):
#   54.1057*log(Omega_m) + 138.6805*log(h) = const  (R²=1.00)
#   → Omega_m * h^2.563 ≈ const
_A = 54.1057107179177
_B = 138.680483354876
_NORM = np.sqrt(_A**2 + _B**2)

N1 = float(_A / _NORM)   # 0.36348  — weight on log(Omega_m)
N2 = float(_B / _NORM)   # 0.93160  — weight on log(h)
ALPHA = float(_B / _A)   # 2.5631   — degeneracy exponent


class DegenRotation:
    """
    Log-space rotation for the (Omega_m, h) cosmological degeneracy.

    Forward  (Omega_m, h) → (phi1, phi2):
        phi1 =  n1*log(Om) + n2*log(h)   [constrained by data, low variance]
        phi2 = -n2*log(Om) + n1*log(h)   [degeneracy direction, high variance]

    Inverse (phi1, phi2) → (Omega_m, h):
        Om = exp( n1*phi1 - n2*phi2 )
        h  = exp( n2*phi1 + n1*phi2 )

    The Jacobian of the forward map is 1/(Om*h), so:
        log|det J_forward| = -log(Om) - log(h)
        log|det J_inverse| =  log(Om) + log(h)

    Accepts numpy arrays or torch tensors; output type matches input.
    """

    n1: float = N1
    n2: float = N2
    alpha: float = ALPHA   # Omega_m * h^alpha = const

    def forward(self, Om, h):
        """(Om, h) → (phi1, phi2)."""
        if isinstance(Om, torch.Tensor):
            log_Om, log_h = torch.log(Om), torch.log(h)
        else:
            log_Om, log_h = np.log(Om), np.log(h)
        phi1 =  self.n1 * log_Om + self.n2 * log_h
        phi2 = -self.n2 * log_Om + self.n1 * log_h
        return phi1, phi2

    def inverse(self, phi1, phi2):
        """(phi1, phi2) → (Om, h)."""
        if isinstance(phi1, torch.Tensor):
            exp = torch.exp
        else:
            exp = np.exp
        Om = exp( self.n1 * phi1 - self.n2 * phi2)
        h  = exp( self.n2 * phi1 + self.n1 * phi2)
        return Om, h

    def forward_batch(self, theta):
        """theta: (..., 2) [Om, h] → (..., 2) [phi1, phi2]."""
        phi1, phi2 = self.forward(theta[..., 0], theta[..., 1])
        if isinstance(theta, torch.Tensor):
            return torch.stack([phi1, phi2], dim=-1)
        return np.stack([phi1, phi2], axis=-1)

    def inverse_batch(self, phi):
        """phi: (..., 2) [phi1, phi2] → (..., 2) [Om, h]."""
        Om, h = self.inverse(phi[..., 0], phi[..., 1])
        if isinstance(phi, torch.Tensor):
            return torch.stack([Om, h], dim=-1)
        return np.stack([Om, h], axis=-1)

    def log_abs_det_jacobian_inverse(self, phi1, phi2):
        """log|det J| of (phi1,phi2)→(Om,h). Add to log-prob when converting
        a density learned in rotated space back to physical space."""
        Om, h = self.inverse(phi1, phi2)
        if isinstance(phi1, torch.Tensor):
            return torch.log(Om) + torch.log(h)
        return np.log(Om) + np.log(h)

    def prior_bounds_rotated(self, om_range=(0.24, 0.40), h_range=(0.61, 0.73)):
        """Bounding box of the uniform prior box in (phi1, phi2) space.

        The rectangular (Om, h) prior maps to a rotated parallelogram in
        (phi1, phi2). This returns the axis-aligned bounding box, which is
        what you'd use as the tailed-uniform prior bounds in rotated space.

        Returns
        -------
        phi1_range : (float, float)
        phi2_range : (float, float)
        """
        Om_corners = np.array([om_range[0], om_range[0], om_range[1], om_range[1]])
        h_corners  = np.array([h_range[0],  h_range[1],  h_range[0],  h_range[1]])
        phi1s, phi2s = self.forward(Om_corners, h_corners)
        return (float(phi1s.min()), float(phi1s.max())), \
               (float(phi2s.min()), float(phi2s.max()))


# Singleton — import and use directly
degen_rotation = DegenRotation()


def get_rotated_prior_ranges(om_range=(0.24, 0.40), h_range=(0.61, 0.73)):
    """Return (phi1_range, phi2_range) for use in SBI prior construction."""
    return degen_rotation.prior_bounds_rotated(om_range, h_range)
