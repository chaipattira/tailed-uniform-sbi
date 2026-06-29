# ABOUTME: Prior distributions for cosmology power spectrum inference
# ABOUTME: Provides uniform, normal, and tailed-uniform priors for Om and h parameters
# ABOUTME: Also provides rotated priors aligned with the Omega_m * h^2.563 degeneracy direction

import torch
import ili
from toolbox.distributions import IndependentGaussianTailed
from toolbox.transforms import degen_rotation


def get_param_ranges():
    # Parameter ranges [Om, h]
    param_1_range = (0.24, 0.40)   # Om
    param_2_range = (0.61, 0.73)   # h
    return [param_1_range, param_2_range]


def get_priors(device='cpu'):
    param_ranges = get_param_ranges()
    param_1_range, param_2_range = param_ranges

    param_1_width = param_1_range[1] - param_1_range[0]
    param_2_width = param_2_range[1] - param_2_range[0]

    sigma_scale = 0.1
    sigmas = [sigma_scale * (high - low) for low, high in param_ranges]

    prior_normal = ili.utils.IndependentNormal(
        loc=[param_1_range[0] + param_1_width/2,
             param_2_range[0] + param_2_width/2],
        scale=sigmas,
        device=device
    )

    prior_uniform = ili.utils.Uniform(
        low=[param_1_range[0], param_2_range[0]],
        high=[param_1_range[1], param_2_range[1]],
        device=device
    )

    prior_tailed_uniform = IndependentGaussianTailed(
        a=torch.tensor([param_1_range[0], param_2_range[0]], dtype=torch.float32),
        b=torch.tensor([param_1_range[1], param_2_range[1]], dtype=torch.float32),
        sigma=torch.tensor([sigmas[0], sigmas[1]], dtype=torch.float32),
    )
    return prior_normal, prior_uniform, prior_tailed_uniform


def get_rotated_priors(device='cpu', sigma_scale=0.1):
    """Priors in (phi1, phi2) rotated space aligned with the degeneracy direction.

    phi1 is the data-constrained direction (Omega_m * h^2.563 = const).
    phi2 is the degeneracy direction — the tailed-uniform smoothing here is
    what the reviewer comment is about.

    Returns
    -------
    prior_uniform_rot : Uniform prior in (phi1, phi2)
    prior_tailed_rot  : Tailed-uniform prior in (phi1, phi2)
    phi1_range, phi2_range : bounds used
    """
    phi1_range, phi2_range = degen_rotation.prior_bounds_rotated()

    sigmas_rot = [
        sigma_scale * (phi1_range[1] - phi1_range[0]),
        sigma_scale * (phi2_range[1] - phi2_range[0]),
    ]

    prior_uniform_rot = ili.utils.Uniform(
        low=[phi1_range[0], phi2_range[0]],
        high=[phi1_range[1], phi2_range[1]],
        device=device
    )

    prior_tailed_rot = IndependentGaussianTailed(
        a=torch.tensor([phi1_range[0], phi2_range[0]], dtype=torch.float32),
        b=torch.tensor([phi1_range[1], phi2_range[1]], dtype=torch.float32),
        sigma=torch.tensor(sigmas_rot, dtype=torch.float32),
    )

    return prior_uniform_rot, prior_tailed_rot, phi1_range, phi2_range
