
import torch
import ili
from toolbox.distributions import IndependentTailedNormal


def get_param_ranges():
    # Parameter ranges [Om, h]
    param_1_range = (0.24, 0.40)   # Om
    param_2_range = (0.61, 0.73)    # h
    param_ranges = [param_1_range, param_2_range]
    return param_ranges


def get_priors(device='cpu'):
    param_ranges = get_param_ranges()
    param_1_range, param_2_range = param_ranges  # Om, h

    param_1_width = param_1_range[1] - param_1_range[0]
    param_2_width = param_2_range[1] - param_2_range[0]

    # Scale sigma relative to parameter ranges
    sigma_scale = 0.1
    sigmas = [sigma_scale * (high - low) for low, high in param_ranges]

    # Prior
    prior_normal = ili.utils.IndependentNormal(
        loc=[param_1_range[0] + param_1_width/2,
             param_2_range[0] + param_2_width/2],
        scale=sigmas,
        device=device
    )

    # Create proposal distributions
    prior_uniform = ili.utils.Uniform(
        low=[param_1_range[0], param_2_range[0]],
        high=[param_1_range[1], param_2_range[1]],
        device=device
    )

    prior_tailed = IndependentTailedNormal(
        a=torch.tensor([param_1_range[0], param_2_range[0]],
                       dtype=torch.float32),
        b=torch.tensor([param_1_range[1], param_2_range[1]],
                       dtype=torch.float32),
        sigma=torch.tensor([sigmas[0], sigmas[1]], dtype=torch.float32),
    )
    return prior_normal, prior_uniform, prior_tailed
