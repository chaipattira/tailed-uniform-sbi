import numpy as np
import torch
from scipy.stats import qmc
import symbolic_pofk.syren_new as syren_new


def syren_simulator(
        theta, L=1000, N=128, a=1.0):
    """Simulator: theta -> P(k)"""
    Om, h = theta

    # Set up k-centers
    kf = 2*np.pi/L
    knyq = np.pi*N/L
    kedges = np.arange(0, knyq, kf)
    kcenters = (kedges[:-1] + kedges[1:])/2

    # Also fixed (for now)
    As = 2.105  # 10^9 A_s
    Ob = 0.02242 / h ** 2
    ns = 0.9665
    w0 = -1.0
    wa = 0.0
    mnu = 0.0
    pk_syren_theory = syren_new.pnl_new_emulated(
        kcenters, As, Om, Ob, h, ns, mnu, w0, wa, a=a
    )

    # Compute cosmic variance noise
    var_single = np.abs(pk_syren_theory)**2
    Nk = L**3 * kcenters**2 * kf / (2*np.pi**2)
    var_mode = var_single * 2 / Nk
    std_mode = np.sqrt(var_mode)

    # Add noise to P(k)
    pk_w_noise = pk_syren_theory + std_mode * \
        np.random.randn(*pk_syren_theory.shape)
    return pk_w_noise


def sample_uniform_lhs(n_samples, param_ranges, device='cpu'):
    """
    Generate uniform samples using Latin Hypercube Sampling
    with different ranges per parameter.

    Args:
        n_samples (int): number of samples
        param_ranges (list of tuples): [(low1, high1), (low2, high2), ...]
        device (str): 'cpu' or 'cuda'
    """
    dim = len(param_ranges)
    sampler = qmc.LatinHypercube(d=dim, seed=42)
    u_samples = sampler.random(n_samples)  # in [0,1]^dim

    # scale each column to its parameter range
    samples = np.zeros_like(u_samples)
    for i, (low, high) in enumerate(param_ranges):
        samples[:, i] = low + (high - low) * u_samples[:, i]

    return torch.tensor(samples, dtype=torch.float32, device=device)
