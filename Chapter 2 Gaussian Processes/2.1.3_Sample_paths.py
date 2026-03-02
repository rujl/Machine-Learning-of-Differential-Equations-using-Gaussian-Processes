import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Kernels

def rbf_kernel(x, xprime, sigma=1.0, ell=0.2):
    """Squared Exponential (RBF): k = σ^2 exp(-||x-x'||^2 / (2ℓ^2))"""
    x = np.asarray(x)[:, None]
    xprime = np.asarray(xprime)[None, :]
    r2 = (x - xprime) ** 2
    return sigma**2 * np.exp(-0.5 * r2 / ell**2)

def matern32_kernel(x, xprime, sigma=1.0, ell=0.2):
    """Matérn ν=3/2: k = σ^2 (1 + √3 r/ℓ) exp(-√3 r/ℓ), r=|x-x'|"""
    x = np.asarray(x)[:, None]
    xprime = np.asarray(xprime)[None, :]
    r = np.abs(x - xprime)
    a = np.sqrt(3) * r / ell
    return sigma**2 * (1.0 + a) * np.exp(-a)

def brownian_kernel(x, xprime, sigma=1.0):
    """Brownian motion (Wiener) on x>=0: k = σ^2 min(x,x')"""
    x = np.asarray(x)[:, None]
    xprime = np.asarray(xprime)[None, :]
    return sigma**2 * np.minimum(x, xprime)

# Sampling from prior

def sample_gp_prior(x, kernel_func, n_samps=3, jitter=1e-10, **kparams):
    """Draw samples u ~ N(0, K) where K_ij = k(x_i, x_j)."""
    K = kernel_func(x, x, **kparams)
    K = K + jitter * np.eye(len(x))  # numerical stability
    L = np.linalg.cholesky(K)
    z = np.random.randn(len(x), n_samps)
    return L @ z

# Generate and plot sample paths ----------

if __name__ == "__main__":
    np.random.seed(0)

    x = np.linspace(0, 1, 400)

    # Hyperparameters
    sigma = 1.0
    ell_rbf = 0.15
    ell_m32 = 0.15

    rbf_samples = sample_gp_prior(x, rbf_kernel, n_samps=3, sigma=sigma, ell=ell_rbf)
    m32_samples = sample_gp_prior(x, matern32_kernel, n_samps=3, sigma=sigma, ell=ell_m32)
    bm_samples  = sample_gp_prior(x, brownian_kernel, n_samps=3, sigma=sigma)

    # Plot: Squared-exponential
    plt.figure(figsize=(10, 3))
    plt.plot(x, rbf_samples)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(False)
    plt.show()

    # Plot: Matérn 3/2
    plt.figure(figsize=(10, 3))
    plt.plot(x, m32_samples)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(False)
    plt.show()

    # Plot: Brownian Motion
    plt.figure(figsize=(10, 3))
    plt.plot(x, bm_samples)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(False)
    plt.show()
