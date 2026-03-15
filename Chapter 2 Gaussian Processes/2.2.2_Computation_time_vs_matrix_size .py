import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, cholesky

# Squared-exponential kernel
def rbf(r, ell=1.0, sigma2=1.0):
    return sigma2 * np.exp(-(r**2) / (2 * ell**2))

# Phase 1: Precomputation: preparing each method for sampling

def cholesky_sqrt(cov):
    jitter = 1e-8
    return cholesky(cov + jitter*np.eye(cov.shape[0]), lower=True) # compute Cholesky decomposition

def spectral_sqrt(cov):
    eigvals, eigvecs = np.linalg.eigh(cov)
    sqrt_eigvals = np.sqrt(eigvals)
    return eigvecs, sqrt_eigvals # compute eigendecomposition

def fft_sqrt(c):
    g = np.concatenate([c, c[-2:0:-1]]) # find first row of larger ciruclant matrix
    d = np.maximum(np.real(np.fft.fft(g)), 0) # find eigenvalues using FFT
    N = len(g)
    return d, N


# Phase 2: Sampling

def chol_sample(L):
    z = np.random.randn(L.shape[0]) # take z from N(0,I)
    return L @ z #create sample

def spectral_sample(eigvecs, sqrt_eigvals):
    z = np.random.randn(len(sqrt_eigvals)) # take z from N(0,I)
    # f = Q * sqrt(Lambda) * (Q^T z)
    return eigvecs @ (sqrt_eigvals * (eigvecs.T @ z)) # generate a sample f = Q * sqrt(Lambda) * (Q^T z)

def fft_sample(d, N, n):
    # generate complex Gaussian noise
    xi = np.random.randn(N) + 1j * np.random.randn(N)

    # square root and inverse FFT
    Z = np.fft.ifft(np.sqrt(d) * xi) * np.sqrt(N)

    x = np.real(Z[:n]) # extract real
    y = np.imag(Z[:n]) # extract imaginary
    return x, y

# Timer
def time_repeat(func, *args, trials=5):
    start = time.time()
    for _ in range(trials):
        func(*args)
    return time.time() - start


# Experiment

n_values = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000] # different sizes of covariance matrix
trials_pre = 1        # setup timed once (per n)
trials_sample = 1     # per-method sampling trials (per n)

# initialise time arrays

times_pre_chol = []
times_pre_spec = []
times_pre_fft  = []

times_samp_chol = []
times_samp_spec = []
times_samp_fft  = []

for n in n_values:
    print(f"Running n={n} ...")
    x = np.arange(n)
    c = rbf(np.abs(x - x[0]))
    cov = toeplitz(c)

    # Phase 1: precomputation times
    t_chol_setup = time_repeat(cholesky_sqrt, cov, trials=trials_pre)
    t_spec_setup = time_repeat(spectral_sqrt, cov, trials=trials_pre)
    t_fft_setup  = time_repeat(fft_sqrt, c,   trials=trials_pre)

    times_pre_chol.append(t_chol_setup)
    times_pre_spec.append(t_spec_setup)
    times_pre_fft.append(t_fft_setup)

    # Precompute once to use for Phase 2 sampling timing
    L = cholesky_sqrt(cov)
    eigvecs, sqrt_eigvals = spectral_sqrt(cov)
    d, N = fft_sqrt(c)

    # Phase 2: sampling times
    t_chol_sample = time_repeat(chol_sample, L, trials=trials_sample)
    t_spec_sample = time_repeat(spectral_sample, eigvecs, sqrt_eigvals, trials=trials_sample)
    t_fft_sample  = time_repeat(fft_sample, d, N, n, trials=trials_sample)

    times_samp_chol.append(t_chol_sample)
    times_samp_spec.append(t_spec_sample)
    times_samp_fft.append(t_fft_sample)


# Plot: Phase 1 Precomputation

plt.figure(figsize=(8, 5))
plt.loglog(n_values, times_pre_chol, "s-", label="Cholesky (setup)")
plt.loglog(n_values, times_pre_spec, "o-", label="Spectral (setup)")
plt.loglog(n_values, times_pre_fft,  "^-", label="FFT (setup)")
plt.title("Phase 1: Precomputation cost")
plt.xlabel("Matrix size N")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(False, which="both")
plt.tight_layout()
plt.show()

# Plot: Phase 2 Sampling

plt.figure(figsize=(8, 5))
plt.loglog(n_values, times_samp_chol, "s-", label="Cholesky Sampling")
plt.loglog(n_values, times_samp_spec, "o-", label="Spectral Sampling")
plt.loglog(n_values, times_samp_fft,  "^-", label=f"FFT Sampling")
plt.title(f"Phase 2: Per-sample cost ({trials_sample} draw)")
plt.xlabel("Matrix size N")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(False, which="both")
plt.tight_layout()
plt.show()