import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# FOR RANDOM AND EVENLY-SPACED POINTS
# --- f(x) and exact solution ---
def f(x):
    return np.sin(2 * np.pi * x)

def u_exact(x):
    return np.sin(2 * np.pi * x)

# --- parameters ---
np.random.seed(1)
n_test = 100 # number of testing points
delta = 1/n_test
noise = 1e-7
sigma = 1 # kernel variance
ell = 1/5

# --- squared exponential kernel and its derivatives w.r.t r ---
#    k(r) = sigma^2 exp(-r² / (2l²))                 # = Kuu
#    d^2k/dr^2 = (r^4/l^4 - 1/l^2) k                 # = Kuf = Kfu
#    d^4k/dr^4 = (r^4/l^8 - 6 r^2/l^6 + 3/l^4) k     # = Kff
# --------------------------------------------------------------
def se_kernel(r, ell=ell, sigma=sigma):
    return sigma**2 * np.exp(-0.5 * (r**2) / (ell**2))

def se_d2(r, ell=ell, sigma = sigma):
    k = se_kernel(r, ell, sigma)
    return ((r**2) / (ell**4) - 1.0 / (ell**2)) * k

def se_d4(r, ell=ell, sigma = sigma):
    k = se_kernel(r, ell, sigma)
    return ((r**4) / (ell**8) - 6.0 * (r**2) / (ell**6) + 3.0 / (ell**4)) * k
# ----------------------------------------------------------
# Covariance blocks for u and f
#    f(x) = -alpha u''(x)
# ----------------------------------------------------------

# diffusion coefficient in -alpha u'' = f
alpha = 1/(4*np.pi**2)         
def K_uu(x1, x2):
    r = x1[:, None] - x2[None, :]
    return se_kernel(r)        # Cov[u(x), u(x')]

def K_uf(x1, x2):
    r = x1[:, None] - x2[None, :]
    return -alpha * se_d2(r)   # Cov[u(x), f(x')]

def K_fu(x1, x2):
    r = x1[:, None] - x2[None, :]
    return -alpha * se_d2(r)   # Cov[f(x), u(x')]

def K_ff(x1, x2):
    r = x1[:, None] - x2[None, :]
    return alpha**2 * se_d4(r) # Cov[f(x), f(x')]

# ---------------------------
# Training / test points
# ---------------------------
rng = np.random.default_rng(0)
# Interior forcing data f(x) = sin(2pix) (i.e., training points)
n_f = 8
# x_f = np.linspace(0, 1, n_f) # for evenly-spaced points
x_f = rng.random(n_f)          # for randomly-spaced points
y_f = f(x_f)                    

# test points (for f points)
x_test = np.linspace(0, 1, n_test)   

# boundary conditions u(0) = u(1) = 0
x_b = np.array([0.0, 1.0])
y_b = np.array([0.0, 0.0])

# ----------------------------------------------------------
# Build joint covariance K_obs for [u(boundary); f(interior)]
# ----------------------------------------------------------

Kuu_uu = K_uu(x_b, x_b)             # 2 x 2
Kuu_uf = K_uf(x_b, x_f)             # 2 x n_f
Kff_fu = K_fu(x_f, x_b)             # n_f x 2
Kff_ff = K_ff(x_f, x_f)             # n_f x n_f

top    = np.hstack([Kuu_uu, Kuu_uf])   # 2 x (2 + n_f)
bottom = np.hstack([Kff_fu, Kff_ff])   # n_f x (2 + n_f)
K_obs  = np.vstack([top, bottom])      # (2 + n_f) x (2 + n_f)

# Add jitter for numerical stability
K_obs += noise * np.eye(K_obs.shape[0])

# observation vector
y_obs = np.concatenate([y_b, y_f])   # length 2 + n_f

# ----------------------------------------------------------
# GP posterior for u(x*) given observations
# ----------------------------------------------------------

# Covariance between test points and observed [u(boundary); f(interior)]
K_star_u   = K_uu(x_test, x_b)       # n_test x 2
K_star_f   = K_uf(x_test, x_f)   # n_test x n_f
K_star = np.hstack([K_star_u, K_star_f])  # n_test x (2 + n_f)

# Prior covariance of test points with itself
K_star_star = K_uu(x_test, x_test)

# Solve (K_obs) * alpha = y_obs using Cholesky factorisation for better efficiency --> K_obs = LL^T 
L = np.linalg.cholesky(K_obs)
alpha_vec = scipy.linalg.cho_solve((L, True), y_obs) # Solve Lz = y and L^T alpha to get alpha = (K_obs)^{-1} y_obs

u_mean = K_star @ alpha_vec         # equivalent to eq 2.23
v = scipy.linalg.cho_solve((L, True), K_star.T)
u_cov = K_star_star - K_star @ v    # equivalent to eq 2.24
u_std = np.sqrt(np.clip(np.diag(u_cov), 0, np.inf))

# ---------------------------
# Error & Condition 
# ---------------------------
u_true = u_exact(x_test)
err = np.abs(u_true - u_mean)
mean_err = np.mean(err)
max_err = np.max(err)
print('Training points:', x_f)
print('Max absolute error:', max_err)
print("Mean absolute error:", mean_err)
print("Kobs condition number:", np.linalg.cond(K_obs))

# ---------------------------
# Plot 
# ---------------------------
plt.figure(figsize=(9,4))
plt.plot(x_test, u_mean, label='GP mean (SE kernel)')    # u ~ posterior mean
plt.plot(x_test, u_true, '--', label=r'True $u(x) = \sin(2\pi x)$')
plt.scatter(x_f, y_f, facecolors='none', edgecolors='k', s=40, label="f points")
plt.scatter(x_b, y_b, c='k', marker='x', label= r"boundary $u$")
plt.fill_between(
    x_test,
    u_mean - 1.96*u_std,
    u_mean + 1.96*u_std,
    color='C0',
    alpha=0.2,
    label='95% CI')

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(r"GP regression with randomly distributed points")
plt.legend()
plt.grid(True)
plt.show()
