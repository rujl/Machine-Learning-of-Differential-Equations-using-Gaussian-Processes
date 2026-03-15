import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# True function f: R^d → R
# ----------------------------
def f(X):
    return np.sum(np.abs(X)**2, axis=1, keepdims=True)  # ||x||^2

# ----------------------------
# RBF kernel
# ----------------------------
def rbf_kernel(X1, X2, length_scale=1.0, sigma2=1.0):
    dists = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
    return sigma2 * np.exp(-0.5 * dists / length_scale**2)

# ----------------------------
# GP posterior
# ----------------------------
def gp_posterior(X_train, y_train, X_test, kernel_func, noise=1e-8):
    K_xx = kernel_func(X_train, X_train) + noise*np.eye(len(X_train))
    K_xxs = kernel_func(X_train, X_test)
    K_xsx = K_xxs.T
    K_xsxs = kernel_func(X_test, X_test)
    K_inv = np.linalg.inv(K_xx)
    mu_post = K_xsx @ K_inv @ y_train
    cov_post = K_xsxs - K_xsx @ K_inv @ K_xxs
    return mu_post.ravel(), np.diag(cov_post)

# ----------------------------
# Generate training grid in d dimensions
# ----------------------------
def generate_grid(d, m):
    grid_1d = np.linspace(-np.pi, np.pi, m)
    mesh = np.meshgrid(*[grid_1d]*d, indexing='xy')
    X = np.stack([g.ravel() for g in mesh], axis=-1)
    return X

# ----------------------------
# Experiment: time vs dimension
# ----------------------------
m = 5                 # points per dimension
dims = range(1, 6)    # test d = 1,...,7 (beyond that gets big fast)
times = []
N_values = []

np.random.seed(0)

for d in dims:
    X_train = generate_grid(d, m)
    y_train = f(X_train)
    
    # Random test points (fixed number)
    X_test = np.random.uniform(-1, 1, size=(200, d))
    
    start = time.time()
    mu_post, _ = gp_posterior(X_train, y_train, X_test, rbf_kernel)
    end = time.time()
    
    elapsed = end - start
    times.append(elapsed)
    N_values.append(len(X_train))
    
# ----------------------------
# Plotting
# ----------------------------
plt.figure(figsize=(12,5))

# Time vs dimension
plt.subplot(1,2,1)
plt.plot(dims, times, 'o-')
plt.xlabel('Dimension d')
plt.ylabel('Computation time (s)')
plt.yscale('log')  # log scale
plt.grid(False)
plt.savefig("gp_computation_time_vs_dimension.eps", bbox_inches="tight")
plt.show()