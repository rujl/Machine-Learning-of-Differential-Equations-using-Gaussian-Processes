import numpy as np
import matplotlib.pyplot as plt

# True function for reference
def f(x):
    return x * np.cos(x)

# Reproducible training data
np.random.seed(1)
N = 5
X_train = np.random.uniform(-np.pi, np.pi, N)[:, None]
y_train = f(X_train).ravel()

# Test points for plotting
X_test = np.linspace(-np.pi, np.pi, 500)[:, None]
y_true = f(X_test).ravel()

# Hyperparameters
sigma2 = 1.0
length_scale = 1.0
noise = 1e-8

# Squared Exponential (RBF) kernel
def rbf_kernel(x1, x2, length_scale, sigma2):
    dists = np.subtract.outer(x1[:, 0], x2[:, 0])**2
    return sigma2 * np.exp(-0.5 * dists / length_scale**2)

# GP posterior
def gp_posterior_se(X_train, y_train, X_test):
    # Build covariance matrix
    K_xx = rbf_kernel(X_train, X_train, length_scale, sigma2) + noise * np.eye(len(X_train))
    K_xxs = rbf_kernel(X_train, X_test, length_scale, sigma2)
    K_xsx = K_xxs.T
    K_xsxs = rbf_kernel(X_test, X_test, length_scale, sigma2)

    L = np.linalg.cholesky(K_xx)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    mu_post = K_xsx @ alpha # posterior mean
    v = np.linalg.solve(L, K_xxs)
    cov_post = K_xsxs - v.T @ v # posterior covariance

    return mu_post, np.diag(cov_post)

# Compute posterior
mu, var = gp_posterior_se(X_train, y_train, X_test)
mu = mu.ravel()
std = np.sqrt(var).ravel()

# Plot (single figure)
plt.figure(figsize=(9, 4))
plt.plot(X_test[:, 0], y_true, "k--", label="True function")
plt.plot(X_train[:, 0], y_train, "ro", label="Train points")
plt.plot(X_test[:, 0], mu, "b", label="GP mean")
plt.fill_between(X_test[:, 0], mu - 1.96 * std, mu + 1.96 * std, alpha=0.2, label="95% CI")
plt.title("GP Posterior with Squared Exponential Kernel")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()