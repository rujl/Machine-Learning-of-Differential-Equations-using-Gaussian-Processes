import numpy as np
import matplotlib.pyplot as plt

# True function for reference
def f(x):
    return x * np.cos(x)

# Test points
X_test = np.linspace(-np.pi, np.pi, 500)[:, None]
y_true = f(X_test).ravel()

# Hyperparameters
sigma2 = 1.0
length_scale = 1.0

# RBF kernel
def rbf_kernel(x1, x2, length_scale, sigma2):
    dists = np.subtract.outer(x1[:, 0], x2[:, 0])**2
    return sigma2 * np.exp(-0.5 * dists / length_scale**2)

# GP prior: one sample draw
def gp_prior_one_draw(X_test, jitter=1e-10):
    K = rbf_kernel(X_test, X_test, length_scale, sigma2) + jitter * np.eye(len(X_test))
    sample = np.random.multivariate_normal(np.zeros(len(X_test)), K)  # (n_test,)
    std = np.sqrt(np.diag(K))  # (n_test,)
    return sample, std

np.random.seed(1)
prior_sample, std = gp_prior_one_draw(X_test)

x = X_test.ravel()

plt.figure(figsize=(9, 4))

# 95% prior CI around mean
plt.fill_between(x, -1.96*std, 1.96*std, alpha=0.2, label='95% CI')
plt.plot(x, np.zeros_like(x), 'k-', lw=1, label='Prior mean ')

# One prior sample
plt.plot(x, prior_sample, 'b-', lw=1.5, label='One prior sample')

# show true function as reference
plt.plot(x, y_true, 'k--', alpha=0.6, label='True function')

plt.title('GP Prior with Squared Exponential Kernel')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()