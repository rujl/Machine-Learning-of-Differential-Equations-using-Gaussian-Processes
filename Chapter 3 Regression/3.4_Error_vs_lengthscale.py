import numpy as np
import matplotlib.pyplot as plt

# RBF kernel
def rbf_kernel(X1, X2, lengthscale, sigma_f=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
    return (sigma_f**2) * np.exp(-0.5 * sqdist / (lengthscale**2))

# GP posterior mean
def gp_posterior_mean(X_train, y_train, X_test, lengthscale, sigma_f=1.0, noise=1e-6):
    K   = rbf_kernel(X_train, X_train, lengthscale, sigma_f) + noise * np.eye(X_train.shape[0])
    K_s = rbf_kernel(X_train, X_test,  lengthscale, sigma_f)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu_s = K_s.T @ alpha
    return mu_s

# True function
def f_true(X):
    return np.linalg.norm(X, axis=1, keepdims=True)

rng = np.random.default_rng(0)

train_n = 20
lengthscales = [0.1, 0.5, 1.0]

# Fix dimension
d = 3

# x-axis grid
x_grid = np.linspace(-np.pi, np.pi, 500)

# Training data
X_train = rng.uniform(-np.pi, np.pi, size=(train_n, d))
y_train = f_true(X_train)

plt.figure(figsize=(7,5))

for ell in lengthscales:

    # Construct test slice: (c, c, ..., x)
    c = 1.0
    X_test = np.full((x_grid.size, d), c)
    X_test[:, -1] = x_grid

    y_true = f_true(X_test)

    mu_s = gp_posterior_mean(X_train, y_train, X_test, lengthscale=ell)

    error = np.abs(mu_s - y_true).ravel()

    plt.plot(x_grid, error, lw=2, label=f'ℓ = {ell}')

plt.xlabel(r"$x \in [-\pi,\pi]$")
plt.ylabel("Absolute Error")
plt.title("GP approximation error along a 1D slice")
plt.yscale("log")   # log scale for error
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()