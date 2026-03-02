import numpy as np
import matplotlib.pyplot as plt

# RBF kernel
def rbf_kernel(X1, X2, lengthscale, sigma_f=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    # pairwise squared distances
    sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
    return (sigma_f**2) * np.exp(-0.5 * sqdist / (lengthscale**2))

# GP posterior mean
def gp_posterior_mean(X_train, y_train, X_test, lengthscale, sigma_f=1.0, noise=1e-6):
    K   = rbf_kernel(X_train, X_train, lengthscale, sigma_f) + noise * np.eye(X_train.shape[0])
    K_s = rbf_kernel(X_train, X_test,  lengthscale, sigma_f)  # (n_train, n_test)

    # Cholesky solve: K alpha = y
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))  # shape (n_train,1)

    mu_s = K_s.T @ alpha  # (n_test, n_train) @ (n_train,1) -> (n_test,1)
    return mu_s

# True function
def f_true(X):
    # Euclidean norm of each row, returned as (n,1)
    return np.linalg.norm(X, axis=1, keepdims=True)

rng = np.random.default_rng(0)

train_n = 20 # 20 training points
test_n  = 500 # 500 test points
dimensions   = [1, 2, 3, 4, 5, 6] # 6 dimensions
lengthscales = [0.1, 0.5, 1.0] # 3 different length scales

plt.figure(figsize=(7,5))

# For each length scale, compute GP mean and MAE across dimensions
for ell in lengthscales:
    mae_values = []
    for d in dimensions:
        X_train = rng.uniform(-np.pi, np.pi, size=(train_n, d))
        y_train = f_true(X_train)

        X_test  = rng.uniform(-np.pi, np.pi, size=(test_n, d))
        y_test  = f_true(X_test)

        mu_s = gp_posterior_mean(X_train, y_train, X_test, lengthscale=ell)  # (test_n,1)
        mae = np.mean(np.abs(mu_s - y_test))
        mae_values.append(mae)

    plt.plot(dimensions, mae_values, 'o-', lw=2, label=f'ℓ = {ell}')

plt.xlabel("Dimension d")
plt.ylabel("Mean Absolute Error")
plt.title("GP mean absolute error vs dimension for different lengthscales")
plt.xticks(dimensions)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()