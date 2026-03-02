import numpy as np
import matplotlib.pyplot as plt
from gp_kernels import rbf_kernel, gp_posterior

# Hyperparameters
sigma2 = 1.0
length_scale = 1.0
noise = 1e-8

# True function
def f(X):
    return np.sqrt(np.sum(X**2, axis=1))

dims = [1, 2, 3, 5, 10, 20] # dimensions to test
errors_slice = [] # initialise errors array 

c = 1.0 # constant we want to fix
x_grid = np.linspace(-1, 1, 500) # test points

for d in dims:
    # Training data
    X_train = np.random.uniform(-1, 1, (50, d))
    y_train = f(X_train)

    # 1D slice test set: (c, c, ..., c, x)
    X_test = np.full((x_grid.size, d), c)
    X_test[:, -1] = x_grid  # vary last coordinate

    y_true = f(X_test)

    # GP prediction (your function)
    mu_post, _ = gp_posterior(X_train, y_train, X_test, rbf_kernel)

    mae = np.mean(np.abs(y_true - mu_post))
    errors_slice.append(mae)

plt.figure(figsize=(6,4))
plt.plot(dims, errors_slice, marker='o')
plt.title("GP Regression Error vs Dimension")
plt.xlabel("Input dimension (d)")
plt.ylabel("Mean Absolute Error on Slice")
plt.grid(False)
plt.show()