import numpy as np
import matplotlib.pyplot as plt

# Target function
f = lambda x: x * np.cos(x)
noise = 1e-6

# Kernels
def se_kernel(x1, x2, l=1.0):
    r = np.abs(x1 - x2)
    return np.exp(-0.5 * (r / l)**2)

def matern32_kernel(x1, x2, l=1.0):
    r = np.abs(x1 - x2)
    factor = np.sqrt(3) * r / l
    return (1 + factor) * np.exp(-factor)

def brownian_kernel(x1, x2):
    return np.minimum(x1, x2)

# Covariance
def build_K(x1, x2, kernel):
    return np.array([[kernel(a, b) for b in x2] for a in x1])

# GP prediction
def gp_predict(x_train, y_train, x_test, kernel):
    K = build_K(x_train, x_train, kernel) + noise * np.eye(len(x_train))
    K_s = build_K(x_test, x_train, kernel)
    K_inv = np.linalg.inv(K)
    mean = K_s @ K_inv @ y_train
    return mean

# Test points, 300 in [-pi, pi]
x_test = np.linspace(-np.pi, np.pi, 300)
y_true = f(x_test)

kernels = {
    "SE": se_kernel,
    "Matérn 3/2": matern32_kernel,
    "Brownian": brownian_kernel
}

Ns = np.arange(3, 50, 2)
errors = {name: {"mae": [], "max": []} for name in kernels}

# Loop over N and compute errors
for N in Ns:
    x_train = np.random.uniform(-np.pi, np.pi, N)
    y_train = f(x_train)

    for name, kernel in kernels.items():
        y_pred = gp_predict(x_train, y_train, x_test, kernel)

        mae = np.mean(np.abs(y_true - y_pred))
        max_err = np.max(np.abs(y_true - y_pred))

        errors[name]["mae"].append(mae)
        errors[name]["max"].append(max_err)

# Plot mean absolute error
plt.figure(figsize=(8, 5))
for name in kernels:
    plt.semilogy(Ns, errors[name]["mae"], label=name)

plt.xlabel("N (training points)")
plt.ylabel("Mean Absolute Error (log scale)")
plt.title("Mean Error vs N")
plt.legend()
plt.grid(False)
plt.show()

# Plot max error
plt.figure(figsize=(8, 5))
for name in kernels:
    plt.semilogy(Ns, errors[name]["max"], label=name)

plt.xlabel("N (training points)")
plt.ylabel("Max Absolute Error (log scale)")
plt.title("Maximum Error vs N")
plt.legend()
plt.grid(False)
plt.show()