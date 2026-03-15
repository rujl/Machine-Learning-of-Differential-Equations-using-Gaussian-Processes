import numpy as np
import matplotlib.pyplot as plt

# Define our true function f(x)=xcosx
def f(x):
    return x * np.cos(x)

# Generate random numbers using the same seed for reproducibility
np.random.seed(1)
N = 5
X_train = np.random.uniform(-np.pi, np.pi, N)[:, None]
y_train = f(X_train)

# Test points for plotting
X_test = np.linspace(-np.pi, np.pi, 500)[:, None]
y_true = f(X_test)

# Hyperparameters
sigma2 = 1.0
length_scale = 1.0
noise = 1e-8

# Define our kernel functions
def rbf_kernel(x1, x2, length_scale, sigma2):
    dists = np.subtract.outer(x1[:, 0], x2[:, 0])**2
    return sigma2 * np.exp(-0.5 * dists / length_scale**2)

def brownian_kernel(x1, x2, sigma2):
    return sigma2 * np.minimum.outer(x1[:, 0], x2[:, 0])

def matern32_kernel(x1, x2, length_scale, sigma2):
    dists = np.abs(np.subtract.outer(x1[:, 0], x2[:, 0]))
    sqrt3_d = np.sqrt(3) * dists / length_scale
    return sigma2 * (1 + sqrt3_d) * np.exp(-sqrt3_d)

# GP Regression function
def gp_posterior(X_train, y_train, X_test, kernel_func):
    K_xx = kernel_func(X_train, X_train) + noise * np.eye(len(X_train))
    K_xxs = kernel_func(X_train, X_test)
    K_xsx = K_xxs.T
    K_xsxs = kernel_func(X_test, X_test)

    K_inv = np.linalg.inv(K_xx)

    mu_post = K_xsx @ K_inv @ y_train
    cov_post = K_xsxs - K_xsx @ K_inv @ K_xxs
    return mu_post, np.diag(cov_post)

# Run GP for each kernel
kernels = {
    'Squared Exponential': lambda x1, x2: rbf_kernel(x1, x2, length_scale, sigma2),
    'Brownian': lambda x1, x2: brownian_kernel(x1, x2, sigma2),
    'Matern 3/2': lambda x1, x2: matern32_kernel(x1, x2, length_scale, sigma2)
}

plt.figure(figsize=(10, 8))

for i, (name, kernel) in enumerate(kernels.items(), 1):
    mu, var = gp_posterior(X_train, y_train, X_test, kernel)
    mu = mu.ravel()
    std = np.sqrt(var).ravel()

    plt.subplot(3, 1, i)
<<<<<<< HEAD
    plt.plot(X_test, y_true, 'k--', label='True function')
    plt.plot(X_train, y_train, 'ro', label='Train points')
    plt.plot(X_test, mu, 'b', label='GP mean')
    plt.fill_between(X_test[:,0], mu - 1.96*std, mu + 1.96*std, alpha=0.2, label='95% CI')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    plt.title(f'GP Regression with {name} Kernel')
=======
    plt.plot(X_test[:, 0], y_true[:, 0], 'k--', label='True function')
    plt.plot(X_train[:, 0], y_train[:, 0], 'o', color='red', label='Train points')
    plt.plot(X_test[:, 0], mu, color='orange', linewidth=2, label='GP mean')
    plt.fill_between(
        X_test[:, 0],
        mu - 1.96 * std,
        mu + 1.96 * std,
        color="#c6dbef",
        label='95% CI'
    )
    #plt.title(f'GP Regression with {name} Kernel')
    plt.savefig(f"{name}.eps", format="eps", bbox_inches="tight")
>>>>>>> c6dbdcb139b545af42e5376530082dbe4d715a68
    plt.legend()
    plt.grid(False)



plt.tight_layout()
plt.show()