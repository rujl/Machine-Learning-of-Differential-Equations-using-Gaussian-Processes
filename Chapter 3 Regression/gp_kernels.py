import numpy as np

# True function for reference
def f(x):
    return x * np.cos(x)

# Generate random numbers using the same seed for reproducibility

np.random.seed(1) # Initialise random seed
N = 5  # Number of training points
X_train = np.random.uniform(-np.pi, np.pi, N)[:, None] # Generate training points
y_train = f(X_train)

# Test points for plotting
X_test = np.linspace(-np.pi, np.pi, 500)[:, None]
y_true = f(X_test)

# Hyperparameters
sigma2 = 1.0
length_scale = 1.0
noise = 1e-8

# Kernels

def rbf_kernel(x1, x2, length_scale, sigma2):
    dists = np.subtract.outer(x1[:,0], x2[:,0])**2
    return sigma2 * np.exp(-0.5 * dists / length_scale**2)

def brownian_kernel(x1, x2, sigma2):
    return sigma2 * np.minimum.outer(x1[:,0], x2[:,0])

def matern32_kernel(x1, x2, length_scale, sigma2):
    dists = np.abs(np.subtract.outer(x1[:,0], x2[:,0]))
    sqrt3_d = np.sqrt(3) * dists / length_scale
    return sigma2 * (1 + sqrt3_d) * np.exp(-sqrt3_d)

# GP posterior
def gp_posterior(X_train, y_train, X_test, kernel_func):

    # Define our covariance matrices
    
    K_xx = kernel_func(X_train, X_train) + noise*np.eye(len(X_train))   # K(X,X) + σ_n²I (train-train)
    K_xxs = kernel_func(X_train, X_test)                                # K(X, X_*) (train-test)
    K_xsx = K_xxs.T                                                     # K(X_*, X) (test-train)
    K_xsxs = kernel_func(X_test, X_test)                                # K(X_*, X_*) (test-test)
    
    K_inv = np.linalg.inv(K_xx)                                         # [K(X,X)+σ_n²I]⁻¹ (invert train-train)

    mu_post = K_xsx @ K_inv @ y_train                                   # μ_* = K(X_*,X)[K+σ²I]⁻¹ y (eq. 2.23 from RasmussenWilliams)
    cov_post = K_xsxs - K_xsx @ K_inv @ K_xxs                           # cov_* = K(X_*,X_*) - ... (eq. 2.24 RasmussenWilliams)
    return mu_post, np.diag(cov_post)

# Run GP for each kernel
kernels = {
    'Squared Exponential': lambda x1,x2: rbf_kernel(x1,x2,length_scale,sigma2),
    'Brownian': lambda x1,x2: brownian_kernel(x1,x2,sigma2),
    'Matern 3/2': lambda x1,x2: matern32_kernel(x1,x2,length_scale,sigma2)
}
