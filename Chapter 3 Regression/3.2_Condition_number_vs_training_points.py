import numpy as np
import matplotlib.pyplot as plt

# Squared exponential kernel
def rbf_kernel(x1, x2, length_scale=0.5, sigma_f=1.0):
    x1 = x1[:, None]
    x2 = x2[None, :]
    sqdist = (x1 - x2)**2
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Matern kernel v=3/2
def matern32_kernel(x1, x2, length_scale=0.5, sigma_f=1.0):
    x1 = x1[:, None]
    x2 = x2[None, :]
    d = np.abs(x1 - x2)
    sqrt3_d = np.sqrt(3) * d / length_scale
    return sigma_f**2 * (1 + sqrt3_d) * np.exp(-sqrt3_d)

# Brownian motion kernel
def brownian_kernel(x1, x2, sigma_f=1.0):
    # Shift inputs from [-1,1] to [0,2] to make min(x,x') well-defined
    x1 = (x1 + 1)[:, None]
    x2 = (x2 + 1)[None, :]
    return sigma_f**2 * np.minimum(x1, x2)

# Hyperparameters
np.random.seed(42)
N_max = 30
length_scale = 0.5
sigma_f = 1.0

# Initialise condition number arrays
cond_rbf = []
cond_matern = []
cond_brownian = []

# Loop to find condition numbers
for N in range(1, N_max+1):
    X = np.random.uniform(-1, 1, size=N)

    K_rbf = rbf_kernel(X, X, length_scale, sigma_f)
    K_matern = matern32_kernel(X, X, length_scale, sigma_f)
    K_brown = brownian_kernel(X, X, sigma_f)

    # If we make K more 'singular' we see our condition number decrease as the ratio of max/min eigenvalues is larger
    
    cond_rbf.append(np.linalg.cond(K_rbf))
    cond_matern.append(np.linalg.cond(K_matern))
    cond_brownian.append(np.linalg.cond(K_brown))

# Plot
plt.figure(figsize=(8,5))
plt.semilogy(range(1, N_max+1), cond_rbf, label="RBF kernel")
plt.semilogy(range(1, N_max+1), cond_matern, label="Matérn (ν=3/2)")
plt.semilogy(range(1, N_max+1), cond_brownian, label="Brownian motion")
plt.xlabel("Number of training points N")
plt.ylabel("Condition number (log scale)")
plt.title("Condition number vs N for different kernels")
plt.grid(False, which="both")
plt.legend()
plt.tight_layout()
plt.show()