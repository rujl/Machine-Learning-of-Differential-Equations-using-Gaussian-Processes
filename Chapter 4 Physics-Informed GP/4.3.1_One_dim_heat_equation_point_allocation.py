import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# MAX ERROR VS POINT ALLOCATION

# parameters 
alpha_val = 1/(4*np.pi**2)
w_val = 25  # 1/ℓ^2
eps = 1e-12

def gp_max_error(n_u, n_f, alpha_val=alpha_val, w_val=w_val, eps=eps):
    """
    Build the GP with n_u 'u' points and n_f 'f' points and return the max error
    in u(x) on [0, 1].
    
    We place u-points evenly on [0,1], and f-points evenly in the interior (0,1).
    The true solution is u(x) = sin(2πx).
    """
    assert n_u + n_f > 0, "Need at least one training point"

    # --- training locations and values ---
    # u-points: anywhere in [0,1]
    if n_u > 0:
        X_u = np.linspace(0.0, 1.0, n_u)
        y_u = np.sin(2 * np.pi * X_u)
    else:
        X_u = np.empty((0,))
        y_u = np.empty((0,))

    # f-points: interior points (avoid exact boundary)
    if n_f > 0:
        X_f = np.linspace(0.0, 1.0, n_f + 2)[1:-1]  # strip endpoints
        y_f = np.sin(2 * np.pi * X_f)
    else:
        X_f = np.empty((0,))
        y_f = np.empty((0,))

    # concatenate training targets: (u, f)
    y_train = np.concatenate([y_u, y_f])
    n_u = len(X_u)
    n_f = len(X_f)
    N = n_u + n_f

    # --- build K_xx (training-training covariance) ---
    if n_u > 0 and n_f > 0:
        K_uu = block_kernel(X_u, X_u, 0, 0, alpha_val, w_val)
        K_uf = block_kernel(X_u, X_f, 0, 1, alpha_val, w_val)
        K_fu = block_kernel(X_f, X_u, 1, 0, alpha_val, w_val)
        K_ff = block_kernel(X_f, X_f, 1, 1, alpha_val, w_val)

        top = np.hstack([K_uu, K_uf])
        bottom = np.hstack([K_fu, K_ff])
        K_xx = np.vstack([top, bottom])
    elif n_u > 0:
        # only u data
        K_xx = block_kernel(X_u, X_u, 0, 0, alpha_val, w_val)
    else:
        # only f data
        K_xx = block_kernel(X_f, X_f, 1, 1, alpha_val, w_val)

    # add jitter and Cholesky
    K_tilde = K_xx + eps * np.eye(N)
    L = np.linalg.cholesky(K_tilde)

    # --- test points ---
    X_star = np.linspace(0, 1, 100)
    M = len(X_star)

    K_starX = np.zeros((M, N))
    k_star_diag = np.zeros(M)

    for i, xs in enumerate(X_star):
        # prior variance of u at xs
        K2_xx = covariance_matrix(xs, xs, alpha_val, w_val)
        k_star_diag[i] = K2_xx[0, 0]  # k_uu(xs, xs)

        cols = []
        if n_u > 0:
            kuu_row = block_kernel([xs], X_u, 0, 0, alpha_val, w_val)[0, :]
            cols.append(kuu_row)
        if n_f > 0:
            kuf_row = block_kernel([xs], X_f, 0, 1, alpha_val, w_val)[0, :]
            cols.append(kuf_row)

        if cols:
            K_starX[i, :] = np.concatenate(cols)

    # mean: K_*X (K_xx + eps I)^{-1} y
    beta = np.linalg.solve(L, y_train)      # L^{-1} y
    gamma = np.linalg.solve(L.T, beta)      # L^{-T} L^{-1} y
    u_mean = K_starX @ gamma

    # variance: k(x*,x*) - v^T v with v = L^{-1} K_X*^T
    v = np.linalg.solve(L, K_starX.T)
    var_u = k_star_diag - np.sum(v**2, axis=0)
    # std_u = np.sqrt(var_u)  # not needed for error, but available

    # exact solution and max error
    u_exact = np.sin(2 * np.pi * X_star)
    max_err = np.max(np.abs(u_mean - u_exact))
    return max_err


# -----------------------------------------------------------
# Scan over allocations with a total budget of 20 points
# -----------------------------------------------------------
budget = 40
n_u_list = []
n_f_list = []
max_errors = []

for n_f in range(0, budget + 1):
    n_u = budget - n_f
    max_err = gp_max_error(n_u, n_f)
    n_u_list.append(n_u)
    n_f_list.append(n_f)
    max_errors.append(max_err)
    print(f"n_u = {n_u:2d}, n_f = {n_f:2d}, max error = {max_err:.4e}")

# Plot max error vs number of u points
# Plot max error vs number of u points (semilog-y)
plt.figure(figsize=(7, 4))

# Avoid issues if any max_errors are exactly zero
min_pos = min([e for e in max_errors if e > 0])
safe_errors = [e if e > 0 else 0.5 * min_pos for e in max_errors]

plt.semilogy(n_u_list, safe_errors, marker='o')
plt.xlabel("Number of u points")
plt.ylabel("Max |u_mean(x) - sin(2πx)| (log scale)")
plt.title(f"Max error vs allocation of {budget} total points")
plt.grid(True, which='both')
plt.tight_layout()
plt.show()
