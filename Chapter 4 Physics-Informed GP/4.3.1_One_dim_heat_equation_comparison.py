import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Symbolic kernel
# ------------------------------------------------------------
w, alpha = sp.symbols('w alpha')
x, y = sp.symbols('x y')

g = sp.exp(-0.5 * w * (x - y)**2)

kuu = g
kuf = -alpha * sp.diff(sp.diff(g, x), x)
kfu = -alpha * sp.diff(sp.diff(g, y), y)
kff = -alpha * sp.diff(sp.diff(kfu, x), x)

A = sp.Matrix([[kuu, kuf],
               [kfu, kff]]) * sp.exp(0.5 * w * (x - y)**2)
ht = sp.exp(-0.5 * w * (x - y)**2) * sp.simplify(A)

covariance_matrix = sp.lambdify((x, y, alpha, w), ht, 'numpy')


def block_kernel(Xa, Xb, row_idx, col_idx, alpha_val, w_val):
    K = np.zeros((len(Xa), len(Xb)))
    for i, xi in enumerate(Xa):
        for j, xj in enumerate(Xb):
            K2 = covariance_matrix(xi, xj, alpha_val, w_val)
            K[i, j] = K2[row_idx, col_idx]
    return K

# ------------------------------------------------------------
# 2. Parameters and boundary data
# ------------------------------------------------------------
alpha_val = 1/(4*np.pi**2)
w_val = 25
eps = 1e-12

# boundary u data
X_u = np.array([0.0, 1.0])
y_u = np.array([0.0, 0.0])

# ------------------------------------------------------------
# 3. GP posterior as a callable u_mean(x), var_u(x)
# ------------------------------------------------------------
def gp_posterior_u(X_u, y_u, X_f, y_f,
                   alpha_val=alpha_val, w_val=w_val, eps=eps):
    """Build GP and return posterior (mean, var) as a callable."""

    n_u = len(X_u)
    n_f = len(X_f)
    N = n_u + n_f

    if N == 0:
        raise ValueError("Need at least one training point")

    # build K_xx (handle cases n_f=0, n_u=0)
    if n_u > 0 and n_f > 0:
        K_uu = block_kernel(X_u, X_u, 0, 0, alpha_val, w_val)
        K_uf = block_kernel(X_u, X_f, 0, 1, alpha_val, w_val)
        K_fu = block_kernel(X_f, X_u, 1, 0, alpha_val, w_val)
        K_ff = block_kernel(X_f, X_f, 1, 1, alpha_val, w_val)
        K_xx = np.vstack([np.hstack([K_uu, K_uf]),
                          np.hstack([K_fu, K_ff])])
    elif n_u > 0:
        K_xx = block_kernel(X_u, X_u, 0, 0, alpha_val, w_val)
    else:  # only f data
        K_xx = block_kernel(X_f, X_f, 1, 1, alpha_val, w_val)

    K_tilde = K_xx + eps*np.eye(N)
    L = np.linalg.cholesky(K_tilde)

    y_train = np.concatenate([y_u, y_f]) if n_f > 0 else y_u

    beta = np.linalg.solve(L, y_train)
    gamma = np.linalg.solve(L.T, beta)

    def posterior(x_query):
        xs = np.atleast_1d(x_query)
        M = len(xs)

        K_starX = np.zeros((M, N))
        diag = np.zeros(M)

        for i, xx in enumerate(xs):
            K2 = covariance_matrix(xx, xx, alpha_val, w_val)
            diag[i] = K2[0, 0]

            cols = []
            if n_u > 0:
                kuu_row = block_kernel([xx], X_u, 0, 0, alpha_val, w_val)[0, :]
                cols.append(kuu_row)
            if n_f > 0:
                kuf_row = block_kernel([xx], X_f, 0, 1, alpha_val, w_val)[0, :]
                cols.append(kuf_row)

            K_starX[i, :] = np.concatenate(cols)

        u_mean = K_starX @ gamma
        v = np.linalg.solve(L, K_starX.T)
        var_u = diag - np.sum(v**2, axis=0)
        return u_mean, var_u

    return posterior

# ------------------------------------------------------------
# 4. Helper: max error vs exact solution sin(2πx)
# ------------------------------------------------------------
def gp_max_error(X_u, y_u, X_f, y_f):
    post = gp_posterior_u(X_u, y_u, X_f, y_f)
    X_star = np.linspace(0, 1, 400)
    u_mean, _ = post(X_star)
    u_exact = np.sin(2*np.pi*X_star)
    return np.max(np.abs(u_mean - u_exact))

# ------------------------------------------------------------
# 5. Helper: select f-points by maximal posterior variance
# ------------------------------------------------------------
def select_f_by_variance(n_f_target, X_u, y_u):
    X_f = np.array([])  # start with no f points
    y_f = np.array([])
    tol = 1e-6

    while len(X_f) < n_f_target:
        post = gp_posterior_u(X_u, y_u, X_f, y_f)
        X_cand = np.linspace(0, 1, 500)[1:-1]  # interior
        _, var_cand = post(X_cand)

        # avoid re-selecting an existing point
        if len(X_f) > 0:
            for xf in X_f:
                var_cand[np.abs(X_cand - xf) < tol] = -np.inf

        idx = np.argmax(var_cand)
        x_star = X_cand[idx]

        X_f = np.append(X_f, x_star)
        y_f = np.append(y_f, np.sin(2*np.pi*x_star))

    return X_f, y_f

# ------------------------------------------------------------
# Helper: condition number of K_tilde
# ------------------------------------------------------------
def gp_condition_number(X_u, X_f,
                        alpha_val=alpha_val, w_val=w_val, eps=eps):

    n_u = len(X_u)
    n_f = len(X_f)
    N = n_u + n_f

    if n_u > 0 and n_f > 0:
        K_uu = block_kernel(X_u, X_u, 0, 0, alpha_val, w_val)
        K_uf = block_kernel(X_u, X_f, 0, 1, alpha_val, w_val)
        K_fu = block_kernel(X_f, X_u, 1, 0, alpha_val, w_val)
        K_ff = block_kernel(X_f, X_f, 1, 1, alpha_val, w_val)

        K_xx = np.vstack([
            np.hstack([K_uu, K_uf]),
            np.hstack([K_fu, K_ff])
        ])

    elif n_u > 0:
        K_xx = block_kernel(X_u, X_u, 0, 0, alpha_val, w_val)
    else:
        K_xx = block_kernel(X_f, X_f, 1, 1, alpha_val, w_val)

    K_tilde = K_xx + eps * np.eye(N)

    return np.linalg.cond(K_tilde)
# ------------------------------------------------------------
# 6. Compare strategies over multiple n_f
# ------------------------------------------------------------
rng = np.random.default_rng(0)

n_f_values = np.array([1,3,5,10,15,20,25,30,35,38])
err_var = []
err_even = []
err_rand = []

cond_var = []
cond_even = []
cond_rand = []

for n_f in n_f_values:
    # variance-based points
    X_f_var, y_f_var = select_f_by_variance(n_f, X_u, y_u)
    e_var = gp_max_error(X_u, y_u, X_f_var, y_f_var)
    c_var = gp_condition_number(X_u, X_f_var)

    # evenly spaced points
    X_f_even = np.linspace(0, 1, n_f+2)[1:-1]
    y_f_even = np.sin(2*np.pi*X_f_even)
    e_even = gp_max_error(X_u, y_u, X_f_even, y_f_even)
    c_even = gp_condition_number(X_u, X_f_even)

    # random points
    X_f_rand = rng.random(n_f)
    y_f_rand = np.sin(2*np.pi*X_f_rand)
    e_rand = gp_max_error(X_u, y_u, X_f_rand, y_f_rand)
    c_rand = gp_condition_number(X_u, X_f_rand)

    err_var.append(e_var)
    err_even.append(e_even)
    err_rand.append(e_rand)

    cond_var.append(c_var)
    cond_even.append(c_even)
    cond_rand.append(c_rand)

    print(
        f"n_f={n_f:2d} | "
        f"err(var,even,rand)=({e_var:.1e},{e_even:.1e},{e_rand:.1e}) | "
        f"cond(var,even,rand)=({c_var:.1e},{c_even:.1e},{c_rand:.1e})"
    )
# ------------------------------------------------------------
# 7. Plot comparison (semilog-y)
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.semilogy(n_f_values, err_var,  '-o', label='Max-variance f points')
plt.semilogy(n_f_values, err_even, '-s', label='Evenly spaced f points')
plt.semilogy(n_f_values, err_rand, '-^', label='Random f points')

plt.xlabel('Number of f points (n_f)')
plt.ylabel('Max |u_mean(x) - sin(2πx)| (log scale)')
plt.title('Error vs number of f points for different placement strategies')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Plot condition number vs n_f
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(n_f_values, cond_var,  '-o', label='Max-variance f points')
plt.plot(n_f_values, cond_even, '-s', label='Evenly spaced f points')
plt.plot(n_f_values, cond_rand, '-^', label='Random f points')

plt.xlabel('Number of f points (n_f)')
plt.ylabel('Condition number of K_tilde (log scale)')
plt.title('Condition number vs number of f points')
plt.legend()
plt.tight_layout()
plt.show()

