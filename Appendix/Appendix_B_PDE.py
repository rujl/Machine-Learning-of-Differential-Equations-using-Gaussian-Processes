# A PDE WITH NO CLOSED-FORM EXPRESSION
# TRUE SOLUTION
import numpy as np
import matplotlib.pyplot as plt

def kappa_piecewise(x, theta1, theta2):
    x = np.asarray(x)
    k = np.empty_like(x, dtype=float)
    k[x <= 0.25] = 0.0
    k[(x > 0.25) & (x <= 0.50)] = theta1
    k[(x > 0.50) & (x <= 0.75)] = theta2
    k[x > 0.75] = 1.0
    return k

def thomas(a, b, c, d):
    n = len(b)
    cp = c.astype(float).copy()
    dp = d.astype(float).copy()
    bp = b.astype(float).copy()

    for i in range(1, n):
        w = a[i-1] / bp[i-1]
        bp[i] -= w * cp[i-1]
        dp[i] -= w * dp[i-1]

    x = np.empty(n)
    x[-1] = dp[-1] / bp[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i+1]) / bp[i]
    return x

def solve_elliptic_direct(theta1, theta2, n=400, u_left=0.0, u_right=2.0):
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]

    a = np.exp(kappa_piecewise(x, theta1, theta2))
    f = 4.0 * x

    a_half = 0.5 * (a[:-1] + a[1:])  # length n-1

    m = n - 2  # number of interior unknowns
    lower = np.zeros(m-1)
    diag  = np.zeros(m)
    upper = np.zeros(m-1)
    rhs   = f[1:-1].copy()

    for j in range(m):
        i = j + 1
        a_imh = a_half[i-1]
        a_iph = a_half[i]

        diag[j] = (a_imh + a_iph) / h**2
        if j > 0:
            lower[j-1] = -a_imh / h**2
        if j < m-1:
            upper[j] = -a_iph / h**2

    # Dirichlet BC contributions
    rhs[0]  -= (-a_half[0]  / h**2) * u_left
    rhs[-1] -= (-a_half[-1] / h**2) * u_right

    u_int = thomas(lower, diag, upper, rhs)

    u = np.empty(n)
    u[0] = u_left
    u[-1] = u_right
    u[1:-1] = u_int
    return x, u
theta1, theta2 = 0.098, 0.430
x, u = solve_elliptic_direct(theta1, theta2, n=400)

# USING GP-INFORMED METHOD
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# ------------------------------------------------------------
# 1. True solution from direct PDE solve
# ------------------------------------------------------------
theta1, theta2 = 0.098, 0.430
x, u_true = solve_elliptic_direct(theta1, theta2, n=400)

# coefficient and RHS on the same grid
a = np.exp(kappa_piecewise(x, theta1, theta2))
f = 4.0 * x


# ------------------------------------------------------------
# 2. Prior kernel on u(x)
# ------------------------------------------------------------
def rbf_kernel(X1, X2, sigma2=1.0, ell=0.18):
    X1 = np.atleast_2d(X1).reshape(-1, 1)
    X2 = np.atleast_2d(X2).reshape(-1, 1)
    sqdist = (X1 - X2.T) ** 2
    return sigma2 * np.exp(-0.5 * sqdist / ell**2)


# ------------------------------------------------------------
# 3. Build discrete PDE operator matrix L
#    so that (L u)_i ≈ -(a u_x)_x at interior points
# ------------------------------------------------------------
def build_pde_matrix(x, a):
    n = len(x)
    h = x[1] - x[0]
    a_half = 0.5 * (a[:-1] + a[1:])   # length n-1

    # maps full grid values u[0],...,u[n-1] to PDE values at interior points
    L = np.zeros((n - 2, n))

    for i in range(1, n - 1):
        row = i - 1
        a_imh = a_half[i - 1]
        a_iph = a_half[i]

        L[row, i - 1] = -a_imh / h**2
        L[row, i]     = (a_imh + a_iph) / h**2
        L[row, i + 1] = -a_iph / h**2

    return L


# ------------------------------------------------------------
# 4. Build observation operator C u = y
#    using:
#      - boundary conditions
#      - PDE collocation constraints
# ------------------------------------------------------------
def build_constraint_matrix(x, a, n_f=40):
    n = len(x)
    L = build_pde_matrix(x, a)

    # choose interior collocation points where PDE is enforced
    colloc_idx = np.linspace(1, n - 2, n_f, dtype=int)
    colloc_idx = np.unique(colloc_idx)

    # boundary matrix
    B = np.zeros((2, n))
    B[0, 0] = 1.0
    B[1, -1] = 1.0

    # PDE rows selected from L
    Lc = L[colloc_idx - 1, :]

    # stack boundary + PDE constraints
    C = np.vstack([B, Lc])

    # right-hand side observations
    y_obs = np.concatenate([
        np.array([0.0, 2.0]),      # u(0)=0, u(1)=2
        4.0 * x[colloc_idx]        # PDE RHS f(x)=4x
    ])

    return C, y_obs, colloc_idx


# ------------------------------------------------------------
# 5. PDE-informed GP posterior on the full grid
# ------------------------------------------------------------
def pde_informed_gp(x, a, sigma2=1.0, ell=0.18, n_f=40, obs_noise=1e-8):
    n = len(x)

    # prior covariance of u on the full grid
    K = rbf_kernel(x, x, sigma2=sigma2, ell=ell)
    K += 1e-10 * np.eye(n)   # tiny jitter

    # linear constraints C u = y
    C, y_obs, colloc_idx = build_constraint_matrix(x, a, n_f=n_f)

    # covariance of the observed linear functionals
    A = C @ K @ C.T + obs_noise * np.eye(C.shape[0])

    # solve for posterior mean
    cho = la.cho_factor(A, lower=True)
    alpha = la.cho_solve(cho, y_obs)

    KCT = K @ C.T
    mu_post = KCT @ alpha

    # posterior variance on the grid
    temp = la.cho_solve(cho, KCT.T)          # shape (m, n)
    var_post = np.diag(K) - np.sum(KCT * temp.T, axis=1)
    var_post = np.maximum(var_post, 0.0)
    std_post = np.sqrt(var_post)

    return mu_post, std_post, colloc_idx


# ------------------------------------------------------------
# 6. Run PDE-informed GP
# ------------------------------------------------------------
mu_pde, std_pde, colloc_idx = pde_informed_gp(
    x, a,
    sigma2=1.0,
    ell=0.18,
    n_f=40,
    obs_noise=1e-8
)
# ------------------------------------------------------------
# 8. Plot: true solution vs PDE-informed GP
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(x, u_true, label="True solution")
plt.plot(x, mu_pde, label="PDE-informed GP mean")
plt.fill_between(
    x,
    mu_pde - 2 * std_pde,
    mu_pde + 2 * std_pde,
    alpha=0.25,
    label=r"$\pm 2$ std"
)
plt.scatter(x[colloc_idx], u_true[colloc_idx], s=20, zorder=3, label="PDE collocation points")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("PDE-informed GP regression")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
