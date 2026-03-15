import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# METHOD 1: SINGLE-OUTPUT GP
# ------------------------------------------------------------
# GP kernel
# ------------------------------------------------------------
def rbf_1d(X1, X2, ell=0.2, sigma2=1.0):
    X1 = np.atleast_2d(X1).reshape(-1,1)
    X2 = np.atleast_2d(X2).reshape(-1,1)
    sqdist = (X1 - X2.T)**2
    return sigma2*np.exp(-0.5*sqdist/ell**2)

ell_th = 0.2
sigma2 = 1.0
noise = 1e-8

x0 = 0.3
# ------------------------------------------------------------
# Method 1 predictor
# ------------------------------------------------------------
def method1_predict(theta_train, y_train, theta_star):

    K = rbf_1d(theta_train, theta_train, ell_th, sigma2) + noise*np.eye(len(theta_train))
    L = la.cholesky(K, lower=True)

    alpha = la.cho_solve((L, True), y_train)

    K_star = rbf_1d(theta_star, theta_train, ell_th, sigma2)
    mu = K_star @ alpha

    K_ss = rbf_1d(theta_star, theta_star, ell_th, sigma2)
    V = la.cho_solve((L, True), K_star.T)

    cov = K_ss - K_star @ V
    std = np.sqrt(np.clip(np.diag(cov),0,np.inf))

    return mu, std

# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------
theta_train = np.linspace(0.1,2,6)

y_train = []

for th1 in theta_train:
    x, u = solve_elliptic_direct(th1, 0.43, n=400)
    y_train.append(np.interp(x0, x, u))

y_train = np.array(y_train)


# ------------------------------------------------------------
# Prediction grid
# ------------------------------------------------------------
theta_star = np.linspace(0.1,2,200)

mu, std = method1_predict(theta_train, y_train, theta_star)


# ------------------------------------------------------------
# True curve
# ------------------------------------------------------------
u_true = []

for th1 in theta_star:
    x, u = solve_elliptic_direct(th1, 0.43, n=400)
    u_true.append(np.interp(x0, x, u))

u_true = np.array(u_true)
# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(10,4))

plt.plot(theta_star, u_true, label="True solution")
plt.plot(theta_star, mu, label="GP mean")

plt.fill_between(
    theta_star,
    mu-2*std,
    mu+2*std,
    alpha=0.3,
    label="±2 std"
)

plt.scatter(theta_train, y_train, color="black", label="Training points")

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$u(0.3,\theta)$")
plt.title("Method 1: Theta-only GP")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------
y_train = []

for th2 in theta_train:
    x, u = solve_elliptic_direct(0.098, th2, n=400)
    y_train.append(np.interp(x0, x, u))

y_train = np.array(y_train)

# ------------------------------------------------------------
# Prediction grid
# ------------------------------------------------------------
mu1, std = method1_predict(theta_train, y_train, theta_star)
# ------------------------------------------------------------
# True curve
# ------------------------------------------------------------
u_true = []

for th2 in theta_star:
    x, u = solve_elliptic_direct(0.098, th2, n=400)
    u_true.append(np.interp(x0, x, u))

u_true = np.array(u_true)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(10,4))

plt.plot(theta_star, u_true, label="True solution")
plt.plot(theta_star, mu1, label="GP mean")

plt.fill_between(
    theta_star,
    mu1-2*std,
    mu1+2*std,
    alpha=0.3,
    label="±2 std"
)

plt.scatter(theta_train, y_train, color="black", label="Training points")

plt.xlabel(r"$\theta_2$")
plt.ylabel(r"$u(0.3,\theta)$")
plt.title("Method 1: Theta-only GP")
plt.legend()
plt.grid()
plt.show()

rmse = np.sqrt(np.mean(mu1 - u_true)**2)

# METHOD 2: MULTI-OUTPUT GP
# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
ell_x = 0.15
ell_th = 0.6
sigma2 = 1.0
noise = 1e-8

x0 = 0.3

theta_train = np.linspace(0.1, 2, 6)
theta_test  = np.linspace(0.1, 2.0, 200)
x_sensors = np.linspace(0.1, 0.9, 6)

def method2_train(theta_train, x_sensors):
    m = len(theta_train)
    N = len(x_sensors)

    y_list = []

    for th in theta_train:
        x, u = solve_elliptic_direct(th, 0.43, n=400)
        u_sensors = np.interp(x_sensors, x, u)
        y_list.append(u_sensors)

    y = np.concatenate(y_list)

    Kx = rbf_1d(x_sensors, x_sensors, ell_x, 1.0)
    Kt = rbf_1d(theta_train, theta_train, ell_th, sigma2)
    K = np.kron(Kt, Kx) + noise*np.eye(m*N)

    L = la.cholesky(K, lower=True)
    alpha = la.cho_solve((L, True), y)

    return L, alpha

def method2_predict(theta_star, x_star, theta_train, x_sensors, L, alpha):
    theta_star = np.asarray(theta_star).ravel()

    kt = rbf_1d(theta_star, theta_train, ell_th, sigma2)
    kx = rbf_1d([x_star], x_sensors, ell_x, 1.0)
    K_star = np.kron(kt, kx)

    mu = K_star @ alpha

    k_theta_ss = rbf_1d(theta_star, theta_star, ell_th, sigma2)
    k_x_ss = rbf_1d([x_star], [x_star], ell_x, 1.0)
    K_ss = k_x_ss[0,0] * k_theta_ss

    V = la.cho_solve((L, True), K_star.T)
    cov = K_ss - K_star @ V
    std = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    return mu, std
# ------------------------------------------------------------
# Train GP
# ------------------------------------------------------------
L, alpha = method2_train(theta_train, x_sensors)
# ------------------------------------------------------------
# Predict slice u(x0, θ)
# ------------------------------------------------------------
mu, std = method2_predict(theta_test, x0,
                           theta_train,
                           x_sensors,
                           L, alpha)
# ------------------------------------------------------------
# True solution slice
# ------------------------------------------------------------
true_vals = []

for th1 in theta_test:
    x, u = solve_elliptic_direct(th1, 0.43, n=400)
    true_vals.append(np.interp(x0, x, u))

true_vals = np.array(true_vals)
# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(10,4))

plt.plot(theta_test, true_vals, label="True solution")
plt.plot(theta_test, mu, label="GP mean")
plt.fill_between(theta_test,
                 mu-2*std,
                 mu+2*std,
                 alpha=0.3)

# training points
y_train= []
for th1 in theta_train:
    x, u = solve_elliptic_direct(th1, 0.43, n=400)
    y_train.append(np.interp(x0, x, u))

plt.scatter(theta_train, y_train,
            color="black",
            label="Training θ")

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$u(x_0,\theta)$")
plt.title("Method 2 (Separable GP)")
plt.legend()
plt.grid()
plt.show()

rmse = np.sqrt(np.mean(mu1 - u_true)**2)
print(f"Method 1 RMSE: {rmse:.6e}")
# ------------------------------------------------------------
# True solution slice
# ------------------------------------------------------------
true_vals = []

for th2 in theta_test:
    x, u = solve_elliptic_direct(0.098, th2, n=400)
    true_vals.append(np.interp(x0, x, u))

true_vals = np.array(true_vals)
# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(10,4))

plt.plot(theta_test, true_vals, label="True solution")
plt.plot(theta_test, mu, label="GP mean")
plt.fill_between(theta_test,
                 mu-2*std,
                 mu+2*std,
                 alpha=0.3)

# training points
y_train= []
for th2 in theta_train:
    x, u = solve_elliptic_direct(0.098, th2, n=400)
    y_train.append(np.interp(x0, x, u))

plt.scatter(theta_train, y_train,
            color="black",
            label="Training θ")

plt.xlabel(r"$\theta_2$")
plt.ylabel(r"$u(x_0,\theta)$")
plt.title("Method 2 (Separable GP)")
plt.legend()
plt.grid()
plt.show()

rmse2 = np.sqrt(np.mean(mu2 - u_true)**2)
print(f"Method 2 RMSE: {rmse:.6e}")

# METHOD 3: PDE-INFORMED GP
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# ============================================================
# 1. PIECEWISE KAPPA & DIRECT SOLVER
# ============================================================
def kappa_piecewise(x, th1, th2):
    x = np.atleast_1d(x)
    k = np.zeros_like(x)
    k = np.where(x <= 0.25, 0.0, k)
    k = np.where((x > 0.25) & (x <= 0.50), th1, k)
    k = np.where((x > 0.50) & (x <= 0.75), th2, k)
    k = np.where(x > 0.75, 1.0, k)
    return k

def solve_elliptic_direct(th1, th2, n=400):
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    a = np.exp(kappa_piecewise(x, th1, th2))
    f = 4.0 * x
    a_h = 0.5 * (a[:-1] + a[1:])
    m = n - 2
    A = (np.diag(a_h[:-1] + a_h[1:]) - np.diag(a_h[1:-1], 1) - np.diag(a_h[1:-1], -1)) / h**2
    rhs = f[1:-1].copy()
    rhs[-1] -= (-a_h[-1] / h**2) * 2.0  # BC: u(1)=2
    u_int = la.solve(A, rhs)
    u = np.concatenate([[0.0], u_int, [2.0]])
    return x, u

# ============================================================
# METHOD 3 — PDE-informed GP (Variable Diffusivity)
# ============================================================
sigma2  = 1.0
ell_x   = 0.2
ell_th  = 0.6

noise_u = 1e-8
noise_f = 1e-8
jitter  = 1e-8

x0 = 0.3
theta2_const = 0.43
theta_train = np.linspace(0.1, 2, 6)
theta_test  = np.linspace(0.1, 2.0, 200)
x_f = np.linspace(0.1, 0.9, 6)

def k_xtheta(x, th, xp, thp):
    dx  = x[:, None] - xp[None, :]
    dth = th[:, None] - thp[None, :]
    return sigma2 * np.exp(-0.5*((dx/ell_x)**2 + (dth/ell_th)**2))

def d2_k_dx2(dx, k):
    return ((dx**2)/ell_x**4 - 1/ell_x**2) * k

def d4_k_dx2dxp2(dx, k):
    return (dx**4/ell_x**8 - 6*dx**2/ell_x**6 + 3/ell_x**4) * k

def f_rhs(x):
    return 4.0 * x

# ------------------------------------------------------------
# METHOD 3: 
# ------------------------------------------------------------
def method3_train(theta_train, x_f):
    # 1. Solution observations at (x0, theta)
    X_u = np.column_stack([np.full_like(theta_train, x0), theta_train])
    y_u = np.array([np.interp(x0, *solve_elliptic_direct(t, theta2_const)) for t in theta_train])

    # 2. PDE collocation points across the domain
    X_f_list = []
    for th in theta_train:
        for xx in x_f:
            X_f_list.append([xx, th])
    X_f_obs = np.array(X_f_list)
    y_f = f_rhs(X_f_obs[:, 0])

    y = np.concatenate([y_u, y_f])
    Nu, Nf = len(X_u), len(X_f_obs)

    # Operator: L = -exp(kappa(x, th1, th2)) * d^2/dx^2
    a_f = np.exp(kappa_piecewise(X_f_obs[:, 0], X_f_obs[:, 1], theta2_const))

    # Covariance blocks
    Kuu = k_xtheta(X_u[:,0], X_u[:,1], X_u[:,0], X_u[:,1])
    
    dx_uf = X_u[:,0,None] - X_f_obs[:,0][None,:]
    k_uf_base = k_xtheta(X_u[:,0], X_u[:,1], X_f_obs[:,0], X_f_obs[:,1])
    Kuf = -a_f[None, :] * d2_k_dx2(dx_uf, k_uf_base)

    dx_ff = X_f_obs[:,0,None] - X_f_obs[:,0][None,:]
    k_ff_base = k_xtheta(X_f_obs[:,0], X_f_obs[:,1], X_f_obs[:,0], X_f_obs[:,1])
    Kff = (a_f[:,None] * a_f[None,:]) * d4_k_dx2dxp2(dx_ff, k_ff_base)

    K = np.block([[Kuu + noise_u*np.eye(Nu), Kuf],
                  [Kuf.T, Kff + noise_f*np.eye(Nf)]])
    
    L = la.cholesky(K + jitter*np.eye(Nu+Nf), lower=True)
    alpha = la.cho_solve((L, True), y)

    return X_u, X_f_obs, L, alpha

def method3_predict(theta_star, x_star, X_u, X_f_obs, L, alpha):
    theta_star = np.asarray(theta_star).ravel()
    X_star = np.column_stack([np.full_like(theta_star, x_star), theta_star])

    # solution-solution covariance
    k_u = k_xtheta(X_star[:,0], X_star[:,1], X_u[:,0], X_u[:,1])

    # solution-PDE covariance
    dx_f = X_star[:,0,None] - X_f_obs[:,0][None,:]
    k_f_base = k_xtheta(X_star[:,0], X_star[:,1], X_f_obs[:,0], X_f_obs[:,1])
    a_f = np.exp(kappa_piecewise(X_f_obs[:,0], X_f_obs[:,1], theta2_const))
    Kuf_star = -a_f[None, :] * d2_k_dx2(dx_f, k_f_base)
    
    K_star = np.hstack([k_u, Kuf_star])

    mu = K_star @ alpha

    v = la.solve_triangular(L, K_star.T, lower=True)
    std = np.sqrt(np.clip(sigma2 - np.sum(v**2, axis=0), 0, np.inf))

    return mu, std

X_u3, X_f3, L3, alpha3 = method3_train(theta_train, x_f)
mu3, std3_x0 = method3_predict(theta_test, x0, X_u3, X_f3, L3, alpha3)

true_vals = [np.interp(x0, *solve_elliptic_direct(th, theta2_const)) for th in theta_test]
true_vals = np.array(true_vals)

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(10, 5))

plt.plot(theta_test, true_vals, '--', label="True solution")
plt.plot(theta_test, mu3, label="GP mean")
plt.fill_between(theta_test, mu3 - 1.96 * std3_x0, mu3 + 1.96 * std3_x0, alpha=0.15, label="95% CI")

# training points (actual labels used in train)
y_train = [np.interp(x0, *solve_elliptic_direct(th, theta2_const)) for th in theta_train]
plt.scatter(theta_train, y_train, color="black", label="Training θ", zorder=5)

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$u(x_0,\theta)$")
plt.title("Method 3 (PDE-informed GP)")
plt.ylim(0.8, 1.2)
plt.legend()
plt.grid(True)
plt.show()

rmse = np.sqrt(np.mean(mu3 - true_vals)**2)
print(f"Method 3 RMSE: {rmse:.6e}")