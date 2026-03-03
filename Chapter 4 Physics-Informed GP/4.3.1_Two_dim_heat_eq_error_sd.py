import numpy as np 
import matplotlib.pyplot as plt 
# ERROR AND STANDARD DEVIATION FOR u AND f POINTS
# ------------------------------------------------- 
# Evaluation grid 
# ------------------------------------------------- 
nt, nx = 100, 100 
t = np.linspace(0, 1, nt) 
x = np.linspace(0, 1, nx) 
T, X = np.meshgrid(t, x) 

# ----------------------------
# 1) SYMBOLIC: build cov block 
# ---------------------------- 
# symbols 
x1, t1, x2, t2 = sp.symbols('x1 t1 x2 t2', real=True) 
wx, wt = sp.symbols('wx wt', positive=True) # weights = 1/ell^2 
# squared-exponential kernel in (x,t) 
alpha = 1 
sigma = 1 
g_sym = sigma ** 2 * sp.exp(-0.5 * ( wx*(x1 - x2)**2 + wt*(t1 - t2)**2 )) 

# derivatives needed 
g_t1 = sp.diff(g_sym, t1) 
g_t2 = sp.diff(g_sym, t2) 
g_x1x1 = sp.diff(g_sym, x1, 2) 
g_x2x2 = sp.diff(g_sym, x2, 2) 
g_t1t2 = sp.diff(g_sym, t1, 1, t2, 1) 
g_t1_x2x2 = sp.diff(g_t1, x2, 2) 
g_x1x1_t2 = sp.diff(g_x1x1, t2, 1) 
g_x1x1_x2x2 = sp.diff(g_x1x1, x2, 2) 

# blocks for u vs f where f = u_t - u_xx 
kuu = g_sym 
kuf = g_t2 - alpha * g_x2x2 # Cov(u, u_t - u_xx) 
kfu = g_t1 - alpha * g_x1x1 # Cov(u_t - u_xx, u) 
kff = g_t1t2 - alpha * g_t1_x2x2 - alpha * g_x1x1_t2 + alpha * g_x1x1_x2x2 
K_block_sym = sp.Matrix([[kuu, kuf], [kfu, kff]]) 

# lambdify: cov_block(x1,t1,x2,t2,wx,wt) -> 2x2 numpy array 
cov_block = sp.lambdify((x1, t1, x2, t2, wx, wt), K_block_sym, 'numpy') 

# ---------------------------- 
# 2) helper to build subblocks 
# ---------------------------- 
def block_kernel_times(Xa, Xb, row_idx, col_idx, wx_val, wt_val): 
    """ Xa, Xb: arrays of 2D points shape (n,2) where columns = [x, t] 
        row_idx,col_idx: 0 for u, 1 for f 
        returns matrix K of shape (len(Xa), len(Xb)) """ 
    Xa = np.asarray(Xa).reshape(-1, 2) 
    Xb = np.asarray(Xb).reshape(-1, 2) 
    K = np.zeros((Xa.shape[0], Xb.shape[0])) 
    for i, (xi, ti) in enumerate(Xa): 
        for j, (xj, tj) in enumerate(Xb): 
            block = cov_block(float(xi), float(ti), float(xj), float(tj), float(wx_val), float(wt_val)) 
            K[i, j] = block[row_idx, col_idx] 
    return K 

# --- f(t, x), g(x) and exact solution --- 
def u_true(t, x): 
    return np.exp(-t) * np.sin(2 * np.pi * x) 

def f(t, x): 
    return np.exp(-t) * (4 * np.pi**2 - 1) * np.sin(2 * np.pi * x) 

def g(x): 
    return np.sin(2* np.pi* x) 

# hyperparameters 
ell_x = 0.25 
ell_t = 0.205 
wx_val = 1.0 / (ell_x**2) 
wt_val = 1.0 / (ell_t**2) 

# Boundary observations (u on x=0 and x=1 at several times) 
times_b = np.linspace(0, 1, 20) # sample times at which boundary u is known 
X_b_0 = np.column_stack([np.zeros_like(times_b), times_b]) # u(0,t) 
X_b_1 = np.column_stack([np.ones_like(times_b), times_b]) # u(1,t) 
X_boundary = np.vstack([X_b_0, X_b_1]) # shape (2*len(times_b), 2) 
y_boundary = np.zeros(X_boundary.shape[0]) # u=0 on boundary 

# Initial condition: u training points (u(x,0) = sin(2 pi x)) 
x_init = np.linspace(0, 1, 20) 
t_init = np.zeros_like(x_init)
X_ic = np.column_stack([x_init, t_init]) # (t=0, x) 
y_ic = np.sin(2 * np.pi * x_init) # u(0, x) 

# Combine u-observations (boundary + initial) 
X_u = np.vstack([X_boundary, X_ic]) # all locations where we observe u 
y_u = np.concatenate([y_boundary, y_ic]) # their observed u values 
# PDE collocation points (interior space-time where we enforce u_t - u_xx = f(x)) 
#xs = np.linspace(0.05, 0.95, 12) # 0 <= x <= 1 
#ts = np.linspace(0.05, 0.95, 12) # 0 < t <= 1 
#X_f = np.array([[xx, tt] for tt in ts for xx in xs]) # shape (n_coll,2) 
rng = np.random.default_rng(0)
X_f = rng.uniform([0,0], [1,1], size=(150,2))
# PDE RHS u_t - u_xx = e^-t(4pi^2-1) sin (2pi x) 
y_f = np.array([f(tt, xx) for (xx, tt) in X_f]) 

# Build observation vector ordering: [u(X_u); f(X_f)] 
y_obs = np.concatenate([y_u, y_f]) 

# ---------------------------- 
# 4) assemble K_obs using blocks 
# ---------------------------- 
K_uu = block_kernel_times(X_u, X_u, 0, 0, wx_val, wt_val) 
K_uf = block_kernel_times(X_u, X_f, 0, 1, wx_val, wt_val) 
K_fu = block_kernel_times(X_f, X_u, 1, 0, wx_val, wt_val) 
K_ff = block_kernel_times(X_f, X_f, 1, 1, wx_val, wt_val) 
top = np.hstack([K_uu, K_uf]) 
bottom = np.hstack([K_fu, K_ff]) 
K_obs = np.vstack([top, bottom]) 

# jitter for numerical stability jitter = 1e-8 
K_obs += jitter * np.eye(K_obs.shape[0]) 
# ---------------------------- 
# 5) build K_star for test locations and solve 
# ---------------------------- 
# Precompute Cholesky 
L = scipy.linalg.cholesky(K_obs, lower=True) 
alpha_vec = scipy.linalg.cho_solve((L, True), y_obs) 

# test grid
x_test = np.linspace(0, 1, 80) 
t_test = np.linspace(0, 1, 80) 
X_star = np.array([[x,t] for t in t_test for x in x_test]) 

# posterior mean 
Kstar_u = block_kernel_times(X_star, X_u, 0, 0, wx_val, wt_val) 
Kstar_f = block_kernel_times(X_star, X_f, 0, 1, wx_val, wt_val) 
K_star = np.hstack([Kstar_u, Kstar_f]) 

# Solve (K_obs) * alpha = y_obs using Cholesky factorisation for better efficiency --> K_obs = LL^T 
L = np.linalg.cholesky(K_obs) 
alpha_vec = scipy.linalg.cho_solve((L, True), y_obs) # Solve Lz = y and L^T alpha to get alpha = (K_obs)^{-1} y_obs 
u_star_mean = K_star @ alpha_vec # equivalent to eq 2.23 
v = scipy.linalg.cho_solve((L, True), K_star.T) 
# diagonal prior variance k((x,t),(x,t)) 
K_diag = np.zeros(X_star.shape[0]) 
for i, (xq, tq) in enumerate(X_star): 
    block = cov_block(xq, tq, xq, tq, wx_val, wt_val) 
    K_diag[i] = block[0,0] # for u; use [1,1] for f 

# posterior variance (diagonal only) 
v = scipy.linalg.cho_solve((L, True), K_star.T) 
u_var = K_diag - np.sum(K_star * v.T, axis=1) 
u_std = np.sqrt(np.clip(u_var, 0, np.inf)) 
U_std = u_std.reshape(len(t_test), len(x_test)) 
U_mean = u_star_mean.reshape(len(t_test), len(x_test)) 
X, T = np.meshgrid(x_test, t_test) 

# COMPUTE POSTERIOR FOR f 
Kstar_u_f = block_kernel_times(X_star, X_u, 1, 0, wx_val, wt_val) 
Kstar_f_f = block_kernel_times(X_star, X_f, 1, 1, wx_val, wt_val) 
K_star_f = np.hstack([Kstar_u_f, Kstar_f_f]) 
f_star_mean = K_star_f @ alpha_vec 
F_mean = f_star_mean.reshape(len(t_test), len(x_test)) 
v_f = scipy.linalg.cho_solve((L, True), K_star_f.T) 
Kf_diag = np.zeros(X_star.shape[0]) 
for i, (xq, tq) in enumerate(X_star): 
    block = cov_block(xq, tq, xq, tq, wx_val, wt_val) 
    Kf_diag[i] = block[1,1] 
v_f = scipy.linalg.cho_solve((L, True), K_star_f.T) 
f_var = Kf_diag - np.sum(K_star_f * v_f.T, axis=1) 
f_std = np.sqrt(np.clip(f_var, 0, np.inf)) 
F_std = f_std.reshape(len(t_test), len(x_test)) 
# ------------------------------------------------- 
# Errors 
# ------------------------------------------------- 
U_err = np.abs(U_mean -  u_true(T, X)) 
F_err = np.abs(F_mean - f(T, X)) 

# -------------------------------------------------
# Plot 
# ------------------------------------------------- 
fig, axes = plt.subplots(2, 2, figsize=(10, 8)) 

# (A) Error in u 
cs = axes[0,0].contourf(X, T, U_err, levels=40, cmap='cool') 
axes[0,0].contour(X, T, U_err, levels=20, colors='k', linewidths=0.3) 
axes[0,0].scatter(xu, tu, marker='s', facecolors='none', edgecolors='k') 
axes[0,0].scatter(xf, tf, marker='o', facecolors='none', edgecolors='k') 
axes[0,0].set_title(r'(A) Error $|\bar u(t,x)-u(t,x)|$') 
fig.colorbar(cs, ax=axes[0,0]) 

# (B) Error in f 
cs = axes[0,1].contourf(X, T, F_err, levels=40, cmap='cool') 
axes[0,1].contour(X, T, F_err, levels=20, colors='k', linewidths=0.3) 
axes[0,1].scatter(xu, tu, marker='s', facecolors='none', edgecolors='k') 
axes[0,1].scatter(xf, tf, marker='o', facecolors='none', edgecolors='k') 
axes[0,1].set_title(r'(B) Error $|\bar f(t,x)-f(t,x)|$') 
fig.colorbar(cs, ax=axes[0,1]) 

# (C) Std for u 
cs = axes[1,0].contourf(X, T, U_std, levels=40, cmap='cool') 
axes[1,0].contour(X, T, U_std, levels=20, colors='k', linewidths=0.3) 
axes[1,0].scatter(xu, tu, marker='s', facecolors='none', edgecolors='k') 
axes[1,0].scatter(xf, tf, marker='o', facecolors='none', edgecolors='k') 
axes[1,0].set_title(r'(C) Standard deviation for $u$') 
fig.colorbar(cs, ax=axes[1,0]) 

# (D) Std for f 
cs = axes[1,1].contourf(X, T, F_std, levels=40, cmap='cool') 
axes[1,1].contour(X, T, F_std, levels=20, colors='k', linewidths=0.3) 
axes[1,1].scatter(xu, tu, marker='s', facecolors='none', edgecolors='k') 
axes[1,1].scatter(xf, tf, marker='o', facecolors='none', edgecolors='k') 
axes[1,1].set_title(r'(D) Standard deviation for $f$') 
fig.colorbar(cs, ax=axes[1,1]) 

for ax in axes.flat: 
    ax.set_xlabel('t') 
    ax.set_ylabel('x') 
    plt.tight_layout() 
plt.show()
