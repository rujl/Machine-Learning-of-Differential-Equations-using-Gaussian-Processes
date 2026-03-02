import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from heat_equation import u_exact, f

# -----------------------------
# Generate training data
# -----------------------------
np.random.seed(0)
n_train = 20

# u training points
tu = np.random.rand(n_train)
xu = np.random.rand(n_train)
yu = u_exact(tu, xu)

# f training points
tf = np.random.rand(n_train)
xf = np.random.rand(n_train)
yf = f(tf, xf)


# -----------------------------
# Create space-time grid
# -----------------------------
nt, nx = 100, 100
t = np.linspace(0, 1, nt)
x = np.linspace(0, 1, nx)
T, X = np.meshgrid(t, x)

U = u_exact(T, X)
F = f(T, X)


# -----------------------------
# Figure 2(A): u(t,x)
# -----------------------------
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot([0.1, 0.15, 0.40, 0.75], projection="3d")
ax1.plot_surface(T, X, U, cmap="viridis", alpha=0.8)
ax1.scatter(tu, xu, yu, marker='s', s=30)
ax1.set_title("Exact $u(t,x)$ with training data")
ax1.set_xlabel("t")
ax1.set_ylabel("x")
ax1.set_zlabel("u(t,x)")
ax1.view_init(elev=25, azim=-110)

# -----------------------------
# Figure 2(B): f(t,x)
# -----------------------------
ax2 = fig.add_subplot([0.48, 0.15, 0.40, 0.75], projection="3d")
ax2.plot_surface(T, X, F, cmap="viridis", alpha=0.8)
ax2.scatter(tf, xf, yf, marker="o", s=30)
ax2.set_title("Exact $f(t,x)$ with training data")
ax2.set_xlabel("t")
ax2.set_ylabel("x")
ax2.set_zlabel("f(t,x)", labelpad=5)
ax2.zaxis.set_label_coords(0.15, 0.5)
ax2.view_init(elev=25, azim=-110)
plt.show()