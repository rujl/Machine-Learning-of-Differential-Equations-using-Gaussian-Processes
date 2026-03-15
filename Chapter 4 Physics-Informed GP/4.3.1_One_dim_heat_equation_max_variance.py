import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# FOR POINTS THAT ARE PLACED WHERE THE VARIANCE IS AT ITS MAXIMUM VALUES 
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

alpha_val = 1/(4*np.pi**2)
w_val = 25
eps = 1e-12

# boundary u-data
X_u = np.array([0.0, 1.0])
y_u = np.array([0.0, 0.0])

X_f = np.array([])
y_f = np.sin(2*np.pi*X_f)

def gp_posterior_u(X_u, y_u, X_f, y_f):

    n_u = len(X_u)
    n_f = len(X_f)
    N = n_u + n_f

    K_uu = block_kernel(X_u, X_u, 0, 0, alpha_val, w_val)
    K_uf = block_kernel(X_u, X_f, 0, 1, alpha_val, w_val)
    K_fu = block_kernel(X_f, X_u, 1, 0, alpha_val, w_val)
    K_ff = block_kernel(X_f, X_f, 1, 1, alpha_val, w_val)

    K_xx = np.vstack([np.hstack([K_uu, K_uf]),
                      np.hstack([K_fu, K_ff])])

    K_tilde = K_xx + eps*np.eye(N)
    L = np.linalg.cholesky(K_tilde)

    y_train = np.concatenate([y_u, y_f])

    beta = np.linalg.solve(L, y_train)
    gamma = np.linalg.solve(L.T, beta)

    # return a function that computes posterior at arbitrary x*
    def posterior(x_query):
        xs = np.atleast_1d(x_query)
        M = len(xs)

        K_starX = np.zeros((M, N))
        diag = np.zeros(M)

        for i, xx in enumerate(xs):
            K2 = covariance_matrix(xx, xx, alpha_val, w_val)
            diag[i] = K2[0, 0]

            kuu_row = block_kernel([xx], X_u, 0, 0, alpha_val, w_val)[0, :]
            kuf_row = block_kernel([xx], X_f, 0, 1, alpha_val, w_val)[0, :]
            K_starX[i, :] = np.concatenate([kuu_row, kuf_row])

        u_mean = K_starX @ gamma
        v = np.linalg.solve(L, K_starX.T)
        var_u = diag - np.sum(v**2, axis=0)

        return u_mean, var_u

    return posterior


budget = 12

while len(X_u) + len(X_f) < budget: #while we are in budget

    post = gp_posterior_u(X_u, y_u, X_f, y_f) # fetch posterior mean and variance

    X_candidate = np.linspace(0, 1, 300)[1:-1]  # interior only
    _, var_candidate = post(X_candidate) # build GP with our current X_u, X_f, compute variance at each candidate point

    idx = np.argmax(var_candidate) # find location with max variance
    x_star = X_candidate[idx] # choose the candidate with the highest variance

    X_f = np.append(X_f, x_star) # add that point to the f points
    y_f = np.append(y_f, np.sin(2*np.pi*x_star))

    print(f"Added f-point at x = {x_star:.4f}")


posterior = gp_posterior_u(X_u, y_u, X_f, y_f) # compute new GP

X_test = np.linspace(0, 1, 300)
u_mean, var_u = posterior(X_test)
std_u = np.sqrt(np.maximum(var_u, 0))
u_exact = np.sin(2*np.pi*X_test)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.fill_between(X_test, u_mean - 2*std_u, u_mean + 2*std_u,
                 color="C0", alpha=0.2, label="95% CI")
plt.plot(X_test, u_mean, label="GP mean")
plt.plot(X_test, u_exact, "--", label="Exact")

plt.scatter(X_f, y_f, facecolors='none', edgecolors='k', s=40, label="f points")
plt.scatter(X_u, y_u, c='k', marker='x', label="boundary u")

# plt.title("GP regression with max-variance f-points")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('max_method.eps')
