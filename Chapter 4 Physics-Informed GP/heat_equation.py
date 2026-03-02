import numpy as np 

# -----------------------------
# Exact solutions
# -----------------------------
def u_exact(t, x):
    return np.exp(-t) * np.sin(2 * np.pi * x)

def f(t, x):
    return np.exp(-t) * (4 * np.pi**2 - 1) * np.sin(2 * np.pi * x)

