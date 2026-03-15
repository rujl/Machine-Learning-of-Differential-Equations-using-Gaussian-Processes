import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -----------------------------------------------------------
# Heatmap of max error over (n_u, n_f)
# -----------------------------------------------------------

max_n_u = 40   # max number of u points to explore
max_n_f = 40   # max number of f points to explore

errors = np.full((max_n_u + 1, max_n_f + 1), np.nan)

for n_u in range(max_n_u + 1):
    for n_f in range(max_n_f + 1):
        if n_u + n_f == 0:
            continue  # skip the completely empty case
        err = gp_max_error(n_u, n_f)
        errors[n_u, n_f] = err

# Avoid exact zeros for log scale
min_pos = np.nanmin(errors[errors > 0])
errors_safe = np.where(errors > 0, errors, min_pos * 0.5)

plt.figure(figsize=(7, 5))
im = plt.imshow(
    errors_safe.T,
    origin='lower',
    aspect='auto',
    extent=[0, max_n_u, 0, max_n_f],
    norm=LogNorm()
)

plt.xlabel("Number of u points (n_u)")
plt.ylabel("Number of f points (n_f)")
plt.title("Max error heatmap over (n_u, n_f)")
cbar = plt.colorbar(im)
cbar.set_label("Max |u_mean(x) - sin(2πx)| (log scale)")
plt.tight_layout()
plt.show()
