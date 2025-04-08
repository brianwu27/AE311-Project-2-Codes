import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Grid setup
x, y = np.meshgrid(np.linspace(-500, 500, 1000), np.linspace(-500, 500, 1000))
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Parameters
Vmax = 100.0      
a = 200.0          
Gamma = Vmax * 2 * np.pi * a
alpha = 0.5
Lambda = alpha * Gamma

# Velocity components (polar)
V_r = (Lambda / (2 * np.pi * r))
V_theta = np.zeros_like(r)
inside_core = r <= a
outside_core = r > a
V_theta[inside_core] = (Gamma / (2 * np.pi * a**2)) * r[inside_core]
V_theta[outside_core] = (Gamma / (2 * np.pi * r[outside_core]))

# Convert to Cartesian
Vx = V_r * np.cos(theta) - V_theta * np.sin(theta)
Vy = V_r * np.sin(theta) + V_theta * np.cos(theta)

# Pressure field in kPa
P = 101.325 - ((3.0625 * 10**5) / r**2)
P[r == 0] = np.nan  # avoid divide-by-zero

# Plotting streamlines colored by pressure
fig, ax = plt.subplots(figsize=(8, 7))

# Streamlines with pressure-based color
strm = ax.streamplot(
    x, y, Vx, Vy,
    color=P,
    linewidth=1,
    cmap='coolwarm',
    density=2,
    arrowsize=0,
    norm=plt.Normalize(vmin=0, vmax=125)
)

# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(strm.lines, cax=cax, label='Pressure (kPa)')

# Labels and formatting
ax.set_title("Streamlines Colored by Pressure")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()