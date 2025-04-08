import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Grid setup
x1, y1 = np.meshgrid(np.linspace(-1000, 1000, 500), np.linspace(-1000, 1000, 500))
x2, y2 = np.meshgrid(np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000))

# Streamfunction (left)
r1 = np.sqrt(x1**2 + y1**2)
theta1 = np.arctan2(y1, x1)

# Parameters
Vmax = 100.0 
a = 400.0 
Gamma = Vmax * 2 * np.pi * a 
alpha = 0.5
Lambda = alpha * Gamma       

# Rankine vortex streamfunction
psi_vortex = np.zeros_like(r1)
mask_inner = r1 <= a
mask_outer = r1 > a
psi_vortex[mask_inner] = (Gamma / (4 * np.pi * a**2)) * r1[mask_inner]**2
psi_vortex[mask_outer] = ((Gamma / (2 * np.pi)) * np.log(r1[mask_outer] / a) + (Gamma / (4 * np.pi)))

# Sink streamfunction
psi_sink = -Lambda / (2 * np.pi) * theta1

# Total streamfunction
psi_total = psi_vortex + psi_sink

# For velocity field
r2 = np.sqrt(x2**2 + y2**2)
theta2 = np.arctan2(y2, x2)

# Tangential and radial velocity components
V_theta = np.zeros_like(r2)
V_r = (Lambda/(2*np.pi*r2))
core_radius = a

inside_core = r2 <= core_radius
outside_core = r2 > core_radius
V_theta[inside_core] = (Gamma/(2*np.pi*a**2))*r2[inside_core]
V_theta[outside_core] = (Gamma/(2*np.pi*r2[outside_core]))

# Convert to Cartesian
Vx = V_r * np.cos(theta2) - V_theta * np.sin(theta2)
Vy = V_r * np.sin(theta2) + V_theta * np.cos(theta2)
V_mag = np.sqrt(Vx**2 + Vy**2)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Streamfunction contours
contours = ax1.contour(x1, y1, psi_total, levels=50, colors='black', linestyles='solid')
ax1.set_title(
    fr'Tornado Streamlines ($V_{{\theta,\,\text{{max}}}}$ = {Vmax:.1f} m/s, $a$ = {a:.1f} m)'
)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_aspect('equal')
ax1.text(
    0.05, 0.95,
    f'Γ = {Gamma:.1f}\nΛ = {Lambda:.1f}',
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
)

# Streamplot of velocity field
strm = ax2.streamplot(x2, y2, Vx, Vy, color=V_mag, linewidth=1, cmap='jet', density=4, arrowsize=0, norm=plt.Normalize(vmin=0, vmax=160))
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(strm.lines, cax=cax, label='Velocity Magnitude (m/s)')

ax2.set_title(
    fr'Velocity Map ($V_{{\theta,\,\text{{max}}}}$ = {Vmax:.1f} m/s, $a$ = {a:.1f} m)'
)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_aspect('equal')
ax2.text(
    0.05, 0.95,
    f'Γ = {Gamma:.1f}\nΛ = {Lambda:.1f}',
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
)

plt.tight_layout()
plt.show()
