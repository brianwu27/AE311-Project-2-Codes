import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define grid
x, y = np.meshgrid(np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000))
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)


Vmax1 = 100.0       
a1 = 200.0          
Gamma1 = Vmax1 * 2 * np.pi * a1
alpha1 = 0.5
Lambda1 = alpha1 * Gamma1
x1, y1 = -300, 0

Vmax2 = 100.0       
a2 = 200.0         
Gamma2 = Vmax2 * 2 * np.pi * a2
alpha2 = 0.5
Lambda2 = alpha2 * Gamma2
x2, y2 = 300, 0

# Define Rankine vortex clearly
def rankine_vortex(x, y, x0, y0, Gamma, Rc):
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    psi = np.zeros_like(r)
    mask_inner = r <= Rc
    mask_outer = r > Rc
    psi[mask_inner] = (Gamma / (4 * np.pi * Rc**2)) * r[mask_inner]**2
    psi[mask_outer] = (Gamma / (2 * np.pi)) * np.log(r[mask_outer] / Rc) + Gamma / (4 * np.pi)
    return psi

# Define Sink clearly
def sink(x, y, x0, y0, Lambda):
    theta = np.arctan2(y - y0, x - x0)
    return -(Lambda / (2 * np.pi)) * theta

# Total streamfunction
psi_total_left = (rankine_vortex(x, y, x1, y1, Gamma1, a1) + sink(x, y, x1, y1, Lambda1) +
                  rankine_vortex(x, y, x2, y2, Gamma2, a2) + sink(x, y, x2, y2, Lambda2))

# Velocity 
def velocity_field(x, y, x0, y0, Gamma, Lambda, Rc):
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    theta = np.arctan2(y - y0, x - x0)

    Vr = Lambda / (2 * np.pi * r)
    Vtheta = np.zeros_like(r)
    Vtheta[r <= Rc] = (Gamma / (2 * np.pi * Rc**2)) * r[r <= Rc]
    Vtheta[r > Rc] = Gamma / (2 * np.pi * r[r > Rc])

    Vx = Vr * np.cos(theta) - Vtheta * np.sin(theta)
    Vy = Vr * np.sin(theta) + Vtheta * np.cos(theta)

    return Vx, Vy

# Combined velocity
Vx1, Vy1 = velocity_field(x, y, x1, y1, Gamma1, Lambda1, a1)
Vx2, Vy2 = velocity_field(x, y, x2, y2, Gamma2, Lambda2, a2)

Vx_total = Vx1 + Vx2
Vy_total = Vy1 + Vy2
V_mag = np.sqrt(Vx_total**2 + Vy_total**2)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left plot 
axes[0].contour(x, y, psi_total_left, levels=100, colors='black', linewidths=0.5, linestyles='solid')
axes[0].set_title(
    fr'Dual Tornado Streamlines ($V_{{\theta,\,\text{{max}}}}$ = {Vmax1:.1f} m/s, $a$ = {a1:.1f} m)'
)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_aspect('equal')
axes[0].grid(True)
# Text box for vortex 1 (left)
axes[0].text(
    0.03, 0.95,
    f"$\\Gamma_1$ = {Gamma1:.1f} m²/s\n$\\Lambda_1$ = {Lambda1:.1f} m²/s",
    transform=axes[0].transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
)

# Text box for vortex 2 (right)
axes[0].text(
    0.70, 0.95,
    f"$\\Gamma_2$ = {Gamma2:.1f} m²/s\n$\\Lambda_2$ = {Lambda2:.1f} m²/s",
    transform=axes[0].transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
)


# Right plot (streamplot with velocity)
strm = axes[1].streamplot(
    x, y, Vx_total, Vy_total, 
    color=V_mag, linewidth=1, cmap='jet', density=2,
    arrowstyle='->', arrowsize=1,norm = plt.Normalize(vmin=0, vmax=160)  
)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(strm.lines, cax=cax, label='Velocity Magnitude (m/s)')
axes[1].set_title(
    fr'Velocity Map ($V_{{\theta,\,\text{{max}}}}$ = {Vmax1:.1f} m/s, $a$ = {a1:.1f} m)'
)
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('y (m)')
axes[1].set_aspect('equal')
axes[1].grid(True)
axes[1].text(
    0.30, 0.95,
    f"$\\Gamma_1$ = {Gamma1:.0f} m²/s\n$\\Lambda_1$ = {Lambda1:.0f} m²/s",
    transform=axes[1].transAxes,
    fontsize=10,
    verticalalignment='top',
    ha='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
)
axes[1].text(
    0.70, 0.95,
    f"$\\Gamma_2$ = {Gamma2:.0f} m²/s\n$\\Lambda_2$ = {Lambda2:.0f} m²/s",
    transform=axes[1].transAxes,
    fontsize=10,
    verticalalignment='top',
    ha='left',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
)


plt.tight_layout()
plt.show()