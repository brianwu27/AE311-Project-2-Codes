import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x, y = np.meshgrid(np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000))

# Parameters 
Gamma1 = 125664    
Gamma2 = -125664  
Lambda = 62832
a = 200

# Velocity function
def velocity_field(x, y, vortex_pos, Gamma):
    x0, y0 = vortex_pos
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    theta = np.arctan2(y - y0, x - x0)

    V_theta = np.zeros_like(r)
    V_r = Lambda / (2 * np.pi * r)

    inside_core = r <= a
    outside_core = r > a

    V_theta[inside_core] = (Gamma / (2 * np.pi * a**2)) * r[inside_core]
    V_theta[outside_core] = Gamma / (2 * np.pi * r[outside_core])

    Vx = V_r * np.cos(theta) - V_theta * np.sin(theta)
    Vy = V_r * np.sin(theta) + V_theta * np.cos(theta)

    return Vx, Vy


# Positions
initial_distance = 400
frames = 40
pos1 = np.array([-initial_distance, 0])
pos2 = np.array([initial_distance, 0])

fig, ax = plt.subplots(figsize=(8, 8))

# Color bar
cbar = None

# Animation
def animate(i):
    global cbar
    ax.clear()
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Two Tornadoes Forced Towards Each Other (Opposite Rotation)')
    ax.grid()

    # Linear motion
    t = i / frames
    current_pos1 = (1 - t) * pos1
    current_pos2 = (1 - t) * pos2

    # Combined velocity field
    Vx1, Vy1 = velocity_field(x, y, current_pos1, Gamma1)
    Vx2, Vy2 = velocity_field(x, y, current_pos2, Gamma2)

    Vx_total = Vx1 + Vx2
    Vy_total = Vy1 + Vy2

    V_mag = np.sqrt(Vx_total**2 + Vy_total**2)

    # Streamplot
    strm = ax.streamplot(x, y, Vx_total, Vy_total, color=V_mag, cmap='jet', linewidth=1, density=2, arrowstyle='->', arrowsize=1.5, norm = plt.Normalize(vmin=0, vmax=160))

    # Add colorbar only once
    if cbar is None:
        cbar = plt.colorbar(strm.lines, ax=ax, label='Velocity Magnitude')

ani = FuncAnimation(fig, animate, frames=frames, interval=100)

plt.show()