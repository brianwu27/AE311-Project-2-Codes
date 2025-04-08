import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
Gamma1, Gamma2 = 125664.0, -125664.0  
a = 200                
Lambda = Gamma1/2       

# Initial positions
pos1 = np.array([-750.0, 2000.0])
pos2 = np.array([750.0, 2000.0])

# Biot-Savart induced velocity function
def induced_velocity(pos_from, pos_to, Gamma):
    r = pos_to - pos_from
    distance_squared = np.sum(r**2)
    perpendicular = np.array([-r[1], r[0]])
    velocity = (Gamma / (2 * np.pi * distance_squared)) * perpendicular
    return velocity

# Rankine vortex streamfunction
def rankine_vortex(x, y, x0, y0, Gamma, Rc):
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    psi = np.zeros_like(r)

    mask_inner = r <= Rc
    psi[mask_inner] = (Gamma / (4 * np.pi * Rc**2)) * r[mask_inner]**2

    mask_outer = r > Rc
    psi[mask_outer] = (Gamma / (2 * np.pi)) * np.log(r[mask_outer] / Rc) + Gamma / (4 * np.pi)

    return psi

# Sink streamfunction
def sink(x, y, x0, y0, Lambda):
    theta = np.arctan2(y - y0, x - x0)
    return -(Lambda / (2 * np.pi)) * theta

# Setup plot
fig, ax = plt.subplots(figsize=(7, 7))
x, y = np.meshgrid(np.linspace(-5000, 5000, 500), np.linspace(-5000, 5000, 500))


# Time step
dt = 2

# Animation function
def animate(i):
    global pos1, pos2
    ax.clear()
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-2500, 2500)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Two Realistic Tornadoes (Opposite Rotation)")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Compute induced velocities
    v1 = induced_velocity(pos1, pos2, Gamma2)
    v2 = induced_velocity(pos2, pos1, Gamma1)

    # Update positions
    pos1 += v1 * dt
    pos2 += v2 * dt

    # Streamfunctions
    psi1 = rankine_vortex(x, y, pos1[0], pos1[1], Gamma1, a) + sink(x, y, pos1[0], pos1[1], Lambda)
    psi2 = rankine_vortex(x, y, pos2[0], pos2[1], Gamma2, a) + sink(x, y, pos2[0], pos2[1], Lambda)

    psi_total = psi1 + psi2

    contour = ax.contour(x, y, psi_total, levels=50, cmap='jet')

ani = FuncAnimation(fig, animate, frames=200, interval=50)

plt.show()
