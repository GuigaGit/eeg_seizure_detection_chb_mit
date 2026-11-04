"""
Phase portrait (position x velocity) of a pendulum with adjustable damping.

    theta'' = -(g/L) * sin(theta) - b * theta'

b = 0  -> ideal pendulum (energy conserved, closed orbits)
b > 0  -> real pendulum (energy lost, trajectories spiral into the origin)

Requirements:  pip install numpy scipy matplotlib
Run:           python pendulum_phase_portrait.py
               python pendulum_phase_portrait.py --damping 0.3
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# ----------------------------- parameters ---------------------------------
G = 9.81          # gravity (m/s^2)
L = 1.0           # pendulum length (m)
T_MAX = 25.0      # simulation time per trajectory (s)
B_MAX = 2.0       # max value of the damping slider

# Initial conditions (theta0 [rad], omega0 [rad/s]) for the trajectories
INITIAL_CONDITIONS = [
    (0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 0.0), (3.0, 0.0),
    (-1.0, 0.0), (-2.0, 0.0),
    (0.0, 6.5), (0.0, -6.5), (0.0, 8.0), (0.0, -8.0),   # rotating (over the top)
]


# ------------------------------ physics -----------------------------------
def pendulum(t, y, b):
    theta, omega = y
    return [omega, -(G / L) * np.sin(theta) - b * omega]


def trajectory(theta0, omega0, b):
    sol = solve_ivp(
        pendulum, (0, T_MAX), [theta0, omega0], args=(b,),
        t_eval=np.linspace(0, T_MAX, 4000), rtol=1e-9, atol=1e-9,
    )
    return sol.y[0], sol.y[1]


# ------------------------------ plotting ----------------------------------
def draw(ax, b):
    ax.clear()

    # Background vector field
    th = np.linspace(-2 * np.pi, 2 * np.pi, 40)
    om = np.linspace(-10, 10, 40)
    TH, OM = np.meshgrid(th, om)
    dTH = OM
    dOM = -(G / L) * np.sin(TH) - b * OM
    ax.streamplot(TH, OM, dTH, dOM, color="lightgray", density=1.2,
                  linewidth=0.7, arrowsize=0.8)

    # Trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(INITIAL_CONDITIONS)))
    for (th0, om0), c in zip(INITIAL_CONDITIONS, colors):
        x, v = trajectory(th0, om0, b)
        ax.plot(x, v, color=c, lw=1.4)
        ax.plot(th0, om0, "o", color=c, ms=4)  # starting point

    # Equilibria: stable at 0, unstable at +-pi
    ax.plot([0], [0], "ko", ms=6)
    ax.plot([-np.pi, np.pi], [0, 0], "rx", ms=8, mew=2)

    kind = "Ideal pendulum (b = 0)" if b == 0 else f"Real pendulum (b = {b:.2f} 1/s)"
    ax.set_title(f"Phase portrait - {kind}")
    ax.set_xlabel(r"Position  $\theta$  (rad)")
    ax.set_ylabel(r"Velocity  $\dot{\theta}$  (rad/s)")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-10, 10)
    ax.set_xticks(np.arange(-2, 3) * np.pi)
    ax.set_xticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--damping", type=float, default=0.0,
                        help="initial damping coefficient b (0 = ideal pendulum)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.18)
    draw(ax, args.damping)

    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(slider_ax, "Damping b", 0.0, B_MAX,
                    valinit=args.damping, valstep=0.01)

    def update(val):
        draw(ax, slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
