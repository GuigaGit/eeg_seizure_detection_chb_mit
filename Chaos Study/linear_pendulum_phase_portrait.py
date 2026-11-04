"""
Phase portrait (position x velocity) of a LINEAR pendulum (small-angle
approximation, sin(theta) ~ theta) with adjustable damping.

    theta'' = -(g/L) * theta - b * theta'

b = 0  -> ideal pendulum: perfect circles (energy conserved)
b > 0  -> damped pendulum: circles become inward spirals
          (b < 2*w0 underdamped, b = 2*w0 critical, b > 2*w0 overdamped)

The velocity axis is normalized by w0 = sqrt(g/L), so the ideal orbits are
true circles. Use --raw to plot the real velocity (orbits become ellipses).

Requirements:  pip install numpy scipy matplotlib
Run:           python linear_pendulum_phase_portrait.py
               python linear_pendulum_phase_portrait.py --damping 0.5
               python linear_pendulum_phase_portrait.py --raw
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# ----------------------------- parameters ---------------------------------
G = 9.81                   # gravity (m/s^2)
L = 1.0                    # pendulum length (m)
W0 = np.sqrt(G / L)        # natural angular frequency (rad/s)
T_MAX = 20.0               # simulation time per trajectory (s)
B_MAX = 2.5 * W0           # slider range (past critical damping at b = 2*W0)

# Initial conditions (theta0 [rad], omega0 [rad/s]); amplitudes 0.5 ... 3.0
INITIAL_CONDITIONS = [(1.0, 0.0)]


# ------------------------------ physics -----------------------------------
def pendulum(t, y, b):
    theta, omega = y
    return [omega, -(G / L) * theta - b * omega]


def trajectory(theta0, omega0, b, raw):
    sol = solve_ivp(
        pendulum, (0, T_MAX), [theta0, omega0], args=(b,),
        t_eval=np.linspace(0, T_MAX, 6000), rtol=1e-10, atol=1e-10,
    )
    x, v = sol.y
    return x, (v if raw else v / W0)


# ------------------------------ plotting ----------------------------------
def draw(ax, b, raw):
    ax.clear()
    scale = 1.0 if raw else 1.0 / W0
    lim_x, lim_v = np.sqrt(2), np.sqrt(2) * (W0 if raw else 1.0)

    # Background vector field (in plotted coordinates)
    th = np.linspace(-lim_x, lim_x, 35)
    v = np.linspace(-lim_v, lim_v, 35)
    TH, V = np.meshgrid(th, v)
    omega = V / scale                              # real velocity
    dTH = omega
    dV = (-(G / L) * TH - b * omega) * scale
    ax.streamplot(TH, V, dTH, dV, color="lightgray", density=1.2,
                  linewidth=0.7, arrowsize=0.8)

    # Trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(INITIAL_CONDITIONS)))
    for (th0, om0), c in zip(INITIAL_CONDITIONS, colors):
        x, vel = trajectory(th0, om0, b, raw)
        ax.plot(x, vel, color=c, lw=1.4)
        ax.plot(x[0], vel[0], "o", color=c, ms=4)  # starting point

    ax.plot([0], [0], "ko", ms=6)  # equilibrium

    kind = "Ideal pendulum (b = 0)" if b == 0 else f"Damped pendulum (b = {b:.2f} 1/s)"
    ax.set_title(f"Linear pendulum phase portrait - {kind}")
    ax.set_xlabel(r"Position  $\theta$  (rad)")
    ax.set_ylabel(r"Velocity  $\dot{\theta}$  (rad/s)" if raw
                  else r"Normalized velocity  $\dot{\theta}/\omega_0$  (rad)")
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_v, lim_v)
    if not raw:
        ax.set_aspect("equal")  # makes the ideal orbits look like true circles
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--damping", type=float, default=0.0,
                        help="initial damping coefficient b (0 = ideal pendulum)")
    parser.add_argument("--raw", action="store_true",
                        help="plot real velocity instead of velocity / w0")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    draw(ax, args.damping, args.raw)

    slider_ax = fig.add_axes([0.2, 0.04, 0.6, 0.03])
    slider = Slider(slider_ax, "Damping b", 0.0, B_MAX,
                    valinit=args.damping, valstep=0.01)

    def update(val):
        draw(ax, slider.val, args.raw)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
