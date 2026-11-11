"""
Phase portrait (position x velocity) of a LINEAR pendulum (small-angle
approximation, sin(theta) ~ theta) with adjustable damping.

    theta'' = -(g/L) * theta - b * theta'

b = 0  -> ideal pendulum: perfect circles (energy conserved)
b > 0  -> damped pendulum: circles become inward spirals
          (b < 2*w0 underdamped, b = 2*w0 critical, b > 2*w0 overdamped)

The velocity axis is normalized by w0 = sqrt(g/L), so the ideal orbits are
true circles. Use --raw to plot the real velocity (orbits become ellipses).

Labels are in Portuguese and the text uses Latin Modern Roman (the OpenType
version of Computer Modern, the default LaTeX font), so the figures match the
dissertation. The font file is found with `kpsewhich` (MiKTeX / TeX Live);
if it is not available, DejaVu Serif is used instead.

Requirements:  pip install numpy scipy matplotlib
Run:           python linear_pendulum_phase_portrait.py
                   -> saves the two dissertation figures (b = 0 and
                      b = 0.67 1/s) to "Previa Dissertacao/Imagens"
               python linear_pendulum_phase_portrait.py --out some/dir
               python linear_pendulum_phase_portrait.py --interactive
                   -> opens a window with a damping slider (exploration only)
               python linear_pendulum_phase_portrait.py --interactive --damping 0.5
               python linear_pendulum_phase_portrait.py --raw
"""
import argparse
import os
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# ----------------------------- parameters ---------------------------------
G = 9.81                   # gravity (m/s^2)
L = 1.0                    # pendulum length (m)
W0 = np.sqrt(G / L)        # natural angular frequency (rad/s)
T_MAX = 20.0               # simulation time per trajectory (s)
B_MAX = 2.5 * W0           # slider range (past critical damping at b = 2*W0)

# Initial conditions (theta0 [rad], omega0 [rad/s])
INITIAL_CONDITIONS = [(1.0, 0.0)]

# Figures used in the dissertation: (damping b [1/s], output file name)
DISSERTATION_FIGURES = [
    (0.0, "espaço_pendulo_simples.png"),
    (0.67, "espaço_pendulo_real.png"),
]
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Previa Dissertacao", "Imagens"
)

# Figure size (inches) and font size (pt). The figure is included in LaTeX at
# 0.5\textwidth (~8.5 cm), so a 5.5 in figure is scaled by ~0.6: 15 pt here
# becomes ~9 pt in the document, close to the size of the caption text.
FIG_SIZE = (5.5, 5.5)
FONT_SIZE = 15
DPI = 300


# ------------------------------- fonts ------------------------------------
def find_latin_modern():
    """Path of lmroman10-regular.otf from the local TeX distribution, or None."""
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return None
    try:
        result = subprocess.run([kpsewhich, "lmroman10-regular.otf"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    path = result.stdout.strip()
    return path if path and os.path.isfile(path) else None


def setup_fonts():
    family = "DejaVu Serif"
    path = find_latin_modern()
    if path:
        font_manager.fontManager.addfont(path)
        family = font_manager.FontProperties(fname=path).get_name()
    else:
        print("Aviso: fonte Latin Modern Roman nao encontrada (kpsewhich); "
              "usando DejaVu Serif.")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [family],
        "mathtext.fontset": "cm",       # Computer Modern for math symbols
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 2,
    })


def decimal_comma(x, _pos):
    """Tick label with decimal comma and a true minus sign (pt-BR style)."""
    return f"{x:g}".replace(".", ",").replace("-", "−")


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
        ax.plot(x, vel, color=c, lw=1.6)
        ax.plot(x[0], vel[0], "o", color=c, ms=5)  # starting point

    ax.plot([0], [0], "ko", ms=6)  # equilibrium

    if b == 0:
        ax.set_title(r"Pêndulo ideal ($b$ = 0)")
    else:
        b_txt = f"{b:.2f}".replace(".", ",")
        ax.set_title(rf"Pêndulo amortecido ($b$ = {b_txt} s$^{{-1}}$)")
    ax.set_xlabel(r"Posição $\theta$ (rad)")
    ax.set_ylabel(r"Velocidade $\dot{\theta}$ (rad/s)" if raw
                  else r"Velocidade normalizada $\dot{\theta}/\omega_0$ (rad)")
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_v, lim_v)
    ax.xaxis.set_major_formatter(FuncFormatter(decimal_comma))
    ax.yaxis.set_major_formatter(FuncFormatter(decimal_comma))
    if not raw:
        ax.set_aspect("equal")  # makes the ideal orbits look like true circles
    ax.grid(alpha=0.3)


def save_dissertation_figures(out_dir, raw):
    os.makedirs(out_dir, exist_ok=True)
    for b, name in DISSERTATION_FIGURES:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        draw(ax, b, raw)
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Figura salva em {os.path.normpath(path)}")


def interactive(b0, raw):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    draw(ax, b0, raw)

    slider_ax = fig.add_axes([0.2, 0.04, 0.6, 0.03])
    slider = Slider(slider_ax, "Amortecimento $b$", 0.0, B_MAX,
                    valinit=b0, valstep=0.01)

    def update(val):
        draw(ax, slider.val, raw)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                        help="open a window with a damping slider instead of "
                             "saving the dissertation figures")
    parser.add_argument("--damping", type=float, default=0.0,
                        help="initial damping coefficient b in interactive mode")
    parser.add_argument("--raw", action="store_true",
                        help="plot real velocity instead of velocity / w0")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help="output directory for the saved figures")
    args = parser.parse_args()

    setup_fonts()
    if args.interactive:
        interactive(args.damping, args.raw)
    else:
        save_dissertation_figures(args.out, args.raw)


if __name__ == "__main__":
    main()
