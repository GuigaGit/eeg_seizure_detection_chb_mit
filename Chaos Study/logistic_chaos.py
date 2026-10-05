#!/usr/bin/env python3
r"""
Chaos in the logistic map   x -> r x (1 - x).

Produces two figures:

  1_orbit.png      cobweb, two orbits from almost identical starts,
                   and their separation on a log axis
  2_precision.png  the same orbit in float64 vs computed exactly,
                   showing one bit of the initial condition lost per step

At r = 4 the substitution x = sin^2(pi*theta) gives

    f(sin^2 pi t) = 4 sin^2 pi t cos^2 pi t = sin^2(2 pi t)

so the map is exactly the doubling map theta -> 2 theta mod 1. Doubling
shifts the binary expansion of theta left by one place, so the Lyapunov
exponent is ln 2 exactly: one bit of the initial condition consumed per
iteration. That is what the second figure measures, by running the orbit
in exact integer arithmetic alongside the float64 version.

Figure 2 needs the conjugacy, so it is only produced for r = 4.

Examples
--------
    python logistic_chaos.py
    python logistic_chaos.py --r 3.5          # periodic: orbits converge
    python logistic_chaos.py --eps 1e-8 --n 60

Requires: numpy, matplotlib
"""

import argparse
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = np.log(2.0)


def f(x, r):
    return r * x * (1.0 - x)


def df(x, r):
    return r * (1.0 - 2.0 * x)


def orbit(x0, r, n):
    """x_0 ... x_n."""
    out = np.empty(n + 1)
    out[0] = x = float(x0)
    for i in range(1, n + 1):
        x = f(x, r)
        out[i] = x
    return out


def lyapunov(x0, r, n, burn=1000):
    """lambda = <ln|f'(x)|> along the orbit."""
    x = float(x0)
    for _ in range(burn):
        x = f(x, r)
    s, m = 0.0, 0
    for _ in range(n):
        d = abs(df(x, r))
        if d > 1e-300:
            s += np.log(d)
            m += 1
        x = f(x, r)
    return s / max(m, 1)


def exact_orbit(n, seed=20260916):
    """Exact x_k for r = 4, via integer arithmetic on the doubling map.

    theta is an integer numerator over 2^B with B = n + 64, so doubling
    is an exact shift for all n steps. Only the final sin^2 is rounded.
    """
    B = n + 64
    t = random.Random(seed).getrandbits(B) | 1      # odd: no early collapse
    mask = (1 << B) - 1
    xs = []
    for _ in range(n + 1):
        xs.append(np.sin(np.pi * (t / (1 << B))) ** 2)
        t = (t << 1) & mask
    return np.array(xs)


def main():
    p = argparse.ArgumentParser(description="Chaotic orbits of the logistic map.")
    p.add_argument("--r", type=float, default=0.9, help="parameter r")
    p.add_argument("--x0", type=float, default=0.2, help="initial condition")
    p.add_argument("--eps", type=float, default=1e-12,
                   help="perturbation of the twin orbit")
    p.add_argument("--n", type=int, default=90, help="iterations to show")
    p.add_argument("--nlyap", type=int, default=1_000_000,
                   help="iterations used for the Lyapunov exponent")
    p.add_argument("--out", default=".", help="output directory")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tag = f"r{a.r:g}"

    x1 = orbit(a.x0, a.r, a.n)
    x2 = orbit(a.x0 + a.eps, a.r, a.n)
    sep = np.abs(x2 - x1)
    steps = np.arange(a.n + 1)
    lam = lyapunov(a.x0, a.r, a.nlyap)

    # ================= 1. orbit, cobweb, divergence =====================
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(0, 1, 800)
    ax.plot(g, f(g, a.r), "k", lw=1.5, label=r"$x_{n+1} = r\,x_n(1-x_n)$")
    ax.plot(g, g, "0.6", lw=1, ls="--")
    cx, cy = [x1[0]], [0.0]
    for k in range(min(a.n, 60)):
        cx += [x1[k], x1[k + 1]]
        cy += [x1[k + 1], x1[k + 1]]
    ax.plot(cx, cy, lw=0.6, color="crimson", alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"$x_n$"); ax.set_ylabel(r"$x_{n+1}$")
    ax.set_title(f"cobweb, r = {a.r:g}", fontsize=10)
    ax.legend(fontsize=9, loc="lower center")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, x1, "o-", ms=3, lw=0.9, color="#1f77b4",
            label=rf"$x_0$ = {a.x0:g}")
    ax.plot(steps, x2, "o-", ms=3, lw=0.9, color="#d62728", alpha=0.8,
            label=rf"$x_0$ + {a.eps:g}")
    ax.set_xlabel("n"); ax.set_ylabel(r"$x_n$")
    ax.set_title("two orbits from almost identical starts", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, :])
    ax.semilogy(steps, np.maximum(sep, 1e-18), "o-", ms=3, lw=0.9,
                color="#2ca02c", label=r"$|x_n^{(2)} - x_n^{(1)}|$")
    theory = a.eps * np.exp(lam * steps)
    keep = theory < 3.0
    ax.semilogy(steps[keep], theory[keep], "--", lw=1.6, color="k",
                label=rf"$\epsilon\,e^{{{lam:.4f}\,n}}$")
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    lo = min(a.eps, sep[sep > 0].min() if np.any(sep > 0) else a.eps)
    ax.set_ylim(lo * 1e-2, 30)
    nstar = np.log(1.0 / a.eps) / lam if lam > 0 else np.inf
    if np.isfinite(nstar) and nstar < a.n:
        ax.axvline(nstar, color="0.5", ls=":", lw=1)
        ax.annotate(f"prediction lost at n ≈ {nstar:.0f}",
                    xy=(nstar, 1.0), xytext=(nstar + 4, lo * 30),
                    fontsize=9, color="0.25",
                    arrowprops=dict(arrowstyle="->", color="0.45", lw=1))
    ax.set_xlabel("n"); ax.set_ylabel("separation")
    ax.set_title("straight line on a log axis = exponential = chaos",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle(rf"logistic map, r = {a.r:g}   —   $\lambda$ = {lam:.5f}",
                 fontsize=12)
    f1 = os.path.join(a.out, f"1_orbit_{tag}.png")
    fig.savefig(f1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ================= 2. the 53-bit horizon ============================
    f2 = None
    if abs(a.r - 4.0) < 1e-12:
        N = 120
        xe = exact_orbit(N)
        xf = orbit(xe[0], 4.0, N)
        err = np.abs(xf - xe)
        k = np.arange(N + 1)

        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
        axs[0].plot(k, xe, lw=1, color="k", label="exact (integer shift)")
        axs[0].plot(k, xf, lw=1, color="crimson", alpha=0.8,
                    label="float64 logistic")
        axs[0].set_xlabel("n"); axs[0].set_ylabel(r"$x_n$")
        axs[0].set_title("the same orbit, computed two ways", fontsize=10)
        axs[0].legend(fontsize=9); axs[0].grid(alpha=0.25)

        axs[1].semilogy(k, np.maximum(err, 1e-18), lw=1.1, color="#2ca02c",
                        label="actual error")
        axs[1].semilogy(k, 2.2e-16 * np.exp(LN2 * k), "--", color="k", lw=1.4,
                        label=r"$\epsilon_{mach}\,2^{\,n}$")
        axs[1].axhline(1.0, color="0.5", ls=":", lw=1)
        axs[1].axvline(53, color="0.5", ls=":", lw=1)
        axs[1].text(53, 1e-16, " 53 bits gone", fontsize=8, color="0.3")
        axs[1].set_ylim(1e-18, 10)
        axs[1].set_xlabel("n"); axs[1].set_ylabel("error of the float64 orbit")
        axs[1].set_title("one bit of the initial condition lost per step",
                         fontsize=10)
        axs[1].legend(fontsize=9); axs[1].grid(alpha=0.25)

        fig.suptitle("Why no computed chaotic orbit is the orbit you asked for",
                     fontsize=12)
        fig.tight_layout()
        f2 = os.path.join(a.out, f"2_precision_{tag}.png")
        fig.savefig(f2, dpi=140)
        plt.close(fig)

    # ================= printed summary ==================================
    print(f"\nlogistic map,  r = {a.r:g},  x0 = {a.x0:g},  eps = {a.eps:g}\n")
    print("   n        x_n (x0)        x_n (x0+eps)      separation")
    for i in range(13):
        print(f"  {i:2d}   {x1[i]:.12f}   {x2[i]:.12f}   {sep[i]:.3e}")

    print(f"\nLyapunov exponent over {a.nlyap:,} iterations: "
          f"lambda = {lam:.6f}")
    if lam > 0:
        print(f"  positive -> errors grow like e^({lam:.4f} n), "
              f"doubling every {LN2/lam:.2f} steps")
        print(f"  an initial error of {a.eps:g} reaches O(1) at "
              f"n ~ {np.log(1/a.eps)/lam:.0f}")
        if abs(a.r - 4.0) < 1e-12:
            print(f"  exact value from the conjugacy: ln 2 = {LN2:.6f}   "
                  f"(error {abs(lam - LN2):.1e})")
    else:
        print("  non-positive -> NOT chaotic here; the orbit settles onto "
              "a periodic cycle")

    print("\nwrote:")
    for path in (f1, f2):
        if path:
            print("  " + path)
    if f2 is None:
        print("  (figure 2 needs the r = 4 conjugacy, so it was skipped)")


if __name__ == "__main__":
    main()
