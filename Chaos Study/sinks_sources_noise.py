#!/usr/bin/env python3
"""
Sinks, sources and saddles under noise
======================================

Definition 2.2 (Alligood/Sauer/Yorke) says that p is a *sink* when

    lim_{k->inf} f^k(v) = p     for every v in some epsilon-neighborhood N_eps(p),

and a *source* when every v in N_eps(p), except p itself, eventually leaves N_eps(p).

Both definitions rely on the iteration being exact.  Any real system (and any
computer) iterates

    v_{k+1} = f(v_k) + sigma * xi_k ,        xi_k ~ N(0, I),

with a small random kick sigma * xi_k: measurement error, thermal noise,
round-off, an unmodelled term.  That single change breaks the equality in the
definitions, and the two cases break in *opposite* ways:

  SINK    The limit no longer exists.  The orbit still collapses toward p at the
          rate |lambda|^k, but only until contraction balances the kicks.  From
          then on it rattles forever inside a disk of radius

              r_rms = sigma * sqrt(2 / (1 - lambda^2))          (2-D, A = lambda*R)

          The attracting *point* has become an attracting *set* N_r(p).  The orbit
          gets arbitrarily close to that disk and never closer to p: dist(v_k, p)
          has a noise floor, not a limit of 0.

  SOURCE  The equality p = f(p) is no longer observable.  Start the orbit exactly
          on p: noiseless it stays there forever, but with noise of any size the
          state is pushed off and then amplified by mu^k.  The escape time is only
          logarithmic in sigma,

              T(d) ~ log(d * sqrt(mu^2 - 1) / (sigma * sqrt(2))) / log(mu),

          so dividing the noise by 10^6 buys only ~ 14 / log(mu) extra iterations.
          A source is invisible in practice: you can never sit on one.

  SADDLE  Both effects at once.  Starting exactly on the stable manifold the
          noiseless orbit converges to p; noise kicks the orbit off the stable
          direction and the unstable eigenvalue does the rest.  The stable
          manifold is a set of measure zero and noise makes it unreachable.

Usage
-----
    python sinks_sources_noise.py                       # default figure
    python sinks_sources_noise.py --sigma 1e-4 --seed 7
    python sinks_sources_noise.py --out my_figure.png
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# maps
# --------------------------------------------------------------------------- #
def rot(theta: float) -> np.ndarray:
    """2-D rotation matrix (so the orbits spiral and are easier to look at)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def linear_map(A: np.ndarray, p: np.ndarray):
    """f(v) = p + A (v - p).  Fixed point exactly at p; local behaviour set by A.

    Eigenvalues of A inside the unit circle -> sink, outside -> source,
    some inside and some outside -> saddle.
    """

    def f(v: np.ndarray) -> np.ndarray:
        return p + A @ (v - p)

    return f


def orbit(f, v0, n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Iterate v_{k+1} = f(v_k) + sigma * xi_k and return the (n+1, 2) orbit."""
    v = np.asarray(v0, dtype=float)
    out = np.empty((n + 1, v.size))
    out[0] = v
    for k in range(n):
        v = f(v) + sigma * rng.standard_normal(v.size)
        out[k + 1] = v
    return out


def dist_to(orb: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(orb - p, axis=1)


# --------------------------------------------------------------------------- #
# theory
# --------------------------------------------------------------------------- #
def sink_noise_floor(sigma: float, lam: float) -> float:
    """RMS radius of the invariant cloud around a sink with A = lam * rotation."""
    return sigma * np.sqrt(2.0 / (1.0 - lam**2))


def source_escape_time(d: float, sigma: float, mu: float) -> float:
    """Iterations for an orbit started ON the source to reach distance d."""
    return np.log(d * np.sqrt(mu**2 - 1.0) / (sigma * np.sqrt(2.0))) / np.log(mu)


# --------------------------------------------------------------------------- #
# numerical experiments
# --------------------------------------------------------------------------- #
def measure_floor(lam: float, sigma: float, rng, n: int = 20_000) -> float:
    """Measured RMS distance to the sink, averaged over the second half of the run."""
    A = lam * rot(0.35)
    p = np.zeros(2)
    orb = orbit(linear_map(A, p), np.array([1.0, 0.0]), n, sigma, rng)
    d = dist_to(orb[n // 2 :], p)
    return float(np.sqrt(np.mean(d**2)))


def measure_escape(mu: float, sigma: float, d: float, rng, trials: int = 200) -> float:
    """Median number of iterations to leave N_d(p), starting exactly at the source."""
    A = mu * rot(0.35)
    e = np.zeros((trials, 2))              # e_k = v_k - p, so we start ON p
    t = np.full(trials, np.nan)
    for k in range(1, 2000):
        e = e @ A.T + sigma * rng.standard_normal(e.shape)
        out = (np.linalg.norm(e, axis=1) >= d) & np.isnan(t)
        t[out] = k
        if not np.isnan(t).any():
            break
    return float(np.nanmedian(t))


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=1e-3, help="noise amplitude (default 1e-3)")
    ap.add_argument("--lam", type=float, default=0.90, help="sink contraction rate |lambda| < 1")
    ap.add_argument("--mu", type=float, default=1.15, help="source expansion rate |mu| > 1")
    ap.add_argument("--steps", type=int, default=400, help="iterations per orbit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sinks_sources_noise.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sigma, lam, mu, n = args.sigma, args.lam, args.mu, args.steps

    p = np.array([1.0, 1.0])               # the fixed point, same for all three cases
    A_sink = lam * rot(0.35)
    A_src = mu * rot(0.35)
    A_sad = np.array([[0.40, 0.0], [0.0, 1.60]])   # stable x-axis, unstable y-axis

    fig, ax = plt.subplots(3, 2, figsize=(13.5, 15.0))
    fig.suptitle(
        f"Sinks, sources and saddles when the iteration is not exact "
        f"(noise $\\sigma={sigma:g}$)", fontsize=15, y=0.995)

    # ---- (a) sink orbit in the plane -------------------------------------- #
    a = ax[0, 0]
    orb = orbit(linear_map(A_sink, p), p + np.array([1.6, 0.9]), n, sigma, rng)
    r = sink_noise_floor(sigma, lam)
    th = np.linspace(0, 2 * np.pi, 400)
    a.plot(orb[:, 0], orb[:, 1], lw=0.7, color="#2c6fbb")
    a.plot(*p, "o", color="crimson", ms=7, zorder=5, label="fixed point $\\mathbf{p}$")
    a.set_title("(a) sink: the orbit spirals in and then never stops moving")
    a.set_xlabel("$x$"); a.set_ylabel("$y$"); a.legend(fontsize=8, loc="lower right")
    a.set_aspect("equal")

    # zoom on the invariant cloud: this is as close as the orbit ever gets
    zoom = a.inset_axes((0.60, 0.54, 0.36, 0.36))
    tail = orb[n // 2:]
    zoom.plot(tail[:, 0], tail[:, 1], lw=0.5, color="#2c6fbb")
    zoom.plot(*p, "o", color="crimson", ms=5, zorder=5)
    zoom.plot(p[0] + 3 * r * np.cos(th), p[1] + 3 * r * np.sin(th), "--",
              color="crimson", lw=1.2)
    w = 4.0 * r
    zoom.set_xlim(p[0] - w, p[0] + w); zoom.set_ylim(p[1] - w, p[1] + w)
    zoom.set_aspect("equal"); zoom.set_xticks([]); zoom.set_yticks([])
    zoom.set_title(f"zoom $\\pm{w:.0e}$: it orbits $\\mathbf{{p}}$ forever", fontsize=7)

    # ---- (b) sink: distance vs k, noiseless vs noisy ----------------------- #
    b = ax[0, 1]
    v0 = p + np.array([1.6, 0.9])
    d_clean = dist_to(orbit(linear_map(A_sink, p), v0, n, 0.0, rng), p)
    d_noisy = dist_to(orb, p)
    b.semilogy(np.maximum(d_clean, 1e-300), color="#444", lw=1.2,
               label="$\\sigma=0$: $\\to 0$ (Definition 2.2)")
    b.semilogy(d_noisy, color="#2c6fbb", lw=0.9, label=f"$\\sigma={sigma:g}$: stalls")
    b.axhline(r, color="crimson", ls="--", lw=1.3,
              label=f"predicted floor $\\sigma\\sqrt{{2/(1-\\lambda^2)}}={r:.2e}$")
    b.set_title("(b) sink: the limit is replaced by a noise floor")
    b.set_xlabel("iteration $k$"); b.set_ylabel("$|\\mathbf{v}_k-\\mathbf{p}|$")
    b.set_ylim(r / 100, 10); b.legend(fontsize=8)

    # ---- (c) source: starting exactly on p --------------------------------- #
    c = ax[1, 0]
    f_src = linear_map(A_src, p)
    d_exact = dist_to(orbit(f_src, p, n, 0.0, rng), p)                 # stays on p
    d_eps = dist_to(orbit(f_src, p + np.array([1e-15, 0.0]), n, 0.0, rng), p)
    d_noise = dist_to(orbit(f_src, p, n, sigma, rng), p)               # starts ON p
    c.semilogy(np.where(d_exact == 0.0, 1.5e-18, d_exact), color="#444", lw=1.8,
               label="$\\sigma=0$, $\\mathbf{v}_0=\\mathbf{p}$: distance $\\equiv 0$")
    c.text(n * 0.02, 3.5e-18, "exactly 0, drawn on the axis floor",
           fontsize=7, color="#444")
    c.semilogy(np.maximum(d_eps, 1e-320), color="#7a52a1", lw=1.0,
               label="$\\sigma=0$, off by $10^{-15}$")
    c.semilogy(np.maximum(d_noise, 1e-320), color="#2c6fbb", lw=1.0,
               label=f"$\\sigma={sigma:g}$, $\\mathbf{{v}}_0=\\mathbf{{p}}$")
    T = source_escape_time(1.0, sigma, mu)
    c.axvline(T, color="crimson", ls="--", lw=1.2, label=f"predicted escape $k\\approx{T:.0f}$")
    c.axhline(1.0, color="crimson", ls=":", lw=1.0)
    c.set_title("(c) source: the fixed point cannot be occupied")
    c.set_xlabel("iteration $k$"); c.set_ylabel("$|\\mathbf{v}_k-\\mathbf{p}|$")
    c.set_ylim(1e-18, 1e4); c.legend(fontsize=8, loc="lower right")

    # ---- (d) saddle: leaving the stable manifold ---------------------------- #
    d = ax[1, 1]
    f_sad = linear_map(A_sad, p)
    v0s = p + np.array([1.5, 0.0])                       # exactly on the stable manifold
    clean = orbit(f_sad, v0s, 60, 0.0, rng)
    noisy = orbit(f_sad, v0s, 60, sigma, rng)
    keep = np.linalg.norm(noisy - p, axis=1) < 3.0
    d.plot(clean[:, 0], clean[:, 1], "o-", ms=3, color="#444", lw=1.0,
           label="$\\sigma=0$: converges to $\\mathbf{p}$")
    d.plot(noisy[keep, 0], noisy[keep, 1], "o-", ms=3, color="#2c6fbb", lw=1.0,
           label=f"$\\sigma={sigma:g}$: escapes")
    d.axhline(p[1], color="seagreen", lw=1.0, ls="--", label="stable manifold")
    d.axvline(p[0], color="darkorange", lw=1.0, ls="--", label="unstable manifold")
    d.plot(*p, "o", color="crimson", ms=7, zorder=5)
    d.set_title("(d) saddle: noise knocks the orbit off the stable manifold")
    d.set_xlabel("$x$"); d.set_ylabel("$y$"); d.legend(fontsize=8)
    d.set_xlim(p[0] - 0.4, p[0] + 1.8); d.set_ylim(p[1] - 2.2, p[1] + 2.2)

    # ---- (e) noise floor scales linearly with sigma ------------------------- #
    e = ax[2, 0]
    sig_grid = np.logspace(-8, -1, 12)
    meas = [measure_floor(lam, s, rng) for s in sig_grid]
    e.loglog(sig_grid, meas, "o", color="#2c6fbb", label="measured RMS distance")
    e.loglog(sig_grid, sink_noise_floor(sig_grid, lam), "-", color="crimson",
             label="$\\sigma\\sqrt{2/(1-\\lambda^2)}$")
    e.set_title(f"(e) sink: how close you can get ($\\lambda={lam}$)")
    e.set_xlabel("noise $\\sigma$"); e.set_ylabel("limiting $|\\mathbf{v}_k-\\mathbf{p}|$")
    e.legend(fontsize=8); e.grid(alpha=0.3, which="both")

    # ---- (f) escape time grows only logarithmically ------------------------- #
    f_ax = ax[2, 1]
    esc = [measure_escape(mu, s, 1.0, rng) for s in sig_grid]
    f_ax.semilogx(sig_grid, esc, "o", color="#2c6fbb", label="measured median escape time")
    f_ax.semilogx(sig_grid, source_escape_time(1.0, sig_grid, mu), "-", color="crimson",
                  label="$\\log(d\\sqrt{\\mu^2-1}/\\sigma\\sqrt{2})/\\log\\mu$")
    f_ax.set_title(f"(f) source: shrinking the noise barely helps ($\\mu={mu}$)")
    f_ax.set_xlabel("noise $\\sigma$"); f_ax.set_ylabel("iterations to leave $N_1(\\mathbf{p})$")
    f_ax.legend(fontsize=8); f_ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=140)
    print(f"figure written to {args.out}")

    # ---- numbers ------------------------------------------------------------ #
    print("\n--- sink ---------------------------------------------------------")
    print(f"contraction lambda = {lam}, noise sigma = {sigma:g}")
    print(f"  predicted floor   {sink_noise_floor(sigma, lam):.6e}")
    print(f"  measured  floor   {measure_floor(lam, sigma, rng):.6e}")
    print(f"  last distance     {d_noisy[-1]:.6e}   (never 0, never settles)")
    print(f"  min  distance     {d_noisy.min():.6e}   over {n} iterations")
    print("\n--- source -------------------------------------------------------")
    print(f"expansion mu = {mu}, starting exactly at the fixed point")
    for s in (1e-3, 1e-9, 1e-15):
        print(f"  sigma={s:8.1e} -> leaves N_1(p) after about "
              f"{source_escape_time(1.0, s, mu):5.1f} iterations")
    print("  (a million-fold quieter system only delays the escape by "
          f"{np.log(1e6)/np.log(mu):.0f} steps)")


if __name__ == "__main__":
    main()
