#!/usr/bin/env python3
"""Convergence study for the AD-vs-FD gradient validation: does agreement improve
with more sampling?

Two panels, both plotting cos(FD, AD) of the gradient image:

  (a) vs samples per pixel, at fixed film resolution.
      Also plots the *self-consistency* cos(AD_seed0, AD_seed1), which measures how
      much Monte Carlo noise is left. Once that curve reaches 1.0, extra spp cannot
      buy anything more -- whatever gap to 1.0 remains in cos(FD,AD) is systematic.

  (b) vs film resolution, at fixed spp, with the FD step scaled to the pixel size
      (eps = `--eps_px` pixels) so the reference stays comparable across resolutions.

Both are run for the single-bounce (m=1) and two-bounce (m=2) paths.

Usage (from the repo root, in the mitsuba env):
    python figures/gradcheck_convergence.py                  # measure + plot
    python figures/gradcheck_convergence.py --from_cache     # re-plot only
    python figures/gradcheck_convergence.py --axis z
"""
import os
import sys
import argparse
from os.path import join

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, join(REPO, "figures"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr

import utils
from utils import *          # noqa: F401,F403
from sdf import *            # noqa: F401,F403
from integrators import *    # noqa: F401,F403

from make_gradcheck_figure import (configure, ad_forward, fd_central, cosine, AXIS_VEC)

# categorical slots 1 & 2 (blue, orange) + a recessive gray for the noise reference
C_M1, C_M2, C_REF = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"


# --------------------------------------------------------------------------- #
#  Measurement
# --------------------------------------------------------------------------- #
def build(args):
    """Load config/transducer, integrator and the GradientSDF-wrapped scene SDF."""
    _, td = configure(args.scene, args.config)

    integ = mi.load_dict({"type": args.integrator})
    integ.set_max_distance(td["max_distance"])
    if hasattr(integ, "set_psf_sigma"):
        integ.set_psf_sigma(td["psf_sigma_lateral"], td["psf_sigma_axial"])
    if hasattr(integ, "bounce_reflectance"):
        integ.bounce_reflectance = args.reflectance

    base = SDF(); base.load(join(utils.DATA_DIR, args.scene, f"{args.scene}.npy"))
    sdf = GradientSDF(base)
    params = mi.traverse(sdf)
    poses = np.load(join(utils.DATA_DIR, args.scene, f"{args.pose_set}.npy"))
    return td, integ, sdf, params, poses


def sensor_at(td, poses, pose, res):
    return utils.get_ultrasound_sensors(
        resx=res, resy=res, td_width=td["td_width"], td_height=td["td_height"],
        poses=poses[pose:pose + 1])[0]


def measure(integ, sdf, params, sensor, axis, md, spp, eps, self_consistency=True):
    """cos(FD,AD) plus, optionally, the seed-to-seed self-consistency of each."""
    _, ad0 = ad_forward(integ, sdf, params, sensor, axis, md, spp, 0)
    fd0 = fd_central(integ, sdf, params, sensor, axis, eps, md, spp, 0)
    out = dict(cos_fd_ad=cosine(fd0, ad0),
               mag_ratio=float(np.linalg.norm(fd0.ravel()) /
                               (np.linalg.norm(ad0.ravel()) + 1e-12)))
    if self_consistency:
        _, ad1 = ad_forward(integ, sdf, params, sensor, axis, md, spp, 1)
        fd1 = fd_central(integ, sdf, params, sensor, axis, eps, md, spp, 1)
        out["cos_ad_ad"] = cosine(ad0, ad1)
        out["cos_fd_fd"] = cosine(fd0, fd1)
    return out


def run(args):
    td, integ, sdf, params, poses = build(args)
    axis = mi.Point3f(*[mi.Float(v) for v in AXIS_VEC[args.axis]])
    res = {}

    # ---- (a) spp sweep at fixed resolution -------------------------------- #
    print(f"[spp sweep] res={args.res} eps={args.eps:.2e}", flush=True)
    sensor = sensor_at(td, poses, args.pose, args.res)
    for md in args.max_depths:
        rows = []
        for spp in args.spps:
            r = measure(integ, sdf, params, sensor, axis, md, spp, args.eps)
            rows.append(r)
            print(f"  md={md} spp={spp:6d}  cos(FD,AD)={r['cos_fd_ad']:.4f}  "
                  f"cos(AD,AD)={r['cos_ad_ad']:.4f}  cos(FD,FD)={r['cos_fd_fd']:.4f}",
                  flush=True)
        for k in rows[0]:
            res[f"spp_{k}_md{md}"] = np.array([r[k] for r in rows])

    # ---- (b) resolution sweep, FD step scaled to the pixel ---------------- #
    print(f"[res sweep] spp={args.res_spp} eps={args.eps_px} px", flush=True)
    for md in args.max_depths:
        rows = []
        for r_ in args.resolutions:
            eps = args.eps_px * td["td_width"] / r_
            s = sensor_at(td, poses, args.pose, r_)
            r = measure(integ, sdf, params, s, axis, md, args.res_spp, eps,
                        self_consistency=False)
            rows.append(r)
            print(f"  md={md} res={r_:5d}  eps={eps:.2e}  "
                  f"cos(FD,AD)={r['cos_fd_ad']:.4f}  |FD|/|AD|={r['mag_ratio']:.4f}",
                  flush=True)
        for k in rows[0]:
            res[f"res_{k}_md{md}"] = np.array([r[k] for r in rows])

    res["spps"] = np.array(args.spps)
    res["resolutions"] = np.array(args.resolutions)
    res["max_depths"] = np.array(args.max_depths)
    return res


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def plot(d, args, out_base):
    fs = args.base_fs
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": fs, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    mds = list(d["max_depths"])
    col = {1: C_M1, 2: C_M2}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(args.figw, args.figh))

    # --- (a) vs spp --- #
    x = d["spps"]
    if len(mds):                      # MC noise reference (AD seed-to-seed), one curve
        axa.plot(x, d[f"spp_cos_ad_ad_md{mds[0]}"], color=C_REF, lw=1.4, ls=(0, (3, 2)),
                 marker="", zorder=1,
                 label="AD self-consistency\n(seed 0 vs 1)")
    for md in mds:
        axa.plot(x, d[f"spp_cos_fd_ad_md{md}"], color=col[md], lw=2.0,
                 marker="o", ms=4.0, zorder=3, label=f"$m={md}$")
        y = d[f"spp_cos_fd_ad_md{md}"][-1]
        axa.annotate(f"{y:.3f}", (x[-1], y), textcoords="offset points",
                     xytext=(4, -1 if md == 2 else 3), fontsize=fs - 1,
                     color=INK2, ha="left", va="center", annotation_clip=False)
    axa.set_xscale("log", base=2)
    axa.set_xlabel("samples per pixel")
    axa.set_ylabel(r"cos$(\partial I_{\mathrm{FD}},\ \partial I_{\mathrm{AD}})$")
    axa.set_title(f"(a) vs. sampling rate  (res {args.res})", fontsize=fs + 0.5, pad=4)
    axa.legend(frameon=False, fontsize=fs - 1, loc="lower right", handlelength=1.6,
               borderpad=0.2, labelspacing=0.45)

    # --- (b) vs film resolution --- #
    xr = d["resolutions"]
    for md in mds:
        axb.plot(xr, d[f"res_cos_fd_ad_md{md}"], color=col[md], lw=2.0,
                 marker="s", ms=4.0, zorder=3, label=f"$m={md}$")
        y = d[f"res_cos_fd_ad_md{md}"][-1]
        axb.annotate(f"{y:.3f}", (xr[-1], y), textcoords="offset points",
                     xytext=(4, -1 if md == 2 else 3), fontsize=fs - 1,
                     color=INK2, ha="left", va="center", annotation_clip=False)
    axb.set_xscale("log", base=2)
    axb.set_xlabel("film resolution (px)")
    axb.set_title(f"(b) vs. image resolution  (spp {args.res_spp})",
                  fontsize=fs + 0.5, pad=4)
    axb.legend(frameon=False, fontsize=fs - 1, loc="lower right", handlelength=1.6,
               borderpad=0.2)

    for ax in (axa, axb):
        ax.grid(True, which="major", color="#e2e1dd", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbab5"); ax.spines[s].set_linewidth(0.7)
        ax.tick_params(labelsize=fs - 1, length=2.5, color="#bbbab5", labelcolor=INK2)
        ax.set_ylim(args.ymin, 1.005)

    fig.suptitle(f"Gradient validation converges in spp but is limited by image "
                 f"resolution  ({args.scene}, pose {args.pose}, "
                 f"$\\partial/\\partial {args.axis}$)",
                 fontsize=fs + 1, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 0.985, 0.93))
    fig.savefig(out_base + ".png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="vertebrae")
    ap.add_argument("--config", default="gradient_validation")
    ap.add_argument("--integrator", default="sdf-echo")
    ap.add_argument("--pose_set", default="poses")
    ap.add_argument("--pose", type=int, default=121)
    ap.add_argument("--axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--reflectance", type=float, default=0.5)
    ap.add_argument("--max_depths", type=int, nargs="+", default=[1, 2])
    # panel (a)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--eps", type=float, default=5e-4)
    ap.add_argument("--spps", type=int, nargs="+",
                    default=[10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120])
    # panel (b)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--res_spp", type=int, default=1280)
    ap.add_argument("--eps_px", type=float, default=0.5,
                   help="FD step in pixels for the resolution sweep")
    # figure
    ap.add_argument("--figw", type=float, default=7.0)
    ap.add_argument("--figh", type=float, default=2.9)
    ap.add_argument("--base_fs", type=float, default=8.0)
    ap.add_argument("--ymin", type=float, default=0.78)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default=join(REPO, "figures", "gradcheck_convergence"))
    ap.add_argument("--from_cache", action="store_true",
                    help="skip rendering; reload from <out>_data.npz")
    args = ap.parse_args()

    cache = args.out + "_data.npz"
    if args.from_cache:
        d = dict(np.load(cache))
        print(f"[from_cache] {cache}")
    else:
        d = run(args)
        np.savez(cache, **d)
        print(f"[cached] {cache}")
    plot(d, args, args.out)
    print(f"saved: {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
