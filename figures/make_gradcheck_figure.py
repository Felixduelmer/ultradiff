#!/usr/bin/env python3
"""SIGGRAPH gradient-validation figure for the multi-bounce US integrator.

Produces a 2x4 panel:

              B-mode        FD dI/dx     AD dI/dx     FD - AD
  m = 1      [ md=1 ]      [   .   ]    [   .   ]    [ diff ]
  m = 2      [ md=2 ]      [   .   ]    [   .   ]    [ diff ]

Row 1 is the single-bounce path (max_depth=1), row 2 the two-bounce path
(max_depth=2). Gradients are the image derivative w.r.t. a global SDF translation
along one axis (via GradientSDF): forward-mode AD vs central finite differences.

B-modes are shown as normalized intensity in [0, 1] with a millimetre scale bar.
The physical scale is derived from the transducer aperture: the image spans the
full transducer width, so `--td_width_mm` (default 51.3 mm = 0.513 world units x
100 mm/unit, the real probe used to build the scene) sets mm-per-pixel directly.

Two shared colorbars, stacked on the right: seismic (all six gradient panels,
including the difference, on one scale) above, grayscale (normalized B-mode) below.

Usage (from the repo root, in the mitsuba env):
    python figures/make_gradcheck_figure.py                       # pose 121, x-axis
    python figures/make_gradcheck_figure.py --axis z
    python figures/make_gradcheck_figure.py --scalebar_mm 5 --td_width_mm 51.3

See figures/gradcheck_convergence.py for how the default res/spp/eps were chosen.
"""
import os
import sys
import json
import argparse
from os.path import join

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr

import utils
from utils import *          # noqa: F401,F403  (Float, Point3f, DATA_DIR/CONFIG_DIR, ...)
from sdf import *            # noqa: F401,F403  (SDF, GradientSDF)
from integrators import *    # noqa: F401,F403  (registers sdf-echo)

AXIS_VEC = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


# --------------------------------------------------------------------------- #
#  Rendering / gradient helpers
# --------------------------------------------------------------------------- #
def configure(scene, config):
    """Load config + per-scene transducer settings and set the global eps knobs."""
    cfg = utils.load_render_config(join(utils.CONFIG_DIR, f"{config}.json"))
    td_path = join(utils.DATA_DIR, scene, "td_setting.json")
    td = cfg["transducer"]
    if os.path.exists(td_path):
        with open(td_path) as stream:
            td = json.load(stream)["transducer"]
    return cfg["rendering"], td


def _chan0(img):
    a = np.array(img, dtype=np.float32)
    return a[..., 0] if a.ndim == 3 else a


def render(integ, sdf, sensor, max_depth, spp, seed):
    return integ.render(scene=None, sdf=sdf, sensor=sensor,
                        seed=seed, spp=spp, max_depth=max_depth)


def ad_forward(integ, sdf, params, sensor, axis, max_depth, spp, seed):
    """Forward-mode AD: returns (primal image, directional derivative dI/dp.axis)."""
    dr.disable_grad(params["p"]); dr.enable_grad(params["p"])
    dr.set_grad(params["p"], axis)
    params.update()
    img = render(integ, sdf, sensor, max_depth, spp, seed)
    primal = _chan0(img)
    dr.forward_to(img)
    return primal, _chan0(dr.grad(img))


def fd_central(integ, sdf, params, sensor, axis, eps, max_depth, spp, seed):
    """Central finite-difference image derivative along axis (common random numbers)."""
    with dr.suspend_grad():
        p0 = mi.Point3f(params["p"])
        params["p"] = p0 + mi.Point3f(mi.Float(eps) * axis); params.update()
        ip = _chan0(render(integ, sdf, sensor, max_depth, spp, seed))
        params["p"] = p0 - mi.Point3f(mi.Float(eps) * axis); params.update()
        im = _chan0(render(integ, sdf, sensor, max_depth, spp, seed))
        params["p"] = p0; params.update()
    return (ip - im) / (2.0 * eps)


def cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


# --------------------------------------------------------------------------- #
#  Drawing helpers
# --------------------------------------------------------------------------- #
def add_scale_bar(ax, width_mm, depth_mm, length_mm, fontsize=6.5):
    """White horizontal scale bar (lower-left), drawn in millimetre data coords
    (panels use a mm extent with equal aspect, so pixels are isotropic)."""
    x0, y0 = 0.06 * width_mm, 0.935 * depth_mm
    ax.add_patch(Rectangle((x0, y0), length_mm, 0.02 * depth_mm,
                           facecolor="white", edgecolor="black", linewidth=0.4, zorder=5))
    # left-align at the bar start so the (wide) label never clips on a small panel
    ax.text(x0, y0 - 0.012 * depth_mm, f"{length_mm:g} mm",
            color="white", ha="left", va="bottom", fontsize=fontsize, zorder=6, clip_on=False)


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def make_figure(rows, gmax, axis_name, meta, out_base):
    # Authored at the paper's single-column width so \includegraphics[width=\columnwidth]
    # renders it 1:1; point sizes are chosen for that final size.
    fs = meta["base_fs"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": fs,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in the PDF
    })
    dIdx = r"$\partial I/\partial %s$" % axis_name
    col_titles = ["B-mode", "FD  " + dIdx, "AD  " + dIdx, r"FD $-$ AD"]

    # panels carry a millimetre extent (x: aperture, y: depth) with equal aspect,
    # so the displayed pixels are physically isotropic.
    w_mm, d_mm = meta["width_mm"], meta["depth_mm"]
    extent = [0.0, w_mm, d_mm, 0.0]

    fig = plt.figure(figsize=(meta["figw"], meta["figh"]))
    gs = GridSpec(2, 4, figure=fig, wspace=meta["wspace"], hspace=meta["hspace"],
                  left=0.075, right=0.855, top=0.905, bottom=0.035)
    gray_h = seis_h = None
    row_ax = {}   # one panel per row, to align the colorbars to real drawn heights

    for ri, row in enumerate(rows):
        panels = [("bmode_disp", "gray", 0.0, 1.0),
                  ("fd", "seismic", -gmax, gmax),
                  ("ad", "seismic", -gmax, gmax),
                  ("diff", "seismic", -gmax, gmax)]
        for ci, (key, cmap, vmn, vmx) in enumerate(panels):
            ax = fig.add_subplot(gs[ri, ci])
            h = ax.imshow(row[key], cmap=cmap, vmin=vmn, vmax=vmx, extent=extent,
                          aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#999"); s.set_linewidth(0.4)
            if ci == 0:
                gray_h = h
                ax.set_ylabel(row["label"], fontsize=fs + 1, labelpad=3)
                add_scale_bar(ax, w_mm, d_mm, meta["scalebar_mm"], fontsize=fs - 1)
            elif ci == 1:
                seis_h = h
            if ci == 3:
                row_ax[ri] = ax
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=fs + 0.5, pad=3)
            if key == "diff":
                ax.text(0.04, 0.95, f"cos = {row['cos']:.3f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=fs - 0.5, color="#111",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bbb", alpha=0.85))

    # --- two stacked colorbars on the right, each aligned to a row's height ---
    # aspect="equal" resizes the panels, so query their true positions after a draw.
    fig.canvas.draw()
    b0, b1 = row_ax[0].get_position(), row_ax[1].get_position()
    cb_x = max(b0.x1, b1.x1) + 0.006
    cb_w = 0.020

    seis_cax = fig.add_axes([cb_x, b0.y0, cb_w, b0.height])   # top row -> gradients
    cb2 = fig.colorbar(seis_h, cax=seis_cax)
    cb2.set_label(dIdx, fontsize=fs); cb2.ax.tick_params(labelsize=fs - 1.5, length=2)

    gray_cax = fig.add_axes([cb_x, b1.y0, cb_w, b1.height])   # bottom row -> B-mode
    cb1 = fig.colorbar(gray_h, cax=gray_cax)
    cb1.set_label("Intensity", fontsize=fs); cb1.ax.tick_params(labelsize=fs - 1.5, length=2)
    cb1.set_ticks([0, 0.5, 1.0])

    # Put both colorbar labels the same distance from their bars, independent of
    # differing tick-label widths, so "dI/dx" and "Intensity" line up vertically.
    fig.canvas.draw()

    def _ticklabel_right(cb):
        x1 = 0.0
        for lbl in cb.ax.get_yticklabels():
            bb = lbl.get_window_extent().transformed(fig.transFigure.inverted())
            x1 = max(x1, bb.x1)
        return x1

    label_x = max(_ticklabel_right(cb2), _ticklabel_right(cb1)) + 0.002
    cb2.ax.yaxis.set_label_coords(label_x, b0.y0 + b0.height / 2.0, transform=fig.transFigure)
    cb1.ax.yaxis.set_label_coords(label_x, b1.y0 + b1.height / 2.0, transform=fig.transFigure)

    # dpi sets the raster resolution of the imshow panels embedded in the PDF
    # (text/lines stay vector). Default figure dpi (100) would downsample them.
    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.02, dpi=meta["dpi"])
    fig.savefig(out_base + ".png", bbox_inches="tight", pad_inches=0.02, dpi=meta["dpi"])
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
    # res/spp/eps are set from the convergence study in figures/gradcheck_convergence.py:
    # spp saturates by ~1280, the FD-AD agreement is limited by film resolution, and the
    # best FD step is ~0.5 px (= 0.5 * td_width / res = 2.5e-4 at res 1024). Lowering res
    # without also raising eps to ~0.5 px will understate the agreement.
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--spp", type=int, default=1280)
    ap.add_argument("--eps", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reflectance", type=float, default=0.5)
    ap.add_argument("--unit_to_mm", type=float, default=100.0,
                    help="scene(world) units -> mm (matches evaluate_extra_mesh: 100)")
    ap.add_argument("--td_width_mm", type=float, default=None,
                    help="physical transducer aperture in mm; default derives it from "
                         "td_width * unit_to_mm (0.513 * 100 = 51.3 mm)")
    ap.add_argument("--scalebar_mm", type=float, default=10.0,
                    help="scale-bar length in millimetres")
    # wider than a single column so the aspect-equal panels fill the height and the
    # two rows sit close together (scale in LaTeX with \includegraphics[width=...]).
    ap.add_argument("--figw", type=float, default=4.1, help="figure width in inches")
    ap.add_argument("--figh", type=float, default=2.15, help="figure height in inches")
    ap.add_argument("--base_fs", type=float, default=7.0, help="base font size in pt")
    ap.add_argument("--wspace", type=float, default=0.05, help="inter-column spacing")
    ap.add_argument("--hspace", type=float, default=0.02, help="inter-row spacing")
    ap.add_argument("--dpi", type=int, default=600,
                    help="raster dpi for the embedded imshow panels (PDF + PNG)")
    ap.add_argument("--gmax_scale", type=float, default=2.0,
                    help="multiplier on the shared gradient colour scale; >1 makes the "
                         "FD-AD difference fade more toward white")
    ap.add_argument("--out", default=join(REPO, "figures", "gradcheck_1v2bounce"))
    ap.add_argument("--from_cache", action="store_true",
                    help="skip rendering; reload gradients from <out>_data.npz (fast layout iteration)")
    args = ap.parse_args()

    cache_path = args.out + "_data.npz"
    if args.from_cache:
        d = np.load(cache_path, allow_pickle=False)
        td_width_mm, depth_mm = float(d["td_width_mm"]), float(d["depth_mm"])
        rows = [dict(label=str(d[f"label{i}"]), bmode=d[f"bmode{i}"], fd=d[f"fd{i}"],
                     ad=d[f"ad{i}"], diff=d[f"fd{i}"] - d[f"ad{i}"], cos=float(d[f"cos{i}"]))
                for i in range(2)]
        print(f"[from_cache] {cache_path}")
    else:
        _, td = configure(args.scene, args.config)
        td_width_mm = args.td_width_mm if args.td_width_mm is not None else td["td_width"] * args.unit_to_mm
        depth_mm = td["max_distance"] * (td_width_mm / td["td_width"])
        print(f"transducer aperture = {td_width_mm:.1f} mm (image width) ; "
              f"depth = {depth_mm:.1f} mm ; {td_width_mm / args.res:.4f} mm/px lateral")

        P = np.load(join(utils.DATA_DIR, args.scene, f"{args.pose_set}.npy"))
        sensor = utils.get_ultrasound_sensors(
            resx=args.res, resy=args.res, td_width=td["td_width"],
            td_height=td["td_height"], poses=P[args.pose:args.pose + 1])[0]

        integ = mi.load_dict({"type": args.integrator})
        integ.set_max_distance(td["max_distance"])
        if hasattr(integ, "set_psf_sigma"):
            integ.set_psf_sigma(td["psf_sigma_lateral"], td["psf_sigma_axial"])
        if hasattr(integ, "bounce_reflectance"):
            integ.bounce_reflectance = args.reflectance

        base = SDF(); base.load(join(utils.DATA_DIR, args.scene, f"{args.scene}.npy"))
        sdf = GradientSDF(base)
        params = mi.traverse(sdf)
        axis = mi.Point3f(*[mi.Float(v) for v in AXIS_VEC[args.axis]])

        rows = []
        for md, label in [(1, "m = 1"), (2, "m = 2")]:
            bmode, ad = ad_forward(integ, sdf, params, sensor, axis, md, args.spp, args.seed)
            fd = fd_central(integ, sdf, params, sensor, axis, args.eps, md, args.spp, args.seed)
            c = cosine(fd, ad)
            rel = float(np.linalg.norm((fd - ad).ravel()) / (np.linalg.norm(fd.ravel()) + 1e-12))
            rows.append(dict(label=label, bmode=bmode, fd=fd, ad=ad, diff=fd - ad, cos=c))
            print(f"[md={md}] cos_sim={c:.4f}  rel_L2={rel:.4f}")

        np.savez(cache_path, td_width_mm=td_width_mm, depth_mm=depth_mm,
                 **{f"label{i}": rows[i]["label"] for i in range(2)},
                 **{f"bmode{i}": rows[i]["bmode"] for i in range(2)},
                 **{f"fd{i}": rows[i]["fd"] for i in range(2)},
                 **{f"ad{i}": rows[i]["ad"] for i in range(2)},
                 **{f"cos{i}": rows[i]["cos"] for i in range(2)})
        print(f"[cached] {cache_path}")

    # shared scales
    bmax = max(float(np.percentile(np.concatenate([x["bmode"].ravel() for x in rows]), 99.6)), 1e-6)
    for x in rows:
        x["bmode_disp"] = np.clip(x["bmode"] / bmax, 0.0, 1.0)   # normalized B-mode in [0,1]
    all_g = np.concatenate([np.abs(x[k].ravel()) for x in rows for k in ("fd", "ad", "diff")])
    gmax = max(float(np.percentile(all_g, 99.5)), 1e-6) * args.gmax_scale

    meta = dict(res=args.res, width_mm=td_width_mm, depth_mm=depth_mm,
                scalebar_mm=args.scalebar_mm, figw=args.figw, figh=args.figh,
                base_fs=args.base_fs, wspace=args.wspace, hspace=args.hspace, dpi=args.dpi)
    make_figure(rows, gmax, args.axis, meta, args.out)
    print(f"saved: {args.out}.pdf / .png   (pose {args.pose}, axis {args.axis}, "
          f"eps {args.eps}, spp {args.spp})")


if __name__ == "__main__":
    main()
