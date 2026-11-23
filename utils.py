import json
import os
from os.path import join

import drjit as dr
import mitsuba as mi
import numpy as np


def _select_mitsuba_variant():
    """Prefer CUDA AD, but fall back to LLVM AD when CUDA is unavailable."""
    requested = os.environ.get("MITSUBA_VARIANT", "cuda_ad_rgb")
    candidates = [requested]
    if requested != "cuda_ad_rgb":
        candidates.append("cuda_ad_rgb")
    if "llvm_ad_rgb" not in candidates:
        candidates.append("llvm_ad_rgb")

    last_error = None
    for variant in candidates:
        try:
            mi.set_variant(variant)
            if variant != requested:
                print(f"[info] Mitsuba variant fallback: using '{variant}' (requested '{requested}').")
            return
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to initialize a Mitsuba AD variant from {candidates}: {last_error}"
    )

_select_mitsuba_variant()

from mitsuba import Bool, Mask, UInt, Int, Float

from mitsuba import Point2i, Point2f, Vector2i, Vector2f, Ray2f

from mitsuba import Point3i, Point3f, Vector3i, Vector3f, Ray3f

from mitsuba import Color3f, TensorXf, Interaction3f, SurfaceInteraction3f

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = join(MAIN_DIR, "data")

CONFIG_DIR = join(MAIN_DIR, "configs")

black = Color3f(0, 0, 0)

ITX_EPS = 1e-5

RAY_EPS = 1e-3

BACKGROUND = black

SDF_MODE = 'cubic'

SDF_EPS = 1e-4

SDF_DERIV_EPS = 1e-1

NUM_REFINE = 0

ITX_REFINE = 1e-5

GRZ_MODE = 'bisection'

NUM_GRZ = 12

def load_render_config(path):
    """Apply the tracing parameters used by the SDF and mesh integrators."""
    global ITX_EPS, RAY_EPS, SDF_MODE, NUM_REFINE, ITX_REFINE
    global SDF_EPS, SDF_DERIV_EPS, GRZ_MODE, NUM_GRZ
    with open(path) as stream:
        config = json.load(stream)
    rendering = config["rendering"]
    differentiation = config.get("differentiation", {})
    ITX_EPS = rendering["itx_eps"]
    RAY_EPS = rendering["ray_eps"]
    SDF_MODE = rendering.get("sdf_mode", "linear")
    NUM_REFINE = rendering.get("num_refine", 0)
    ITX_REFINE = rendering.get("itx_refine", 1e-5)
    SDF_EPS = differentiation.get("sdf_eps", 1e-4)
    SDF_DERIV_EPS = differentiation.get("sdf_deriv_eps", 1e-1)
    GRZ_MODE = differentiation.get("grz_mode", "bisection")
    NUM_GRZ = differentiation.get("num_grz", 12)
    return config


def l1_loss(img, ref_img):
    """L1 loss"""
    return dr.mean(dr.abs(img - ref_img))

def laplacian_loss(data, _=None):
    """ Laplacian regularization loss
        Code from https://github.com/rgl-epfl/differentiable-sdf-rendering """

    def linear_idx(p):
        p.x = dr.clamp(p.x, 0, data.shape[0] - 1)
        p.y = dr.clamp(p.y, 0, data.shape[1] - 1)
        p.z = dr.clamp(p.z, 0, data.shape[2] - 1)
        return p.z * data.shape[1] * data.shape[0] + p.y * data.shape[0] + p.x

    shape = data.shape
    z, y, x = dr.meshgrid(*[dr.arange(mi.Float, shape[i]) for i in range(3)], indexing='ij')
    p = mi.Point3i(x, y, z)
    c = dr.gather(mi.Float, data.array, linear_idx(p))
    vx0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(-1, 0, 0)))
    vx1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(1, 0, 0)))
    vy0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, -1, 0)))
    vy1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 1, 0)))
    vz0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, -1)))
    vz1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, 1)))
    laplacian = dr.sqr(c - (vx0 + vx1 + vy0 + vy1 + vz0 + vz1) / 6)
    return dr.sum(laplacian)

def get_ultrasound_sensors(*, resx, resy, poses, td_width, td_height, radius_filter=1.0):
    """Create an orthographic linear array for each supplied probe transform."""
    sensors = []
    for pose in poses:
        sensors.append(mi.load_dict({
            'type': 'orthographic',
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm',
                'width': resx, 'height': resy,
                'filter': {'type': 'tent', 'radius': radius_filter}  # ,'radius': 1.0}
            },
            'to_world': mi.ScalarTransform4f(pose) @ mi.ScalarTransform4f().scale([td_width/2, td_height/2, 1]),
        }))
    return sensors
