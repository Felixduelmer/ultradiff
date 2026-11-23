"""Fit an SDF to synthetic or recorded ultrasound observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

import utils
from utils import dr, mi
import integrators
from sdf import SphereSDF
from acquisition import image_size, load_phantom


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/ultrasound.json"


def make_sensors(config=None):
    if config is None:
        config = utils.load_render_config(DEFAULT_CONFIG)
    width, height = image_size(config)
    transducer = config["transducer"]
    # Two orthogonal sweeps across the superior surface, in normalized scene units.
    poses = []
    for angle in (0, np.pi / 2):
        lateral = np.array([np.cos(angle), np.sin(angle), 0])
        for offset in np.linspace(-0.18, 0.18, 9):
            center = np.array([0.5, 0.5, 0.5]) + offset * lateral
            transform = mi.ScalarTransform4f().look_at(
                origin=center + [0, 0, 0.48], target=center, up=lateral
            )
            poses.append(transform.matrix)
    return utils.get_ultrasound_sensors(poses=poses, resx=width, resy=height,
                                        td_width=transducer["td_width"], td_height=transducer["td_height"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--acquisition", type=Path, help="Tracked spine-phantom acquisition directory")
    inputs.add_argument("--mesh", type=Path, default=Path(__file__).parent / "example/sub-verse048_V23.stl")
    parser.add_argument("--out", type=Path, default=Path("runs/example"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--bounces", type=int, choices=[1, 2])
    args = parser.parse_args()
    config = utils.load_render_config(args.config)
    optimization = config["optimization"]
    rendering = config["rendering"]
    transducer = config["transducer"]
    if args.steps is not None:
        optimization["steps"] = args.steps
    if args.bounces is not None:
        rendering["bounces"] = args.bounces
    for key, value in (("steps", optimization["steps"]), ("grid_resolution", optimization["grid_resolution"]),
                       ("spp", rendering["spp"])):
        if type(value) is not int or value < 1:
            parser.error(f"{key} must be a positive integer")
    if rendering["bounces"] not in (1, 2):
        parser.error("bounces must be 1 or 2")
    interval = optimization.get("redistance_every", 0)
    if type(interval) is not int or interval < 0:
        parser.error("redistance_every must be a nonnegative integer")
    if interval:
        try:
            import fastsweep
        except ImportError as exc:
            parser.error(f"Redistancing requires a compatible fastsweep installation: {exc}")
    _, loss_rows = image_size(config)
    if args.acquisition:
        sensors, target_images, center, scale = load_phantom(args.acquisition, config)
        acquisition = config.setdefault("acquisition", {})
        fraction = acquisition.setdefault("retained_depth_fraction", 0.7)
        if not 0 < fraction <= 1:
            parser.error("retained_depth_fraction must be in (0, 1]")
        acquisition.update(frames=len(sensors), world_center_mm=center.tolist(), scene_units_per_mm=scale)
        loss_rows = max(1, round(config["acquisition"]["retained_depth_fraction"] * loss_rows))
        scene = None
    else:
        # Acquisition presets take their probe geometry from the recording.
        # When used with an STL, inherit missing dimensions from the synthetic probe.
        if any(key not in transducer for key in ("td_width", "td_height", "max_distance")):
            defaults = json.loads(DEFAULT_CONFIG.read_text())["transducer"]
            config["transducer"] = transducer = {**defaults, **transducer}
        mesh = trimesh.load_mesh(args.mesh)
        center = mesh.bounds.mean(axis=0)
        scale = 0.55 / mesh.extents.max()
        mesh.vertices = (mesh.vertices - center) * scale + 0.5
        sensors = make_sensors(config)
    batch = optimization.get("batch_size", 1)
    if type(batch) is not int or not 1 <= batch <= len(sensors):
        parser.error("batch_size must be between one and the number of views")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    renderer = mi.load_dict({"type": "sdf-echo"})
    renderer.set_max_distance(transducer["max_distance"])
    renderer.set_psf_sigma(transducer["psf_sigma_lateral"], transducer["psf_sigma_axial"])
    # SphereSDF builds its tensor from Dr.Jit's default meshgrid (Y, Z, X).
    # Expose the actual scene-space XYZ center in configuration files.
    cx, cy, cz = optimization.get("initial_center", [0.5, 0.5, 0.5])
    sdf = SphereSDF(res=optimization["grid_resolution"], radius=optimization["initial_radius"],
                    center=mi.Point3f(cy, cz, cx))
    if not args.acquisition:
        mesh_path = args.out / "target_normalized.ply"
        mesh.export(mesh_path)
        scene = mi.load_dict({"type": "scene", "target": {"type": "ply", "filename": str(mesh_path.resolve())}})
    options = dict(scene=scene, sdf=sdf, spp=rendering["spp"], max_depth=rendering["bounces"], radius_convolution=1.0)
    if not args.acquisition:
        reference = mi.load_dict({"type": "mesh-echo"})
        reference.set_max_distance(transducer["max_distance"])
        reference.set_psf_sigma(transducer["psf_sigma_lateral"], transducer["psf_sigma_axial"])
        with dr.suspend_grad():
            target_images = np.stack([np.asarray(reference.render(sensor=s, **options)) for s in sensors])
    if not np.isfinite(target_images).all() or target_images[:, :loss_rows].max() <= 0:
        raise RuntimeError("Reference sweep produced no usable echoes")
    params = mi.traverse(sdf)
    optimizer = mi.ad.Adam(lr=optimization["learning_rate"])
    optimizer["grid"] = params["grid"]
    params.update(optimizer)
    order = np.random.default_rng(0).permutation(len(sensors))
    stride = (len(sensors) + batch - 1) // batch
    lr = optimization["learning_rate"]
    for step in range(optimization["steps"]):
        if batch == 1:
            indices = [order[step % len(order)]]
        else:
            indices = [(j * stride + step % stride) % len(sensors) for j in range(batch)]
        loss = 0
        for i in indices:
            seed = step if batch == 1 else step * len(sensors) + i
            image = renderer.render(sensor=sensors[i], seed=seed, **options)
            image_loss = utils.l1_loss(image[:loss_rows], mi.TensorXf(target_images[i, :loss_rows]))
            loss += image_loss / batch
        dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        regularizer = optimization["laplacian_weight"] * utils.laplacian_loss(sdf.grid.tensor())
        dr.backward(regularizer, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        optimizer.step()
        if interval and step % interval == 0:
            distance = fastsweep.redistance(optimizer["grid"])
            optimizer["grid"] = mi.TensorXf(distance.array, shape=optimizer["grid"].shape)
        params.update(optimizer)
        if step in optimization.get("upsample_steps", []):
            # SDF.at() interpolates a node-centered grid with X as the fastest axis.
            from scipy.ndimage import zoom
            grid = np.asarray(optimizer["grid"])[..., 0]
            finer = zoom(grid, 2, order=1, prefilter=False)
            optimizer["grid"] = mi.TensorXf(finer[..., None])
            params.update(optimizer)
            sdf.res = finer.shape[0]
            lr *= optimization["learning_rate_factor"]
            optimizer.set_learning_rate({"grid": lr})
        if batch > 1 and step % 100 == 0:
            print(f"Update {step}: image MAE {float(np.asarray(loss)):.6f}, grid {sdf.res}", flush=True)
    grid = np.asarray(sdf.grid.tensor())[..., 0]
    if not np.isfinite(grid).all():
        raise RuntimeError("Optimization produced nonfinite SDF values")
    vertices, faces, _, _ = marching_cubes(grid, level=0)
    # Convert marching cubes' ZYX node indices to the XYZ coordinates used by SDF.at_trilerp.
    xyz = vertices[:, ::-1] / (grid.shape[0] - 1)
    reconstruction = trimesh.Trimesh((xyz - 0.5) / scale + center, faces, process=True)
    reconstruction.fix_normals()
    reconstruction.export(args.out / "reconstruction.stl")
    with dr.suspend_grad():
        predictions = np.stack([np.asarray(renderer.render(sensor=s, **options)) for s in sensors])
    np.savez_compressed(args.out / "sweep.npz", target=target_images, reconstruction=predictions)
    print(f"Saved {args.out / 'reconstruction.stl'}")


if __name__ == "__main__":
    main()
