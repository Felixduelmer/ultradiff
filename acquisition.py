"""Load the tracked spine-phantom sweep in its calibrated coordinate frame."""
import json

import numpy as np
from skimage.transform import resize

import utils


def image_size(config):
    resolution = config["rendering"]["resolution"]
    if type(resolution) is int:
        resolution = [resolution, resolution]
    if not isinstance(resolution, list) or len(resolution) != 2 or any(
            type(x) is not int or x < 1 for x in resolution):
        raise ValueError("resolution must be a positive integer or [width, height]")
    return tuple(resolution)


def load_phantom(directory, config):
    with (directory / "td_setting.json").open() as stream:
        settings = json.load(stream)
    frames = np.load(directory / settings["images_file_train"], mmap_mode="r")
    poses = np.load(directory / settings["poses_file_train"]).astype(np.float64)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected an (N,H,W,3) uint8 phantom image stack")
    if not len(frames) or poses.shape != (len(frames), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Expected one finite 4×4 probe pose per image")
    td = config["transducer"]
    for key in ("td_width", "td_height", "max_distance"):
        td[key] = settings["transducer"][key]
        if not np.isfinite(td[key]) or td[key] <= 0:
            raise ValueError(f"Invalid phantom probe setting: {key}")

    # Physical tracking translations are in mm; probe dimensions in this
    # acquisition's metadata use 100 mm per scene unit.
    scale = 0.01
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5)
                        for z in (0, 1)]) * [td["td_width"], td["td_height"], td["max_distance"]]
    origins_mm = poses[:, :3, 3].copy()
    bounds = origins_mm[:, None, :] + (corners @ poses[:, :3, :3].transpose(0, 2, 1)) / scale
    center_mm = (bounds.min(axis=(0, 1)) + bounds.max(axis=(0, 1))) / 2
    poses[:, :3, 3] = (origins_mm - center_mm) * scale + 0.5
    width, height = image_size(config)
    sensors = utils.get_ultrasound_sensors(poses=poses, resx=width, resy=height,
                                           td_width=td["td_width"], td_height=td["td_height"])
    images = np.empty((len(frames), height, width), dtype=np.float32)
    for i, frame in enumerate(frames):
        gray = frame.astype(np.float32) @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        image = gray if gray.shape == (height, width) else resize(
            gray, (height, width), order=1, anti_aliasing=True, preserve_range=True)
        peak = image.max()
        if peak > 0:
            image = np.where(image >= 0.3 * peak, image * (0.7 / peak), 0)
        images[i] = image
    return sensors, images, center_mm, scale
