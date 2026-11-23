import math
from utils import *
from sdf import SDF


def _opaque_float(value):
    """Keep scalar constants out of generated kernels for better cache reuse."""
    v = mi.Float(value)
    dr.make_opaque(v)
    return v


class UltrasoundEchoIntegrator(mi.SamplingIntegrator):
    
    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)

        self.max_distance = 2.0
        self.sigma_lateral = 2.0 #1.0   # PSF sigma in pixels (lateral / x)
        self.sigma_axial = 2.0 #2.0     # PSF sigma in pixels (axial / depth / y)

    def sample_monostatic_point(self, ray: mi.Ray3f,
                                itx: mi.SurfaceInteraction3f,
                                valid: mi.Bool,
                                max_distance: mi.Float,
                                strength: mi.Float = mi.Float(1.0)):

        eps = _opaque_float(1e-8)
        o = ray.o
        p = itx.p
        dvec = o - p
        r = dr.norm(dvec)
        wi = dvec / dr.maximum(r, eps)

        Li = mi.Color3f(strength)
        Li = dr.select(valid, Li, mi.Color3f(0.0))

        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = o
        ds.d = wi
        ds.dist = r
        ds.pdf = mi.Float(1.0)

        return ds, Li, r
    
    def sample(self, 
        scene:mi.Scene, sdf:SDF, 
        sampler:mi.Sampler, ray:mi.Ray3f, max_depth = 2, 
        active = True, **kwargs
    ):        
        pass

    def depth_to_film_y(self, dist, usable_h):
        """Map one-way distance to film y coordinate."""
        return dist / self.max_distance * usable_h

    def set_max_distance(self, max_distance):
        self.max_distance = max_distance

    def set_psf_sigma(self, sigma_lateral, sigma_axial):
        """Set PSF sigma values in pixels.
        
        Args:
            sigma_lateral: Gaussian sigma in lateral direction (x/width)
            sigma_axial: Gaussian sigma in axial direction (y/depth)
        """
        self.sigma_lateral = float(sigma_lateral)
        self.sigma_axial = float(sigma_axial)

    
    def _splat_with_boundary(self, pos, intensity, active, film_w, film_h,
                            sigma_lateral=5.0, sigma_axial=2.0, spp=25,
                            lateral_padding=0, axial_padding_bottom=0):
        """
        Splat hitpoints onto a film with Gaussian kernels, boundary correction,
        and proper gradient flow through hitpoint positions.

        Key design: the pixel grid indices are computed from DETACHED positions
        (no gradient), but the Gaussian weights use the ORIGINAL (differentiable)
        positions for the distance computation. This means:
        - Which pixels receive contributions: fixed (no discontinuous jumps)
        - How much each pixel receives: smooth function of pos (gradient flows)

        The gradient of the Gaussian weight w.r.t. position p is:
            dw/dp = w * (pixel_center - p) / sigma^2
        This gives a smooth, physically meaningful gradient in both lateral
        and axial directions.

        Args:
            pos:           mi.Point2f - hitpoint positions in pixel coords
                        (x = lateral, y = axial/depth)
            intensity:     mi.Float   - amplitude at each hitpoint
            active:        mi.Bool    - mask for valid hitpoints
            film_w:        int        - film width in pixels (lateral)
            film_h:        int        - film height in pixels (axial)
            sigma_lateral: float      - Gaussian sigma lateral direction (pixels)
            sigma_axial:   float      - Gaussian sigma axial direction (pixels)
            spp:           int        - rays per A-line (ray density normalization)
            lateral_padding: int      - extra internal pixels on left/right
                                        (returned image is still film_w wide)
            axial_padding_bottom: int - extra internal pixels at the bottom
                                        (ray/depth direction; returned image is
                                        still film_h high)

        Returns:
            mi.TensorXf of shape (film_h, film_w) - the reconstructed image
        """

        pad_x = max(0, int(lateral_padding))
        pad_y_bottom = max(0, int(axial_padding_bottom))
        film_w_ext = int(film_w + 2 * pad_x)
        film_h_ext = int(film_h + pad_y_bottom)

        n_pixels = film_w_ext * film_h_ext
        image = dr.zeros(mi.Float, n_pixels)

        # Hitpoint data
        # Shift to the padded domain. The output is cropped back later.
        px = pos.x + mi.Float(pad_x)
        py = pos.y

        # Sanitize active mask
        active = active & dr.isfinite(px) & dr.isfinite(py)
        active = active & (px >= 0) & (px < film_w_ext) & (py >= 0) & (py < film_h_ext)

        # Kernel setup — truncate at 3 sigma
        radius_lat = int(3.0 * sigma_lateral + 0.5)
        radius_ax = int(3.0 * sigma_axial + 0.5)

        inv_2s2_lat = 1.0 / (2.0 * sigma_lateral * sigma_lateral)
        inv_2s2_ax = 1.0 / (2.0 * sigma_axial * sigma_axial)

        # Analytic normalization for continuous 2D Gaussian
        analytic_norm = 2.0 * dr.pi * sigma_lateral * sigma_axial

        # ---------------------------------------------------------------
        # DETACH the integer grid anchor
        # ---------------------------------------------------------------
        # We use floor of the detached position as the anchor pixel.
        # This determines WHICH pixels are in the kernel footprint.
        # This is detached so the footprint doesn't jump discontinuously.
        #
        # The actual Gaussian distance computation uses the original
        # (differentiable) px, py — so gradients flow through the weights.

        anchor_x = mi.Int32(dr.floor(dr.detach(px)))
        anchor_y = mi.Int32(dr.floor(dr.detach(py)))

        # ---------------------------------------------------------------
        # Step 1: Per-hitpoint boundary correction (DETACHED)
        # ---------------------------------------------------------------
        px_d = dr.detach(px)
        py_d = dr.detach(py)

        truncated_mass = mi.Float(0.0)
        full_mass = mi.Float(0.0)

        for dy in range(-radius_ax, radius_ax + 1):
            for dx in range(-radius_lat, radius_lat + 1):
                tgt_x = anchor_x + dx
                tgt_y = anchor_y + dy

                in_bounds = (active &
                            (tgt_x >= 0) & (tgt_x < film_w_ext) &
                            (tgt_y >= 0) & (tgt_y < film_h_ext))

                # Distance using detached positions (for boundary correction only)
                dx_f = mi.Float(tgt_x) + 0.5 - px_d
                dy_f = mi.Float(tgt_y) + 0.5 - py_d

                w = dr.exp(-(dx_f * dx_f * inv_2s2_lat +
                            dy_f * dy_f * inv_2s2_ax))

                full_mass += dr.select(active, w, 0.0)
                truncated_mass += dr.select(in_bounds, w, 0.0)

        boundary_ratio = truncated_mass / dr.maximum(full_mass, 1e-10)
        boundary_ratio = dr.maximum(boundary_ratio, 1e-10)

        # ---------------------------------------------------------------
        # Step 2: Splat with gradient-carrying Gaussian weights
        # ---------------------------------------------------------------
        # per_hit_scale is detached from pos (only intensity and normalization)
        # The Gaussian weight w is the ONLY thing carrying pos gradient.

        per_hit_scale = intensity / (mi.Float(spp) * analytic_norm * boundary_ratio)

        for dy in range(-radius_ax, radius_ax + 1):
            for dx in range(-radius_lat, radius_lat + 1):
                # Target pixel: from detached anchor (no gradient on which pixel)
                tgt_x = anchor_x + dx
                tgt_y = anchor_y + dy

                in_bounds = (active &
                            (tgt_x >= 0) & (tgt_x < film_w_ext) &
                            (tgt_y >= 0) & (tgt_y < film_h_ext))

                # Distance from DIFFERENTIABLE pos to pixel center
                # THIS is where the gradient lives
                #   dx_f = (constant pixel center) - px  →  d(dx_f)/d(px) = -1
                #   dy_f = (constant pixel center) - py  →  d(dy_f)/d(py) = -1
                dx_f = mi.Float(tgt_x) + 0.5 - px
                dy_f = mi.Float(tgt_y) + 0.5 - py

                # Gaussian weight — gradient flows through here
                #   dw/dpx = w * 2 * dx_f * inv_2s2_lat = w * dx_f / sigma_lat^2
                #   dw/dpy = w * 2 * dy_f * inv_2s2_ax  = w * dy_f / sigma_ax^2
                w = dr.exp(-(dx_f * dx_f * inv_2s2_lat +
                            dy_f * dy_f * inv_2s2_ax))

                idx = mi.UInt32(tgt_y * film_w_ext + tgt_x)

                # Splat: per_hit_scale has no pos gradient, w has pos gradient
                dr.scatter_reduce(dr.ReduceOp.Add, image,
                                per_hit_scale * w, idx, active=in_bounds)

        if pad_x == 0 and pad_y_bottom == 0:
            return mi.TensorXf(image, shape=(film_h, film_w_ext))

        # Crop back to original image size so callers always get (film_h, film_w).
        yy = dr.repeat(dr.arange(mi.UInt32, film_h), film_w)
        xx = dr.tile(dr.arange(mi.UInt32, film_w), film_h) + mi.UInt32(pad_x)
        idx_crop = yy * mi.UInt32(film_w_ext) + xx
        cropped = dr.gather(mi.Float, image, idx_crop)
        return mi.TensorXf(cropped, shape=(film_h, film_w))

        
    def prepare_sampler(self, sensor, film_size, seed, spp):
        """Prepare a sampler forked from the sensor sampler."""
        """Code developed from https://github.com/rgl-epfl/differentiable-sdf-rendering"""
        wavefront_size = dr.prod(film_size) * spp
        sampler = sensor.sampler().clone()
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        # Keep seed symbolic/opaque so changing it per epoch does not force
        # a new kernel variant through sampler state initialization.
        seed_opaque = mi.UInt64(seed)
        dr.make_opaque(seed_opaque)
        sampler.seed(seed_opaque, wavefront_size)
        return sampler
    
    def sample_ray(self, sensor, sampler, spp):
        """ Sample a batch of rays from the sensor.
            Returns ray, ray_weight, position_sample """
        """ Code developed from https://github.com/rgl-epfl/differentiable-sdf-rendering """
        film = sensor.film()
        film_size = film.crop_size()
        film_size.y = 1
        border_size = film.rfilter().border_size()  
        if film.sample_border():
            film_size += 2 * border_size
        film_offset = film.crop_offset()
        
        # Compute the pixel index
        film_w = mi.Int32(int(film_size[0]))
        idx = dr.arange(mi.Int32, dr.prod(film_size) * spp)
        idx //= dr.opaque(mi.Int32, spp)
        
        # Compute the position on the image plane
        position = mi.Vector2i()
        position.y = idx // film_w
        position.x = dr.fma(-film_w, position.y, idx)
        position += mi.Vector2i(film.crop_offset())
        if film.sample_border():
            position -= border_size
        
        # Compute the random offset within the pixel
        offset = sampler.next_2d()
        
        # Compute the sample position
        position_sample = position + offset
        
        # Rescale the sample position in [0, 1]^2
        position_rescaled = (position_sample - film_offset) / mi.Vector2f(film_size)
        
        # Shutter open time
        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()
        
        # Wavelength sample of continuous spectrum
        wavelength_sample = sampler.next_1d()
        
        # Aperture sample for depth of field
        aperture_sample = mi.Point2f(0.5)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()
        
        # Sample the ray
        ray, ray_weight = sensor.sample_ray_differential(time, wavelength_sample, position_rescaled, aperture_sample)
        
        # normalize ray direction
        ray.d = dr.normalize(ray.d)
        
        # return ray, 0, position_sample, aperture_sample
        return ray, ray_weight, position_sample
 
    def render(self, 
        scene:mi.Scene, sdf:SDF, 
        sensor:mi.Sensor=None, seed=0, spp=32, max_depth=2, radius_convolution= 10.0,
        develop=True, active=True, **kwargs
    ):
        # prepare sensor
        if sensor is None:
            sensor = scene.sensors()[0]

        # prepare film
        film : mi.Film = sensor.film()
        film_size = film.crop_size()
        border_size = film.rfilter().border_size()
        if film.sample_border():
            film_size += 2 * border_size
        # Store Python values before converting to mi.Vector2f
        film_size_original_w = int(film_size[0])
        film_size_original_h = int(film_size[1])
        film_size.y = 1
        film.prepare([])

        # prepare sampler
        sampler = self.prepare_sampler(sensor, film_size, seed, spp)

        # sample a batch of rays
        ray, ray_weight, position_sample = self.sample_ray(sensor, sampler, spp)

        # scale the ray differentials
        diff_scale_factor = dr.rsqrt(mi.ScalarFloat(spp))
        ray.scale_differential(diff_scale_factor)
        
        # compute ray contribution
        # motion is the point that we reparametrize
        out = self.sample(scene, sdf, sampler, ray, max_depth)

        # Backward compat: single-bounce integrators return one 5-tuple
        # (L, mask, distance, motion, is_grz); multi-bounce integrators return
        # a LIST of such 5-tuples, one per interface/event along the ray.
        events = out if isinstance(out, list) else [out]

        # Store hit points as a side-channel for near-surface regularization.
        # last_hit_points: all ray positions (valid and invalid); use last_hit_mask to filter.
        # The PRIMARY (first) event is used, matching the single-bounce behavior.
        L0, mask0, distance0, motion0, is_grz0 = events[0]
        self.last_hit_points = motion0
        self.last_hit_mask   = mask0 | is_grz0

        usable_w = film_size_original_w
        usable_h = film_size_original_h
        max_event_distance = dr.maximum(self.max_distance - 1e-6, 0.0)
        to_world_inv = sensor.world_transform().inverse()

        lateral_padding = int(math.ceil(3.0 * max(float(self.sigma_lateral), 0.0)))
        axial_padding_bottom = int(math.ceil(3.0 * max(float(self.sigma_axial), 0.0)))

        # Splat each event independently and sum. Summing per-event images is
        # identical to one combined splat (additive scatter, per-hit boundary
        # correction), and spp stays spp because all events share the same rays
        # and land at different depths.
        intensity_img = None
        for (L, mask, distance, motion, is_grz) in events:
            # --- Lateral position (AD-tracked through reparameterized itx.p) ---
            local = to_world_inv.transform_affine(motion)
            pixel_x = (local.x + 1) * 0.5 * usable_w

            # Keep only physically valid range events. This avoids piling
            # invalid / out-of-range lanes into the bottom row via clamping.
            has_event = ((mask | is_grz)
                         & dr.isfinite(distance)
                         & (distance >= 0.0)
                         & (distance < max_event_distance))

            # --- Film y position (AD-tracked for valid events) ---
            distance_event = dr.select(has_event, distance, 0.0)
            y_event = self.depth_to_film_y(distance_event, usable_h)

            pos = mi.Point2f(pixel_x, y_event)
            ev_img = self._splat_with_boundary(
                pos, L.x, has_event,
                film_size_original_w, film_size_original_h,
                sigma_lateral=self.sigma_lateral,
                sigma_axial=self.sigma_axial,
                spp=spp,
                lateral_padding=lateral_padding,
                axial_padding_bottom=axial_padding_bottom
            )
            intensity_img = ev_img if intensity_img is None else intensity_img + ev_img

        return intensity_img

    def traverse(self, callback):
        super().traverse(callback)
    
    def parameters_changed(self, keys):
        super().parameters_changed(keys)

    def to_string(self):
        return f"{type(self).__name__}[]"

mi.register_integrator("ultrasound-echo", UltrasoundEchoIntegrator)
# Compatibility with existing Mitsuba scene files.
mi.register_integrator("sdf-us-base", UltrasoundEchoIntegrator)
