import utils
from utils import *
from sdf import *
from integrators.base_us import *


def _opaque_float(value):
    """Keep scalar constants out of generated kernels for better cache reuse."""
    v = mi.Float(value)
    dr.make_opaque(v)
    return v

class SDFEchoIntegrator(UltrasoundEchoIntegrator):
    """Render specular SDF echo paths with differentiable time of flight."""

    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)
        self.bounce_reflectance = 0.5

    def sample(self,
        scene:mi.Scene, sdf:SDF,
        sampler:mi.Sampler, ray:mi.Ray3f, max_depth = 2,
        active = True, **kwargs
    ):
        """Multi-bounce monostatic US path.

        Traces a specular reflection chain p0->p1->p2->... through the SDF.
        For each interface it records one echo event whose displayed distance is
        the forward bounce path plus a *straight* return to the transducer,
        halved: bounce 1 -> (p0->p1->p0)/2 = |p0-p1|; bounce 2 ->
        (p0->p1->p2->p0)/2; etc. All echoes are placed on the primary scan line
        (lateral of the first hit) to model US reverberation artifacts, and each
        successive echo is attenuated by `self.bounce_reflectance`.

        Returns a list of 5-tuples (L, mask, distance, motion, is_grz), one per
        bounce, matching the multi-bounce contract consumed by
        UltrasoundEchoIntegrator.render(). max_depth=1 reproduces the single-bounce
        behavior.
        """
        ray_eps = _opaque_float(utils.RAY_EPS)

        # Original transducer origin (constant): the straight return targets p0.
        p0 = dr.detach(ray.o)

        cur_ray = mi.Ray3f(ray)              # ray marched for the current bounce
        acc_fwd = mi.Float(0.0)              # cumulative forward path p0->p1->...->pi
        throughput = mi.Float(1.0)           # per-bounce echo attenuation
        active_path = mi.Bool(True)          # rays still bouncing (missed rays stop)
        no_event = dr.zeros(mi.Bool, dr.width(ray.o))  # no grazing/boundary term here

        motion_primary = None                # lateral scan-line anchor (first hit)
        events = []

        n_bounces = max(1, int(max_depth))
        for depth in range(n_bounces):
            # ----------------------- INTERSECT --------------------------------
            itx = sdf.ray_intersect(cur_ray, method="default")
            valid = itx.is_valid() & active_path

            # ---------------- REPARAMETRIZE HIT FOR AD ------------------------
            # Detached-normal reparametrization: makes itx.p differentiable wrt
            # the SDF for the (lateral / depth) gradients, primal unchanged.
            n_det = dr.detach(itx.n)
            v_geo = - sdf.at(itx.p) / dr.squared_norm(n_det) * n_det
            v = dr.select(valid, v_geo, Vector3f(0))
            itx.p = dr.replace_grad(itx.p, v)

            # --------- DISTANCE: forward path + straight return, halved -------
            # Use a floored radicand instead of dr.norm: a reflected ray is
            # spawned only RAY_EPS off the surface, so at depth >= 1 it can
            # re-hit at a near-zero distance, where d|v|/dv = v/|v| evaluates
            # 0/0 and reverse-mode scatters NaN into the SDF grid. Flooring
            # keeps the derivative finite; for every non-degenerate lane the
            # value is bit-identical to dr.norm, so depth 0 is unaffected.
            seg = dr.select(
                valid, dr.sqrt(dr.maximum(dr.squared_norm(itx.p - cur_ray.o), 1e-18)), 0.0)
            acc_fwd = acc_fwd + seg
            ret = dr.select(
                valid, dr.sqrt(dr.maximum(dr.squared_norm(itx.p - p0), 1e-18)), 0.0)
            dist_event = 0.5 * (acc_fwd + ret)

            # ------------------------- ECHO INTENSITY -------------------------
            _itxL, Li, _r = self.sample_monostatic_point(
                ray=cur_ray, itx=itx, valid=valid,
                max_distance=self.max_distance
            )
            L = dr.select(valid, Li * throughput, utils.BACKGROUND)

            # ----- LATERAL: pin every echo to the primary scan line -----------
            # render() reads only motion's lateral (x); depth comes from
            # dist_event. Reusing the first hit keeps reverbs in the same column.
            if motion_primary is None:
                motion_primary = itx.p
            motion = motion_primary

            events.append((L, valid, dist_event, motion, no_event))

            # ------------- SPAWN REFLECTED RAY (detached) ---------------------
            d = cur_ray.d
            d_refl = dr.normalize(d - 2.0 * dr.dot(d, n_det) * n_det)
            o_next = dr.detach(itx.p) + ray_eps * d_refl
            cur_ray = mi.Ray3f(dr.detach(o_next), dr.detach(d_refl))
            cur_ray.wavelengths = ray.wavelengths

            throughput = throughput * mi.Float(self.bounce_reflectance)
            active_path = active_path & valid

        # last_hit_points side-channel in render() uses events[0].
        return events
    
mi.register_integrator("sdf-echo", SDFEchoIntegrator)
# Compatibility with existing Mitsuba scene files.
mi.register_integrator("sdf-direct-us-diff-2b", SDFEchoIntegrator)
