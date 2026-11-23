import utils
from utils import *
from sdf import *
from integrators.base_us import *

class MeshEchoIntegrator(UltrasoundEchoIntegrator):

    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)

        # Bounce count used when the caller does not pass an explicit max_depth.
        # 1 == the legacy single-bounce reference; >1 renders reverberation.
        self.n_bounces = 1
        # Per-bounce echo attenuation. Must match the SDF multi-bounce model
        # (SDFEchoIntegrator.bounce_reflectance) or the model
        # cannot fit its own ground truth.
        self.bounce_reflectance = 0.5
        # Reflect about the interpolated shading normal (the geometric analogue
        # of the smooth SDF gradient) rather than the faceted face normal.
        self.use_geometric_normal = False

    def render(self, scene, sdf, sensor=None, seed=0, spp=32, max_depth=None,
               radius_convolution=10.0, develop=True, active=True, **kwargs):
        """Use the configured mesh bounce count when the caller omits max_depth."""
        if max_depth is None:
            max_depth = int(getattr(self, "n_bounces", 1))
        return super().render(scene, sdf, sensor=sensor, seed=seed, spp=spp,
                              max_depth=int(max_depth),
                              radius_convolution=radius_convolution,
                              develop=develop, active=active, **kwargs)

    def sample(self,
        scene:mi.Scene, sdf:SDF,
        sampler:mi.Sampler, ray:mi.Ray3f, max_depth = 1,
        active = True, **kwargs
    ):
        """Multi-bounce monostatic mesh path (reference / ground truth).

        Conventions are identical to SDFEchoIntegrator.sample():
        a specular chain p0->p1->p2->..., where echo i is displayed at
        0.5 * (forward path p0..pi + straight return |pi - p0|), every echo is
        pinned to the primary scan line, and each successive echo is scaled by
        self.bounce_reflectance. max_depth=1 reproduces the previous
        single-bounce output (same expression tree; this integrator draws no
        random numbers, so the sampler state is untouched).

        Returns the legacy 5-tuple when max_depth == 1, otherwise a list of
        per-event 5-tuples, which UltrasoundEchoIntegrator.render() splats and sums.
        """
        n_bounces = max(1, int(max_depth))
        ray_eps = mi.Float(utils.RAY_EPS)

        p0 = ray.o                      # transducer origin: return target for all echoes
        cur_ray = Ray3f(ray)            # ray marched for the current bounce
        acc_fwd = mi.Float(0.0)         # cumulative forward path p0->p1->...->pi
        throughput = mi.Float(1.0)      # per-bounce echo attenuation
        active_path = mi.Bool(True)     # rays still bouncing (missed rays stop)
        no_event = dr.zeros(mi.Bool, dr.width(ray.o))  # no grazing term on meshes

        motion_primary = None           # lateral scan-line anchor (first hit)
        events = []

        for depth in range(n_bounces):
            # ------------------------- INTERSECT ---------------------------
            itx : SurfaceInteraction3f = scene.ray_intersect(cur_ray)
            hit = itx.is_valid() & active_path

            # sample emitter. sample_monostatic_point() uses no cosine term, so
            # back-facing hits need no special handling.
            itxL : mi.DirectionSample3f
            Li : mi.Color3f
            itxL, Li, r2 = self.sample_monostatic_point(
                ray=cur_ray,
                itx=itx,
                valid=hit,
                max_distance=self.max_distance,
                strength=mi.Float(1.0)
            )

            if depth == 0:
                # ---------------- PRIMARY EVENT (legacy path) --------------
                # check shadow ray
                rayL : Ray3f = Ray3f(itx.p, itxL.d)
                rayL.o += dr.normalize(rayL.d) * utils.RAY_EPS
                # Only consider occluders *between* the shading point and the sampled emitter point.
                # Otherwise, the intersection with the emitter itself can incorrectly mark the path as blocked.
                rayL.maxt = dr.maximum(itxL.dist - utils.ITX_EPS, mi.Float(0.0))
                itx_shadow : mi.SurfaceInteraction3f = scene.ray_intersect(rayL)
                valid_secondary = ~itx_shadow.is_valid()

                # direct illumination
                valid_ray = hit & valid_secondary
                L = dr.select(valid_ray, Li, black)

                distance = dr.select(valid_ray, dr.norm(ray.o - itx.p), dr.inf)

                motion_primary = itx.p
                events.append((dr.select(hit, L, utils.BACKGROUND), valid_ray,
                               distance, motion_primary, ~valid_ray))

                acc_fwd = dr.select(hit, dr.norm(itx.p - cur_ray.o), 0.0)
            else:
                # ------------------ REVERBERATION EVENT --------------------
                # No visibility test here, deliberately: this echo retraces the
                # bounce chain pi->...->p0, so a straight-line occlusion test to
                # the transducer would be blocked by the very interface that
                # created the reverberation and would delete the entire signal.
                # The SDF multi-bounce model likewise tests visibility at no depth.
                seg = dr.select(hit, dr.norm(itx.p - cur_ray.o), 0.0)
                acc_fwd = acc_fwd + seg
                ret = dr.select(hit, dr.norm(itx.p - p0), 0.0)
                dist_event = 0.5 * (acc_fwd + ret)

                L = dr.select(hit, Li * throughput, utils.BACKGROUND)
                distance = dr.select(hit, dist_event, dr.inf)

                # Lateral: pin every echo to the primary scan line. render()
                # reads only motion's lateral (x); depth comes from `distance`.
                events.append((L, hit, distance, motion_primary, no_event))

            # ------------------ SPAWN REFLECTED RAY ------------------------
            # sh_frame.n (interpolated vertex normal) is the geometric analogue
            # of the SDF's normalized gradient, which sdf.get_intersect() writes
            # into both si.sh_frame.n and si.n.
            n = itx.n if self.use_geometric_normal else itx.sh_frame.n
            n = dr.normalize(n)
            d = cur_ray.d
            # The reflection is even in n, so back-facing hits need no flip.
            d_refl = dr.normalize(d - 2.0 * dr.dot(d, n) * n)
            # scene.ray_intersect() applies no epsilon -> offset or self-hit at t~0.
            o_next = itx.p + ray_eps * d_refl
            cur_ray = Ray3f(o_next, d_refl)
            cur_ray.wavelengths = ray.wavelengths
            cur_ray.time = ray.time

            throughput = throughput * mi.Float(self.bounce_reflectance)
            active_path = active_path & hit

        # Keep the legacy 5-tuple for the single-bounce case; render() accepts
        # either form, but other callers may unpack the tuple directly.
        return events[0] if len(events) == 1 else events

mi.register_integrator("mesh-echo", MeshEchoIntegrator)
# Compatibility with existing Mitsuba scene files.
mi.register_integrator("direct-us", MeshEchoIntegrator)
