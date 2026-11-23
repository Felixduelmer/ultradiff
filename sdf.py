import utils
from utils import *    


def _opaque_float(value):
    """Keep scalar constants out of generated kernels for better cache reuse."""
    return dr.opaque(Float, float(value))


class SDF(mi.Object):
    """ Signed Distance Function of an object defined on 3D voxel grid.
        param res: resolution of the voxel grid
        param grid: voxel grid representing the SDF 
        param bbox: bounding box of the voxel grid"""
    
    def __init__(self):
        mi.Object.__init__(self)
        # load bounding box
        self.bbox = mi.BoundingBox3f(Point3f(0, 0, 0), Point3f(1, 1, 1))

    def at_trilerp(self, p: Point3f) -> Float:

        T = self.grid.tensor()          # TensorXf (res,res,res,1)
        A = T.array                     # flat Float array

        res = int(self.res)

        # clamp q to [0, 1) to avoid x1=res out of bounds
        one_minus_eps = _opaque_float(1.0 - 1e-7)
        qx = dr.clamp(p.x, 0.0, one_minus_eps)
        qy = dr.clamp(p.y, 0.0, one_minus_eps)
        qz = dr.clamp(p.z, 0.0, one_minus_eps)

        x = qx * (res - 1)
        y = qy * (res - 1)
        z = qz * (res - 1)

        x0 = mi.UInt32(dr.floor(x)); x1 = mi.UInt32(dr.minimum(mi.Float(x0 + 1), res - 1))
        y0 = mi.UInt32(dr.floor(y)); y1 = mi.UInt32(dr.minimum(mi.Float(y0 + 1), res - 1))
        z0 = mi.UInt32(dr.floor(z)); z1 = mi.UInt32(dr.minimum(mi.Float(z0 + 1), res - 1))

        fx = x - mi.Float(x0)
        fy = y - mi.Float(y0)
        fz = z - mi.Float(z0)

        # linear index (x fastest, then y, then z), ignoring the last channel dim (1)
        def lin(ix, iy, iz):
            return mi.UInt32(ix + res * (iy + res * iz))

        def g(ix, iy, iz):
            return dr.gather(mi.Float, A, lin(ix, iy, iz))

        c000 = g(x0,y0,z0); c100 = g(x1,y0,z0)
        c010 = g(x0,y1,z0); c110 = g(x1,y1,z0)
        c001 = g(x0,y0,z1); c101 = g(x1,y0,z1)
        c011 = g(x0,y1,z1); c111 = g(x1,y1,z1)

        c00 = dr.lerp(c000, c100, fx)
        c10 = dr.lerp(c010, c110, fx)
        c01 = dr.lerp(c001, c101, fx)
        c11 = dr.lerp(c011, c111, fx)

        c0 = dr.lerp(c00, c10, fy)
        c1 = dr.lerp(c01, c11, fy)

        return dr.lerp(c0, c1, fz)

    
    def load(self, data):
        # if data end with .sdf
        if data.endswith('.sdf'):
            data = np.loadtxt(data) # load data
            res = int(data[0]) # load res
            self.res = res
            grid = data[1:] # load grid
            grid = mi.TensorXf(grid, shape=(res,res,res,1))
            self.grid = mi.Texture3f(grid, use_accel=True)
        if data.endswith('.npy'):
            data = np.load(data) # load data
            res = int(data[0]) # load res
            self.res = res
            grid = data[1:] # load grid
            grid = mi.TensorXf(grid, shape=(res,res,res,1))
            self.grid = mi.Texture3f(grid, use_accel=True)
        elif data.endswith('.vol'):
            data = mi.FileResolver().resolve(data)
            data = (mi.TensorXf(mi.VolumeGrid(data)))
            self.grid = mi.Texture3f(data.shape[:3], 1, use_accel=True)
            if data.ndim == 3:
                data = data[..., None]
            self.grid.set_tensor(data, migrate=False)
            # self.texture.set_tensor(atleast_4d(data), migrate=False)
    
    def in_range(self, p:Point3f) -> Bool:
        """Return whether a point is in range."""
        return dr.all(p > 0) & dr.all(p < 1)

    def at(self, p:Point3f) -> Float:
        """Return the signed distance at a point."""
        return self.at_trilerp(p)

    def gradient(self, p:Point3f) -> Vector3f:
        """Return the gradient of the signed distance at a point."""
        method = utils.SDF_MODE
        if method == 'linear':
            epsilon = _opaque_float(1e-5) # finite difference
            two_epsilon = _opaque_float(2e-5)
            dx = Vector3f(epsilon, 0, 0)
            dy = Vector3f(0, epsilon, 0)
            dz = Vector3f(0, 0, epsilon)
            dfdx = (self.at(p + dx) - self.at(p - dx)) / two_epsilon
            dfdy = (self.at(p + dy) - self.at(p - dy)) / two_epsilon
            dfdz = (self.at(p + dz) - self.at(p - dz)) / two_epsilon
            return Vector3f(dfdx, dfdy, dfdz)
        elif method == 'cubic':
            return self.grid.eval_cubic_grad(p)[1][0]
        else:
            raise ValueError(f"This SDF method ({method}) is not supported.")
        
    def dir_deriv(self, p:Point3f, d:Vector3f) -> Float:
        """Return the directional derivative of the signed distance at a point."""
        grad = dr.normalize(self.gradient(p))
        return dr.dot(grad, d)

    # ------------------ GET GRAZING POINT ------------------
    def get_grz_secant(self, ray:Ray3f, t_min, t_max) -> Point3f:
        """Secant method for finding the grazing point."""
        d = ray.d
        L = dr.detach(ray(t_min))
        R = dr.detach(ray(t_max))
        
        with dr.suspend_grad():
            cnt = Int(0) # >6

            def cond(L, R, cnt):
                return cnt < utils.NUM_GRZ

            def body(L, R, cnt):
                dL = self.dir_deriv(L, d)
                dR = self.dir_deriv(R, d)
                M = R - dR * (R - L) / (dR - dL)
                dM = self.dir_deriv(M, d)
                L = dr.select(dM < 0, M, L)
                R = dr.select(dM > 0, M, R)
                return (L, R, cnt + 1)

            L, R, cnt = dr.while_loop(
                state=(L, R, cnt),
                cond=cond,
                body=body,
                label="get grazing point using secant method",
                max_iterations=utils.NUM_GRZ
            )
        
        return dr.detach(L)

    def get_grz_bisection(self, ray:Ray3f, t_min, t_max) -> Point3f:
        """Bisection method for finding the grazing point."""
        d = ray.d
        L = dr.detach(ray(t_min))
        R = dr.detach(ray(t_max))
        # M = dr.detach((L + R) / 2)
        
        with dr.suspend_grad():
            cnt = Int(0) # >12

            def cond(L, R, cnt):
                return cnt < utils.NUM_GRZ

            def body(L, R, cnt):
                M = (L + R) / 2
                dM = self.dir_deriv(M, d)
                L = dr.select(dM < 0, M, L)
                R = dr.select(dM > 0, M, R)
                return (L, R, cnt + 1)

            L, R, cnt = dr.while_loop(
                state=(L, R, cnt),
                cond=cond,
                body=body,
                label="get grazing point using bisection method",
                max_iterations=utils.NUM_GRZ
            )
        
        return dr.detach(L)

    def get_grz(self, ray:Ray3f, t_min, t_max) -> Point3f:
        method = utils.GRZ_MODE
        if method == 'secant':
            return self.get_grz_secant(ray, t_min, t_max)
        elif method == 'bisection':
            return self.get_grz_bisection(ray, t_min, t_max)
        else:
            raise ValueError(f"This grz method ({method}) is not supported.")
    
    # ------------------ GET RAY INTERSECTION ------------------
    def refine_intersect(self, ray:Ray3f, t_min, t_max) -> Point3f:
        """Bisection method for refining the intersection point."""
        if utils.NUM_REFINE == 0:
            return dr.detach((t_min + t_max) / 2)
        
        L = dr.detach(t_min)
        R = dr.detach(t_max)
        M = dr.detach((L + R) / 2)
        
        with dr.suspend_grad():
            cnt = Int(0) # >20

            def cond(L, R, M, cnt):
                return cnt < utils.NUM_REFINE

            def body(L, R, M, cnt):
                # update endpoints (kept for parity with original code; dL/dR not used)
                _ = self.at(ray(L))
                _ = self.at(ray(R))

                # bisection step
                M = (L + R) / 2
                dM = self.at(ray(M))
                L = dr.select(dM > 0, M, L)
                R = dr.select(dM < 0, M, R)
                return (L, R, M, cnt + 1)

            L, R, M, cnt = dr.while_loop(
                state=(L, R, M, cnt),
                cond=cond,
                body=body,
                label="refine intersection",
                max_iterations=utils.NUM_REFINE
            )
        
        return dr.detach(M)
    
    def get_intersect(self, ray:Ray3f, t:Float, is_valid:Mask) -> SurfaceInteraction3f:
        """ Return the surface interaction without supporting SDF autodiff.
            Code adapted from https://github.com/rgl-epfl/differentiable-sdf-rendering """
        si = dr.zeros(SurfaceInteraction3f)
        si.t = t
        si.p = ray(t)
        si.sh_frame.n = dr.normalize(self.gradient(si.p))
        si.initialize_sh_frame()
        si.n = si.sh_frame.n
        si.wi = dr.select(si.is_valid(), si.to_local(-ray.d), -ray.d)
        si.wavelengths = ray.wavelengths
        # si.uv = si.sh_frame.to_uv(si.p)
        si.dp_du = si.sh_frame.s
        si.dp_dv = si.sh_frame.t
        return si

    def ray_intersect_default(self, ray:Ray3f) -> Float:
        """ Default sphere tracing for ray intersection time. 
            Return the time of intersection. """        
        # copy ray
        ray = Ray3f(ray) 
        
        # parameters for sphere tracing
        t = Float(0)
        no_itx = Mask(True)
        in_range = Mask(True)
        
        # check bounding box
        itx_bbox, mint, maxt = self.bbox.ray_intersect(ray)
        inside_bbox = self.bbox.contains(ray.o)
        ray.maxt = dr.minimum(maxt, ray.maxt)
        ray_eps = _opaque_float(utils.RAY_EPS)
        itx_eps = _opaque_float(utils.ITX_EPS)
        itx_refine = _opaque_float(utils.ITX_REFINE)
        t = dr.select(inside_bbox, 0, mint + ray_eps)
        t = dr.detach(t)
        
        with dr.suspend_grad():
            max_iter = 1024
            cnt = dr.zeros(UInt, dr.width(t))

            def cond(t, no_itx, in_range, cnt):
                return (no_itx & in_range) & (cnt < max_iter)

            def body(t, no_itx, in_range, cnt):
                p = ray(t)
                d = self.at(p)
                t = t + 0.9 * d
                no_itx = (d > itx_eps)
                in_range = self.in_range(p) & (t < ray.maxt)
                return (t, no_itx, in_range, cnt + 1)

            t, no_itx, in_range, cnt = dr.while_loop(
                state=(t, no_itx, in_range, cnt),
                cond=cond,
                body=body,
                label="normal sphere tracing",
                max_iterations=max_iter
            )
        
        # check valid ray
        is_valid = (~no_itx) & in_range
        t[~is_valid] = dr.inf
        t[is_valid] = self.refine_intersect(ray, t - itx_refine, t + itx_refine)
        
        return t, is_valid

    def ray_intersect_grz(self, ray:Ray3f):
        """ Sphere tracing for ray intersection and grazing point.
            Return the time of intersection. """
        # copy ray
        ray = Ray3f(ray) 
        
        # parameters for sphere tracing
        t = Float(0)
        no_itx = Mask(True)
        in_range = Mask(True)
        
        # parameters for solving grazing point
        grz_t = Float(0)
        min_sdf = Float(1)
        pos_dir = Mask(True)
    
        # check bounding box
        itx_bbox, mint, maxt = self.bbox.ray_intersect(ray)
        inside_bbox = self.bbox.contains(ray.o)
        ray.maxt = dr.minimum(maxt, ray.maxt)
        ray_eps = utils.RAY_EPS
        itx_eps = utils.ITX_EPS
        sdf_eps = utils.SDF_EPS
        sdf_deriv_eps = utils.SDF_DERIV_EPS
        itx_refine = utils.ITX_REFINE
        t = dr.select(inside_bbox, 0, mint + ray_eps)
        t = dr.detach(t)

        with dr.suspend_grad():
            max_iter = 1024
            cnt = dr.zeros(UInt, dr.width(t))

            def cond(t, no_itx, in_range, pos_dir, grz_t, min_sdf, cnt):
                return (no_itx & in_range) & (cnt < max_iter)

            def body(t, no_itx, in_range, pos_dir, grz_t, min_sdf, cnt):
                p = Point3f(ray(t))
                d = self.at(p)
                dir = self.dir_deriv(p, ray.d)

                idx = (~pos_dir) & (dir > 0) & (d < min_sdf) & (d > itx_eps)
                grz_t = dr.select(idx, t, grz_t)
                min_sdf = dr.select(idx, d, min_sdf)

                # Add this step size scaling to improve stability (with neural SDF only)
                t = t + 0.9* d
                pos_dir = (dir > 0)
                no_itx = (d > itx_eps)
                in_range = self.in_range(p) & (t < ray.maxt)
                return (t, no_itx, in_range, pos_dir, grz_t, min_sdf, cnt + 1)

            t, no_itx, in_range, pos_dir, grz_t, min_sdf, cnt = dr.while_loop(
                state=(t, no_itx, in_range, pos_dir, grz_t, min_sdf, cnt),
                cond=cond,
                body=body,
                label="sphere tracing for grazing point",
                max_iterations=max_iter
            )
            
            # fine-tune grazing point
            grz = self.get_grz(ray, grz_t - min_sdf, grz_t + min_sdf) 
            # grz = ray(grz_t) # no fine-tune and do not work
            
            # check grazing point
            is_grz = (dr.abs(self.at(grz)) < sdf_eps) & \
                     (dr.abs(self.dir_deriv(grz, ray.d)) < sdf_deriv_eps) & \
                     (self.at(grz) > itx_eps)
        
        # check valid ray
        is_valid = (~no_itx) & in_range
        t[~is_valid] = dr.inf
        t[is_valid] = self.refine_intersect(ray, t - itx_refine, t + itx_refine)
        return t, is_valid, is_grz, grz
    
    def ray_intersect(self, ray:Ray3f, method='default') -> mi.SurfaceInteraction3f:
        """ Return the intersection with a ray. """
        if method == 'default': # interior
            t, is_valid = self.ray_intersect_default(ray)
            return self.get_intersect(ray, t, is_valid)
        elif method == 'grz': # interior + boundary
            t, is_valid, is_bp, bp = self.ray_intersect_grz(ray)
            return self.get_intersect(ray, t, is_valid), is_bp, bp
        else:
            raise ValueError(f"This ray intersect method ({method}) is not supported.")
    
    def traverse(self, callback):
        # Mitsuba 3.7+ traversal callback uses `put()` (nanobind) instead of `put_parameter()`.
        callback.put("grid", self.grid.tensor(), mi.ParamFlags.Differentiable)
        
    
    def parameters_changed(self, keys):
        # Mitsuba 3.7+: if the parameter was registered via `Texture::tensor()`,
        # updates should be applied via `update_inplace()` (avoid set_tensor(self.tensor())).
        if hasattr(self, "grid") and self.grid is not None:
            self.grid.update_inplace()


class SphereSDF(SDF):
    """ Sphere Signed Distance Function. 
        Used for initial SDF for optimization. """
    
    def __init__(self, res=256, center:Point3f=Point3f(0.5,0.5,0.5), radius:Float=0.2):
        super().__init__()
        
        def dist_to_sphere(p:Point3f, center:Point3f, radius:Float) -> Float:
            """Signed distance function for a sphere."""
            # return - dr.norm(p - center) + radius
            return dr.norm(p - center) - radius
        
        self.res = res
        grid = dist_to_sphere(dr.meshgrid(
            dr.linspace(Float, 0, 1, res),
            dr.linspace(Float, 0, 1, res),
            dr.linspace(Float, 0, 1, res)
        ), center=center, radius=radius)
        grid = mi.TensorXf(grid, shape=(res,res,res,1))
        self.grid = mi.Texture3f(grid, use_accel=True)
class GradientSDF(SDF):
    def __init__(self, sdf: SDF):
        super().__init__()
        self.sdf = sdf
        self.p = mi.Vector3f(0.0, 0.0, 0.0)  # translation in world coords

    def at(self, x: mi.Point3f) -> mi.Float:
        return self.sdf.at(x - self.p)

    def gradient(self, x: mi.Point3f) -> mi.Vector3f:
        return self.sdf.gradient(x - self.p)

    def ray_intersect(self, ray: mi.Ray3f, method='default'):

        ray.o = ray.o - self.p

        if method == 'default':
            itx = self.sdf.ray_intersect(ray, method='default')
            # Hit point is in object frame; map back to world by adding the
            # translation (matches the 'grz' branch and at()/gradient()'s x - p).
            itx.p = itx.p + self.p
            return itx

        elif method == 'grz':
            itx, is_grz, grz = self.sdf.ray_intersect(ray, method='grz')

            itx.p = itx.p + self.p
            grz = grz + self.p
            return itx, is_grz, grz

        else:
            raise ValueError(method)

    def traverse(self, callback):
        callback.put("p", self.p, mi.ParamFlags.Differentiable)
        self.sdf.traverse(callback)

    def parameters_changed(self, keys):
        self.sdf.parameters_changed(keys)
