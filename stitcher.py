#!/usr/bin/env python3
"""
make_cross_cubemap.py

Build a horizontal-cross OpenGL cubemap from six images and ALSO create an
equirectangular (lat-long) panorama from the generated cross image.

Default (no args) writes both:
  - cubemap_cross.<ext>
  - equirect.<ext>
"""
import os, sys, argparse, math
import numpy as np

# Try imageio for HDR/EXR; Pillow for the rest
try:
    import imageio.v3 as iio
except Exception:
    iio = None
try:
    from PIL import Image
except Exception:
    Image = None

SUPPORTED_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".tga", ".hdr", ".exr"]
FACE_BASENAMES = ["right", "left", "front", "back", "top", "bottom"]
CROSS_POS = {
    "top":    (1, 0),
    "left":   (0, 1),
    "front":  (1, 1),
    "right":  (2, 1),
    "back":   (3, 1),
    "bottom": (1, 2),
}

def find_common_extension(folder: str) -> str:
    found = None
    for ext in SUPPORTED_EXTS:
        if all(os.path.exists(os.path.join(folder, f + ext)) for f in FACE_BASENAMES):
            found = ext; break
    if not found:
        names = {n.lower(): n for n in os.listdir(folder)}
        for ext in SUPPORTED_EXTS:
            if all((f + ext) in names for f in FACE_BASENAMES):
                found = ext; break
    if not found:
        missing = []
        for f in FACE_BASENAMES:
            present = [e for e in SUPPORTED_EXTS if os.path.exists(os.path.join(folder, f + e))]
            if not present: missing.append(f)
        msg = "Could not find all six faces with a single shared extension.\n"
        if missing: msg += "Missing (any ext): " + ", ".join(missing) + "\n"
        msg += "Expected: right.png left.png front.png back.png top.png bottom.png"
        raise FileNotFoundError(msg)
    return found

def read_image(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".hdr", ".exr"]:
        if iio is None:
            raise RuntimeError(f"Reading {ext} requires `imageio`.")
        arr = iio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr
    else:
        if Image is None:
            raise RuntimeError("Reading non-HDR formats requires Pillow.")
        with Image.open(path) as im:
            return np_from_pillow(im)

def np_from_pillow(im):
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    return np.array(im)

def write_image(path: str, arr):
    ext = os.path.splitext(path)[1].lower()

        # If writing Radiance HDR, drop alpha if present (HDR is RGB)
    if ext == ".hdr" and arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if ext in [".hdr", ".exr"]:
        if iio is None:
            raise RuntimeError("Writing HDR/EXR requires `imageio` (pip install imageio).")
        iio.imwrite(path, arr.astype(np.float32))
        return

    # HDR/EXR via imageio (write floats as-is)
    if ext in [".hdr", ".exr"]:
        if iio is None:
            raise RuntimeError("Writing HDR/EXR requires `imageio` (pip install imageio).")
        iio.imwrite(path, arr.astype(np.float32))
        return

    # LDR via Pillow
    if Image is None:
        raise RuntimeError("Writing PNG/JPG/TIFF/TGA requires Pillow. `pip install Pillow`.")

    a = arr
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)

    if a.dtype.kind == "f":
        # If floats look like 0..255, bring them to uint8 directly.
        maxv = float(np.nanmax(a)) if a.size else 0.0
        minv = float(np.nanmin(a)) if a.size else 0.0
        if maxv > 1.5 or minv < -0.5:
            a = np.clip(a, 0.0, 255.0).astype("uint8")
        else:
            a = np.clip(a, 0.0, 1.0)
            a = (a * 255.0 + 0.5).astype("uint8")
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype("uint8")

    Image.fromarray(a).save(path)


# ---------- equirect from cross ----------

def _face_from_dir(x, y, z):
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    is_x = (ax >= ay) & (ax >= az)
    is_y = (ay >  ax) & (ay >= az)
    is_z = ~(is_x | is_y)

    face = np.empty_like(x, dtype=np.int32)
    denom = np.empty_like(x, dtype=x.dtype)

    m = is_x & (x > 0); face[m] = 0; denom[m] = ax[m]   # +X right
    m = is_x & (x <= 0); face[m] = 1; denom[m] = ax[m]  # -X left
    m = is_z & (z > 0); face[m] = 2; denom[m] = az[m]   # +Z front
    m = is_z & (z <= 0); face[m] = 3; denom[m] = az[m]  # -Z back
    m = is_y & (y > 0); face[m] = 4; denom[m] = ay[m]   # +Y top
    m = is_y & (y <= 0); face[m] = 5; denom[m] = ay[m]  # -Y bottom
    return face, denom

def _uv_on_face(face, x, y, z, denom):
    """Compute (u,v) in [-1,1] on each cube face for directions (x,y,z)."""
    u = np.empty_like(x)
    v = np.empty_like(x)

    # +X (right): u = -z/|x|, v =  y/|x|
    m = (face == 0)
    u[m] = -z[m] / denom[m]
    v[m] =  y[m] / denom[m]

    # -X (left):  u =  z/|x|, v =  y/|x|
    m = (face == 1)
    u[m] =  z[m] / denom[m]
    v[m] =  y[m] / denom[m]

    # +Z (front): u =  x/|z|, v =  y/|z|
    m = (face == 2)
    u[m] =  x[m] / denom[m]
    v[m] =  y[m] / denom[m]

    # -Z (back):  u = -x/|z|, v =  y/|z|
    m = (face == 3)
    u[m] = -x[m] / denom[m]
    v[m] =  y[m] / denom[m]

    # +Y (top):   u =  x/|y|, v = -z/|y|
    m = (face == 4)
    u[m] =  x[m] / denom[m]
    v[m] = -z[m] / denom[m]

    # -Y (bottom): u =  x/|y|, v =  z/|y|
    m = (face == 5)
    u[m] =  x[m] / denom[m]
    v[m] =  z[m] / denom[m]

    return u, v

def _tile_offsets_for_faces(face, F):
    idx_to_name = {0:"right",1:"left",2:"front",3:"back",4:"top",5:"bottom"}
    cols = np.zeros_like(face, dtype=np.int32)
    rows = np.zeros_like(face, dtype=np.int32)
    for idx, name in idx_to_name.items():
        m = (face == idx)
        cx, cy = CROSS_POS[name]
        cols[m], rows[m] = cx, cy
    return cols * F, rows * F

def _bilinear_sample(img, xf, yf):
    H, W = img.shape[:2]
    x0 = np.floor(xf).astype(np.int32); y0 = np.floor(yf).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1);     y1 = np.clip(y0 + 1, 0, H - 1)
    wx = xf - x0; wy = yf - y0
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    wa = (1 - wx) * (1 - wy); wb = wx * (1 - wy); wc = (1 - wx) * wy; wd = wx * wy
    if img.ndim == 3:
        wa = wa[..., None]; wb = wb[..., None]; wc = wc[..., None]; wd = wd[..., None]
    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def equirect_from_cross(cross_img, eq_w=None, eq_h=None):
    """
    Generate an equirectangular panorama from a horizontal-cross cubemap image.
    The cross must be 4*F by 3*F where F is face size.
    If eq_w/eq_h are None, uses eq_w=4F, eq_h=2F.
    """
    H, W = cross_img.shape[:2]
    F = min(W // 4, H // 3)
    if (W != 4 * F) or (H != 3 * F):
        raise ValueError(f"Cross image has unexpected size {W}x{H}. Expected 4F x 3F.")

    if eq_w is None: eq_w = 4 * F
    if eq_h is None: eq_h = 2 * F

    # --- Normalize source for sampling ---
    # If the cross is uint8, convert to float in 0..1 for correct filtering.
    if np.issubdtype(cross_img.dtype, np.integer):
        cross_f = cross_img.astype(np.float32) / 255.0
    else:
        cross_f = cross_img.astype(np.float32)

    # Build lon/lat grid (pixel centers)
    jj = (np.arange(eq_w, dtype=np.float64) + 0.5) / eq_w
    ii = (np.arange(eq_h, dtype=np.float64) + 0.5) / eq_h
    uu, vv = np.meshgrid(jj, ii)

    lon = (uu * 2.0 * math.pi) - math.pi       # [-pi, pi], 0 faces +Z (front)
    lat = (0.5 - vv) * math.pi                 # [-pi/2, pi/2], + up

    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)

    face_idx, denom = _face_from_dir(x, y, z)
    u, v = _uv_on_face(face_idx, x, y, z, denom)   # [-1,1], v up in face space

    fx = (u + 1.0) * 0.5 * (F - 1)
    fy = (1.0 - (v + 1.0) * 0.5) * (F - 1)         # flip to image-down

    offx, offy = _tile_offsets_for_faces(face_idx, F)
    sx = offx.astype(np.float64) + fx
    sy = offy.astype(np.float64) + fy

    # Bilinear sample from normalized cross
    out = _bilinear_sample(cross_f, sx, sy)
    return out


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description="Build cross + equirect (default outputs both).")
    p.add_argument("--folder","-f", default=".")
    p.add_argument("--out","-o", default="cubemap_cross")
    p.add_argument("--ext", default=None)
    p.add_argument("--no-equirect", action="store_true")
    p.add_argument("--eq-out", default="equirect")
    p.add_argument("--eq-ext", default=None)
    p.add_argument("--eq-width", type=int, default=None)
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}", file=sys.stderr); sys.exit(1)

    common_ext = find_common_extension(folder)
    out_ext = args.ext if args.ext else common_ext

    # Load faces
    imgs = {name: read_image(os.path.join(folder, name + common_ext)) for name in FACE_BASENAMES}

    # Validate
    sizes = {(im.shape[0], im.shape[1]) for im in imgs.values()}
    if len(sizes) != 1: raise ValueError(f"All faces must be same size. Got: {sizes}")
    h, w = next(iter(sizes))
    if h != w: raise ValueError(f"Faces must be square. Got {w}x{h}.")
    front = imgs["front"]; ch = front.shape[2] if front.ndim == 3 else 1; dtype = front.dtype

    # Build cross
    canvas_h, canvas_w = 3*h, 4*w
    cross = np.zeros((canvas_h, canvas_w, ch), dtype=dtype)
    for key in ["top","left","front","right","back","bottom"]:
        cx, cy = CROSS_POS[key]; x0, y0 = cx*w, cy*h
        cross[y0:y0+h, x0:x0+w, ...] = imgs[key]

    out_path = os.path.join(folder, args.out + out_ext)
    write_image(out_path, cross)
    print(f"Saved cross cubemap: {out_path}")
    print(f"Face size: {w}x{h} | Cross size: {canvas_w}x{canvas_h}")

    # Equirect (default on)
    if not args.no_equirect:
        target_w = args.eq_width if args.eq_width is not None else 4*w
        target_h = max(1, target_w // 2)
        eq = equirect_from_cross(cross, eq_w=target_w, eq_h=target_h)
        eq_ext = args.eq_ext if args.eq_ext else out_ext
        eq_path = os.path.join(folder, args.eq_out + eq_ext)
        write_image(eq_path, eq)
        print(f"Saved equirectangular pano: {eq_path}")
        print(f"Equirect size: {target_w}x{target_h}")

if __name__ == "__main__":
    main()
