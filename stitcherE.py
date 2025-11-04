#!/usr/bin/env python3
import os, re, sys, argparse
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
_num_re = re.compile(r"(\d+)")

def _read(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".hdr", ".exr"):
        if iio is None:
            raise RuntimeError("Reading HDR/EXR requires imageio (pip install imageio).")
        arr = iio.imread(path)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr
    else:
        if Image is None:
            raise RuntimeError("Reading PNG/JPG/TIFF/TGA requires Pillow (pip install Pillow).")
        with Image.open(path) as im:
            # Preserve alpha if present
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            return np.array(im)

def _write(path: str, arr):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hdr" and arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]  # Radiance HDR is RGB-only
    if ext in (".hdr", ".exr"):
        if iio is None:
            raise RuntimeError("Writing HDR/EXR requires imageio (pip install imageio).")
        iio.imwrite(path, arr.astype(np.float32))
        return
    if Image is None:
        raise RuntimeError("Writing PNG/JPG/TIFF/TGA requires Pillow.")
    a = arr
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.dtype.kind == 'f':
        mx = float(np.nanmax(a)) if a.size else 0.0
        mn = float(np.nanmin(a)) if a.size else 0.0
        if mx > 1.5 or mn < -0.5:
            a = np.clip(a, 0.0, 255.0).astype('uint8')
        else:
            a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype('uint8')
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype('uint8')
    Image.fromarray(a).save(path)

def _find_strips(folder: str, pattern_prefix="strip_"):
    """Collect strips in numeric order; prefer names like strip_###.ext,
    else fall back to any supported ext and sort by first integer in name."""
    entries = []
    for name in os.listdir(folder):
        low = name.lower()
        root, ext = os.path.splitext(low)
        if ext in SUPPORTED_EXTS and root.startswith(pattern_prefix):
            entries.append(name)
    if not entries:
        entries = [n for n in os.listdir(folder)
                   if os.path.splitext(n.lower())[1] in SUPPORTED_EXTS]
    if not entries:
        raise FileNotFoundError("No strips found.")

    def key(nm):
        m = _num_re.search(nm)
        return int(m.group(1)) if m else 0

    entries.sort(key=key, reverse=True)
    return [os.path.join(folder, n) for n in entries]

def _unify_dtype_and_channels(imgs):
    # Match channel count across images; prefer RGBA if any has alpha
    max_c = max((im.shape[2] if im.ndim == 3 else 1) for im in imgs)
    want_c = 4 if max_c == 4 else 3
    out = []
    for im in imgs:
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        c = im.shape[2]
        if c == want_c:
            out.append(im)
        elif c == 3 and want_c == 4:
            a = np.full(im.shape[:2] + (1,), 255 if im.dtype == np.uint8 else 1.0, dtype=im.dtype)
            out.append(np.concatenate([im, a], axis=-1))
        elif c == 4 and want_c == 3:
            out.append(im[..., :3])
        else:
            out.append(im)
    return out

def main():
    ap = argparse.ArgumentParser(description="Stitch vertical equirectangular strips captured in order.")
    ap.add_argument("--folder", "-f", default=".", help="Folder containing strip_### files")
    ap.add_argument("--prefix", default="strip_", help="Filename prefix (default strip_)")
    ap.add_argument("--out", "-o", default="equirect", help="Output base name (no extension)")
    ap.add_argument("--ext", default=None, help="Override output extension, e.g. .png/.exr/.hdr")
    ap.add_argument("--expected", type=int, default=None, help="Optional: expected number of strips")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    paths = _find_strips(folder, pattern_prefix=args.prefix)

    if args.expected is not None and len(paths) != args.expected:
        print(f"[WARN] Found {len(paths)} strips, expected {args.expected}.")

    # Read and standardize
    imgs = [_read(p) for p in paths]
    imgs = _unify_dtype_and_channels(imgs)

    # Equalize height to the smallest (guards against 1px mismatches)
    min_h = min(im.shape[0] for im in imgs)
    imgs = [im[:min_h, :im.shape[1], ...] for im in imgs]

    # Concatenate left→right in numeric order
    ch = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
    total_w = sum(im.shape[1] for im in imgs)
    dtype = imgs[0].dtype
    pano = np.zeros((min_h, total_w, ch), dtype=dtype)
    x = 0
    for im in imgs:
        w = im.shape[1]
        pano[:, x:x+w, ...] = im
        x += w

    # Choose extension (default to first strip's)
    in_ext = os.path.splitext(paths[0])[1]
    out_ext = args.ext if args.ext else in_ext
    if not out_ext.startswith('.'):
        out_ext = '.' + out_ext

    out_path = os.path.join(folder, args.out + out_ext)
    _write(out_path, pano)

    print(f"Saved equirectangular panorama: {out_path}")
    print(f"Final size: {total_w}x{min_h}  |  strips: {len(imgs)}")
    print(f"Order used: {paths}")

if __name__ == "__main__":
    main()
