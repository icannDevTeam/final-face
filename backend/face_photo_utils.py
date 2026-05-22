"""
face_photo_utils.py — shared helpers for Hikvision face enrollment scripts.

Key fact (V4.38 firmware): the device's faceURL pull aborts mid-transfer for
JPEGs larger than ~250 KB, returning `urlDownloadFail`. Always downscale
before serving.
"""
from pathlib import Path
from PIL import Image

# Hikvision V4.38 reliably accepts files this size or smaller via faceURL pull.
# 200 KB picked with margin under the observed ~250 KB break point.
MAX_FACE_PHOTO_BYTES = 200 * 1024
# Long-edge ceiling. 640 px is more than enough for the 112x112 deep-feature
# extractor inside the terminal.
MAX_FACE_LONG_EDGE = 640


def ensure_device_safe(path: str | Path) -> Path:
    """Resize `path` IN PLACE if it exceeds device-safe limits.

    Returns the same Path. Idempotent: small files are untouched.
    """
    p = Path(path)
    if not p.exists():
        return p
    if p.stat().st_size <= MAX_FACE_PHOTO_BYTES:
        # Even if small, ensure dimensions aren't crazy
        try:
            with Image.open(p) as im:
                if max(im.size) <= MAX_FACE_LONG_EDGE:
                    return p
        except Exception:
            return p
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_FACE_LONG_EDGE, MAX_FACE_LONG_EDGE))
            # Step down quality until under size cap
            for q in (92, 85, 78, 70, 60):
                im.save(p, "JPEG", quality=q, optimize=True)
                if p.stat().st_size <= MAX_FACE_PHOTO_BYTES:
                    return p
    except Exception as e:
        print(f"  ⚠ ensure_device_safe failed for {p.name}: {e}")
    return p
