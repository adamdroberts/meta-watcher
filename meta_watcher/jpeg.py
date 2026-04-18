from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    pil = Image.fromarray(image)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
