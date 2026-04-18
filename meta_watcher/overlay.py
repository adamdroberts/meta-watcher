from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from .core import Detection, InventoryItem


def render_overlay(
    frame: np.ndarray,
    *,
    detections: list[Detection],
    inventory: list[InventoryItem],
    mode: str,
    inventory_active: bool,
    recording_active: bool,
    status_text: str,
    hud_timings: dict[str, float] | None = None,
    hud_fps: float = 0.0,
) -> np.ndarray:
    canvas = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")

    for detection in detections:
        color = _color_for_label(detection.track_id or detection.label)
        _draw_mask(draw, canvas, detection, color)
        draw.rectangle(detection.bbox, outline=color + (255,), width=3)
        label = detection.label
        if detection.track_id:
            label = f"{label} {detection.track_id}"
        caption = f"{label} {detection.confidence:.2f}"
        x1, y1, _, _ = detection.bbox
        text_box = (x1, max(0, y1 - 22), x1 + max(110, len(caption) * 7), max(20, y1))
        draw.rectangle(text_box, fill=color + (200,))
        draw.text((text_box[0] + 4, text_box[1] + 2), caption, fill=(255, 255, 255, 255))

    status_height = 88 if not hud_timings else 128
    draw.rectangle((10, 10, 310, status_height), fill=(0, 0, 0, 165))
    draw.text((20, 18), f"Mode: {mode}", fill=(255, 255, 255, 255))
    draw.text((20, 38), f"Inventory scan: {'on' if inventory_active else 'frozen'}", fill=(255, 255, 255, 255))
    draw.text((20, 58), f"Recording: {'active' if recording_active else 'idle'}", fill=(255, 255, 255, 255))
    draw.text((20, 78), status_text[:42], fill=(255, 255, 255, 255))
    if hud_timings:
        infer = hud_timings.get("inference", 0.0)
        ovl = hud_timings.get("overlay", 0.0)
        rec = hud_timings.get("recorder", 0.0)
        draw.text(
            (20, 98),
            f"{hud_fps:>4.1f} fps  i:{infer:>4.0f} o:{ovl:>3.0f} r:{rec:>3.0f} ms",
            fill=(200, 235, 255, 255),
        )

    inv_top = status_height + 12
    draw.rectangle((10, inv_top, 320, inv_top + 32 + (18 * max(1, len(inventory)))), fill=(0, 0, 0, 145))
    draw.text((20, inv_top + 8), "Frozen scene inventory", fill=(255, 255, 255, 255))
    for index, item in enumerate(inventory[:12], start=1):
        draw.text(
            (20, inv_top + 8 + (18 * index)),
            f"{index:02d}. {item.label} ({item.confidence:.2f})",
            fill=(255, 255, 255, 255),
        )

    return np.asarray(canvas.convert("RGB"))


def _draw_mask(draw: ImageDraw.ImageDraw, canvas: Image.Image, detection: Detection, color: tuple[int, int, int]) -> None:
    if detection.mask is None:
        return
    mask = np.asarray(detection.mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.shape != (canvas.height, canvas.width):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((canvas.width, canvas.height)))
    alpha = Image.fromarray((mask > 0).astype(np.uint8) * 70)
    overlay = Image.new("RGBA", canvas.size, color + (0,))
    overlay.putalpha(alpha)
    canvas.alpha_composite(overlay)


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8"), usedforsecurity=False).hexdigest()
    hue = int(digest[:2], 16)
    palette = [
        ImageColor.getrgb("#ff6b6b"),
        ImageColor.getrgb("#ffd166"),
        ImageColor.getrgb("#06d6a0"),
        ImageColor.getrgb("#118ab2"),
        ImageColor.getrgb("#ef476f"),
        ImageColor.getrgb("#f78c6b"),
    ]
    return palette[hue % len(palette)]
