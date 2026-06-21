from __future__ import annotations

from PIL import Image
import cv2
import numpy as np

_WHITE = [255, 255, 255]


def configure_image_limits(max_pixels: int) -> None:
    """Set Pillow's decompression-bomb guard.

    ``max_pixels <= 0`` disables the cap (every image is processed). A positive
    value makes Pillow refuse to decode images larger than the cap. This only
    affects in-memory decoding; the source file on disk is never modified.
    """
    Image.MAX_IMAGE_PIXELS = max_pixels if max_pixels and max_pixels > 0 else None


class ImageUtils:
    """Image preprocessing helpers for WD14 ONNX models.

    The original file is only ever read. All transforms here operate on an
    in-memory copy that is fed to the model and then discarded.
    """

    @staticmethod
    def fill_transparent(image: Image.Image, color: str = "WHITE") -> Image.Image:
        has_alpha = image.mode in ("RGBA", "LA", "PA") or (
            image.mode == "P" and "transparency" in image.info
        )
        if not has_alpha:
            # Opaque image: there is nothing to composite, so skip the RGBA
            # round-trip (extra canvas + paste) and decode straight to RGB. This
            # halves peak buffers for the common JPEG/RGB case and is bit-for-bit
            # identical to compositing a fully-opaque image onto white.
            return image.convert("RGB")
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, color)
        new_image.paste(image, mask=image)
        return new_image.convert("RGB")

    @staticmethod
    def resize_to_square(img: np.ndarray, target_size: int) -> np.ndarray:
        """Scale (down only) so the longest side fits, then white-pad to a square.

        Equivalent in output to "pad to the longest side, then resize to
        ``target_size``", but it never materialises an oversized square canvas,
        so a single very large image cannot exhaust memory. Images already at or
        below ``target_size`` keep their pixels and are only padded (never
        upscaled), preserving the previous behaviour.
        """
        height, width = img.shape[:2]
        longest = max(height, width)
        if longest > target_size:
            scale = target_size / longest
            new_w = max(1, round(width * scale))
            new_h = max(1, round(height * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            height, width = img.shape[:2]

        delta_w = max(0, target_size - width)
        delta_h = max(0, target_size - height)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=_WHITE,
        )

    @staticmethod
    def preprocess_image(image: Image.Image, target_size: int) -> np.ndarray:
        # For large JPEGs, let libjpeg decode at a reduced scale (>= target_size)
        # to cut decode memory. No-op for formats without draft support, and the
        # source file is untouched either way.
        try:
            image.draft("RGB", (target_size, target_size))
        except (ValueError, OSError):  # pragma: no cover - format dependent
            pass
        image = ImageUtils.fill_transparent(image)
        arr = np.array(image)[:, :, ::-1]
        arr = ImageUtils.resize_to_square(arr, target_size)
        return np.expand_dims(arr.astype(np.float32), 0)
