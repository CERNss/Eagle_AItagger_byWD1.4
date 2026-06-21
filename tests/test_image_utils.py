from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from service.image_utils import ImageUtils, configure_image_limits


def test_resize_to_square_pads_small_without_upscaling():
    arr = np.full((10, 20, 3), 128, dtype=np.uint8)
    out = ImageUtils.resize_to_square(arr, 64)

    assert out.shape == (64, 64, 3)
    # 10x20 image centered: top=(64-10)//2=27, left=(64-20)//2=22, pixels untouched
    assert (out[27:37, 22:42] == 128).all()
    # corners are white padding
    assert (out[0, 0] == 255).all()
    assert (out[-1, -1] == 255).all()


def test_resize_to_square_downscales_oversized_to_target():
    # A 2000x50 image would have ballooned to a 2000x2000 square under the old
    # pad-then-resize path; the bounded path must still land at target size.
    arr = np.full((2000, 50, 3), 200, dtype=np.uint8)
    out = ImageUtils.resize_to_square(arr, 64)
    assert out.shape == (64, 64, 3)


def test_resize_to_square_handles_square_at_target():
    arr = np.full((64, 64, 3), 50, dtype=np.uint8)
    out = ImageUtils.resize_to_square(arr, 64)
    assert out.shape == (64, 64, 3)
    assert (out == 50).all()


def test_preprocess_image_output_contract():
    img = Image.new("RGB", (100, 40), (10, 20, 30))
    out = ImageUtils.preprocess_image(img, 48)
    assert out.shape == (1, 48, 48, 3)
    assert out.dtype == np.float32


def test_preprocess_handles_rgba_transparency():
    img = Image.new("RGBA", (32, 32), (255, 0, 0, 0))  # fully transparent
    out = ImageUtils.preprocess_image(img, 32)
    assert out.shape == (1, 32, 32, 3)


def test_fill_transparent_opaque_returns_pixels_unchanged():
    # Opaque image takes the fast path; compositing onto white is a no-op, so the
    # pixels must be bit-for-bit identical to the input.
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    out = ImageUtils.fill_transparent(img)
    assert out.mode == "RGB"
    assert (np.array(out) == [10, 20, 30]).all()


def test_fill_transparent_composites_alpha_onto_white():
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 0))  # fully transparent red
    out = ImageUtils.fill_transparent(img)
    assert out.mode == "RGB"
    assert (np.array(out) == 255).all()  # transparent pixels become white


def test_preprocess_rejects_decompression_bomb(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (50, 50), (0, 0, 0)).save(path)
    original = Image.MAX_IMAGE_PIXELS
    configure_image_limits(16)  # 16px cap; 50x50=2500 >> 2*16 trips the guard
    try:
        with pytest.raises(Image.DecompressionBombError):
            with Image.open(path) as img:
                ImageUtils.preprocess_image(img, 8)
    finally:
        Image.MAX_IMAGE_PIXELS = original


def test_configure_image_limits_toggle():
    original = Image.MAX_IMAGE_PIXELS
    try:
        configure_image_limits(0)
        assert Image.MAX_IMAGE_PIXELS is None
        configure_image_limits(123)
        assert Image.MAX_IMAGE_PIXELS == 123
        configure_image_limits(-5)
        assert Image.MAX_IMAGE_PIXELS is None
    finally:
        Image.MAX_IMAGE_PIXELS = original
