from __future__ import annotations

import numpy as np
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
