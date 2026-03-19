from PIL import Image
import cv2
import numpy as np


class ImageUtils:
    """Image preprocessing helpers for WD14 ONNX models."""

    @staticmethod
    def fill_transparent(image: Image.Image, color: str = "WHITE") -> Image.Image:
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, color)
        new_image.paste(image, mask=image)
        return new_image.convert("RGB")

    @staticmethod
    def make_square(img: np.ndarray, target_size: int) -> np.ndarray:
        old_size = img.shape[:2]
        desired_size = max(max(old_size), target_size)
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 255, 255]
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )

    @staticmethod
    def smart_resize(img: np.ndarray, size: int) -> np.ndarray:
        if img.shape[0] > size:
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        if img.shape[0] < size:
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def preprocess_image(image: Image.Image, target_size: int) -> np.ndarray:
        image = ImageUtils.fill_transparent(image)
        arr = np.array(image)[:, :, ::-1]
        arr = ImageUtils.make_square(arr, target_size)
        arr = ImageUtils.smart_resize(arr, target_size)
        return np.expand_dims(arr.astype(np.float32), 0)
