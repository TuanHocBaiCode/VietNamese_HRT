"""
Augmentation chuyên biệt cho HTR tiếng Việt.
Không dùng flip/crop mạnh vì có thể làm biến dạng vị trí dấu tiếng Việt.
Input:
    - torch.Tensor [1, H, W]
    - numpy.ndarray [H, W]
Giá trị:
    - [0, 1]
    - hoặc [0, 255]
Output:
    - torch.Tensor [1, H, W] hoặc numpy.ndarray tương ứng.
"""

from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

class HTRAugmenter:
    def __init__(
        self,
        prob: float = 0.4,
        elastic_alpha: float = 3.0,
        elastic_sigma: float = 4.0,
        rotate_range: Tuple[float, float] = (-5.0, 5.0),
        shear_range: Tuple[float, float] = (-8.0, 8.0),
        morphology_kernel_size: int = 2,
        blur_kernel_size: int = 3,
        salt_pepper_amount: float = 0.002,
        salt_pepper_salt_ratio: float = 0.5,
    ):
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob phải nằm trong [0, 1].")
        if morphology_kernel_size < 1:
            raise ValueError("morphology_kernel_size phải >= 1.")
        if blur_kernel_size < 1 or blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size phải là số lẻ dương.")
        if not 0.0 <= salt_pepper_amount <= 1.0:
            raise ValueError(
                "salt_pepper_amount phải nằm trong [0, 1]."
            )
        self.prob = prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        self.rotate_range = rotate_range
        self.shear_range = shear_range

        self.morphology_kernel_size = morphology_kernel_size
        self.blur_kernel_size = blur_kernel_size

        self.salt_pepper_amount = salt_pepper_amount
        self.salt_pepper_salt_ratio = salt_pepper_salt_ratio

    # =========================================================
    # Convert input
    # =========================================================
    @staticmethod
    def _to_numpy(
        image: ArrayLike,
    ) -> Tuple[np.ndarray, bool]:
        was_torch = isinstance(image, torch.Tensor)
        if was_torch:
            x = image.detach().cpu().float().numpy()
            if x.ndim == 3:
                if x.shape[0] != 1:
                    raise ValueError(
                        f"Kỳ vọng tensor [1,H,W], nhận {x.shape}"
                    )
                x = x[0]
            elif x.ndim != 2:
                raise ValueError(
                    f"Kỳ vọng [1,H,W] hoặc [H,W], nhận {x.shape}"
                )
        else:
            x = np.asarray(image, dtype=np.float32)
            if x.ndim == 3:
                if x.shape[0] != 1:
                    raise ValueError(
                        f"Kỳ vọng ndarray [1,H,W], nhận {x.shape}"
                    )
                x = x[0]
            elif x.ndim != 2:
                raise ValueError(
                    f"Kỳ vọng [1,H,W] hoặc [H,W], nhận {x.shape}"
                )
        if x.size == 0:
            raise ValueError("Ảnh rỗng.")
        if float(x.max()) > 1.0:
            x = x / 255.0
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        return x, was_torch
    @staticmethod
    def _to_output(
        image: np.ndarray,
        was_torch: bool
    ) -> ArrayLike:
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        if was_torch:
            return torch.from_numpy(
                image
            ).unsqueeze(0)
        return image

    # =========================================================
    # 1. Elastic Distortion
    # =========================================================
    def elastic_distortion(
        self,
        image: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        h, w = image.shape
        dx = rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        dy = rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (0, 0), self.elastic_sigma )
        dy = cv2.GaussianBlur(dy, (0, 0), self.elastic_sigma)
        dx *= self.elastic_alpha
        dy *= self.elastic_alpha
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = grid_x + dx
        map_y = grid_y + dy
        result = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1.0,
        )
        return result

    # =========================================================
    # 2. Morphological Operations
    # =========================================================
    def morphology(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        kernel = np.ones(
            (
                self.morphology_kernel_size,
                self.morphology_kernel_size
            ),
            dtype=np.uint8
        )
        # Ảnh chữ màu đen, nền trắng:
        # erode ảnh -> vùng đen dày hơn
        # dilate ảnh -> vùng đen mảnh hơn
        if operation == "dilation":
            return cv2.erode(
                image,
                kernel,
                iterations=1
            )
        if operation == "erosion":
            return cv2.dilate(
                image,
                kernel,
                iterations=1
            )
        raise ValueError(
            "operation phải là 'dilation' hoặc 'erosion'"
        )

    # =========================================================
    # 3. Affine
    # =========================================================
    def affine(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        h, w = image.shape
        angle = random.uniform(
            *self.rotate_range
        )
        shear_deg = random.uniform(
            *self.shear_range
        )
        center = (
            w / 2.0,
            h / 2.0
        )
        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )
        shear = np.tan(
            np.deg2rad(shear_deg)
        )
        matrix[0, 1] += shear
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1.0,
        )

    # =========================================================
    # 4. Gaussian Blur
    # =========================================================
    def gaussian_blur(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        return cv2.GaussianBlur(image, (self.blur_kernel_size, self.blur_kernel_size), 0
        )

    # =========================================================
    # 5. Motion Blur
    # =========================================================
    def motion_blur(
        self,
        image: np.ndarray,
        max_kernel: int = 5
    ) -> np.ndarray:
        k = random.choice([3, 5])
        k = min(
            k,
            max_kernel
        )
        if k % 2 == 0:
            k -= 1
        kernel = np.zeros(
            (k, k),
            dtype=np.float32
        )
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        return cv2.filter2D(
            image,
            -1,
            kernel
        )

    # =========================================================
    # 6. Salt & Pepper
    # =========================================================
    def salt_pepper(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        output = image.copy()
        h, w = output.shape
        count = int(self.salt_pepper_amount * h * w)
        if count <= 0:
            return output
        ys = np.random.randint(0, h, size=count )
        xs = np.random.randint(0, w, size=count )
        mask_salt = (
            np.random.random(count)
            < self.salt_pepper_salt_ratio
        )
        output[
            ys[mask_salt],
            xs[mask_salt]
        ] = 1.0
        output[
            ys[~mask_salt],
            xs[~mask_salt]
        ] = 0.0
        return output

    # =========================================================
    # Pipeline
    # =========================================================
    def __call__(
        self,
        image: ArrayLike
    ) -> ArrayLike:
        arr, was_torch = self._to_numpy(image)
        rng = np.random.default_rng()
        # Elastic
        if random.random() < self.prob:
            arr = self.elastic_distortion(arr, rng=rng )
        # Morphology
        if random.random() < self.prob:
            operation = random.choice(
                (
                    "dilation",
                    "erosion"
                )
            )
            arr = self.morphology( arr, operation)
        # Affine
        if random.random() < self.prob:
            arr = self.affine(arr)
        # Blur
        if random.random() < self.prob:
            if random.random() < 0.5:
                arr = self.motion_blur(arr)
            else:
                arr = self.gaussian_blur(arr)
        # Salt & Pepper
        if random.random() < self.prob:
            arr = self.salt_pepper(arr)
        return self._to_output(
            arr,
            was_torch
        )
__all__ = [
    "HTRAugmenter"
]