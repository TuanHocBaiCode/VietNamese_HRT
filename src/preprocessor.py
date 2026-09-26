import cv2
import numpy as np
import torch

class ImagePreprocessor:
    def __init__(self, target_height: int = 64, target_width: int = 512):
        self.target_height = target_height
        self.target_width = target_width

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Tự động phát hiện góc nghiêng của dòng chữ và xoay thẳng về phương ngang."""
        # Nghịch đảo ảnh để chữ thành màu trắng, nền màu đen nhằm tìm tọa độ nét chữ
        inv_image = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(inv_image > 0))
        
        if len(coords) < 10:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        
        # Hiệu chỉnh góc xoay theo hệ trục OpenCV
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Giới hạn góc xoay tối đa tránh xoay lộn ngược tài liệu
        if abs(angle) > 20:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        return rotated

    def binarize(self, gray_image: np.ndarray) -> np.ndarray:
        """Nhị phân hóa bằng phương pháp Otsu kết hợp làm mịn nhẹ Gaussian."""
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Resize ảnh giữ nguyên tỷ lệ khung hình (Aspect Ratio) và Pad nền trắng
        để ảnh khớp đúng kích thước (target_height, target_width).
        """
        h, w = image.shape[:2]
        
        # Tính tỷ lệ thu phóng theo chiều cao target_height
        scale = self.target_height / float(h)
        new_w = int(w * scale)
        
        if new_w > self.target_width:
            # Nếu dòng chữ quá dài, nén chiều ngang cho vừa target_width
            resized = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            return resized
        else:
            resized = cv2.resize(image, (new_w, self.target_height), interpolation=cv2.INTER_AREA)
            # Tạo ảnh canvas màu trắng có kích thước chuẩn
            padded = np.full((self.target_height, self.target_width), 255, dtype=np.uint8)
            # Dán ảnh đã resize vào phía bên trái của canvas
            padded[:, :new_w] = resized
            return padded

    def process(self, image_path: str) -> torch.Tensor:
        """Pipeline trọn vẹn: Đọc ảnh -> Xám -> Xoay thẳng -> Resize/Pad -> Chuẩn hóa Tensor [1, H, W]."""
        # 1. Đọc ảnh dạng Grayscale
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Không thể đọc file ảnh: {image_path}")

        # 2. Căn chỉnh góc nghiêng
        deskewed = self.deskew(gray)

        # 3. Resize giữ tỷ lệ và pad nền trắng
        processed_img = self.resize_with_padding(deskewed)

        # 4. Chuẩn hóa giá trị pixel về dải [-1.0, 1.0] hoặc [0.0, 1.0]
        normalized = processed_img.astype(np.float32) / 255.0
        # Đưa về Tensor có shape: (1, H, W)
        tensor_img = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        
        return tensor_img