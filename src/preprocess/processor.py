"""
Image Preprocessing processor for OCR.

Applies various preprocessing techniques to improve OCR accuracy.
Each method is configurable and can be combined.

Key principle: We test multiple preprocessing methods and let the
engine/service select the best one — we never assume preprocessing
always improves accuracy.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


class ImageProcessor:
    """Handles all image preprocessing for OCR."""

    # Available preprocessing methods
    METHODS = [
        "none",                # No preprocessing
        "gray",                # Convert to grayscale
        "gray_norm",           # Grayscale + histogram normalization
        "gray_sharpen",        # Grayscale + sharpening
        "gray_norm_sharpen",   # Grayscale + normalize + sharpen
        "gray_blur_thresh",    # Grayscale + Gaussian blur + adaptive threshold
        "gray_thresh",         # Grayscale + adaptive threshold
        "gray_clahe",          # Grayscale + CLAHE (contrast enhancement)
    ]

    def apply(self, image_path: str, methods: List[str]) -> Optional[np.ndarray]:
        """
        Apply a sequence of preprocessing methods to an image.

        Args:
            image_path: Path to image
            methods: List of method names to apply in order

        Returns:
            Preprocessed image as numpy array (BGR format)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            return self.apply_to_array(img, methods)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def apply_to_array(self, img: np.ndarray, methods: List[str]) -> np.ndarray:
        """Apply preprocessing methods to an existing image array."""
        result = img.copy()

        for method in methods:
            result = self._apply_single(result, method)

        return result

    def _apply_single(self, img: np.ndarray, method: str) -> np.ndarray:
        """Apply a single preprocessing method."""
        if method == "none":
            return img

        elif method == "gray":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif method == "gray_norm":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        elif method == "gray_sharpen":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(gray, -1, kernel)

        elif method == "gray_norm_sharpen":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(normalized, -1, kernel)

        elif method == "gray_blur_thresh":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            return cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 3
            )

        elif method == "gray_thresh":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        elif method == "gray_clahe":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(gray)

        else:
            return img

    def resize_for_ocr(
        self, img: np.ndarray, max_dim: int = 1600
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize image to maximum dimension for OCR.
        Larger images generally give better OCR for handwriting.
        """
        h, w = img.shape[:2]
        if max(w, h) <= max_dim:
            return img, (w, h)

        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized, (new_w, new_h)

    def detect_strikethrough(self, image_path: str) -> List[dict]:
        """
        Detect strike-through lines in the image.
        Uses Hough Line Transform to find horizontal lines through text.

        Returns list of {bbox, angle, length, y_position}
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureH=3)

        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        strikethroughs = []
        if lines is not None:
            h, w = gray.shape
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))

                # Horizontal or near-horizontal lines are potential strikethroughs
                if abs(angle) < 20:
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    # Filter out lines that are clearly too long (page borders)
                    if length > w * 0.1 and length < w * 0.95:
                        y_pos = (y1 + y2) / 2
                        strikethroughs.append({
                            'start': [int(x1), int(y1)],
                            'end': [int(x2), int(y2)],
                            'y_position': float(y_pos),
                            'length': float(length),
                            'angle': float(angle),
                        })

        return strikethroughs
