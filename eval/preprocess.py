"""
Image preprocessing for Warung Lubu OCR — Step 1.

Each step can be turned on/off individually via --steps flag.
Results saved to debug/<image_name>/<step>.png for visual inspection.

Usage:
    from eval.preprocess import Preprocessor
    p = Preprocessor(image_path)
    results = p.apply_all()  # returns dict of {step_name: image_array}

Or CLI:
    python eval/preprocess.py --image D:/dataset/IMG_20261008_115733.jpg --steps auto_rotate,deskew,denoise,binarize
"""
import os
import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# --- Step 1: EXIF Correction + Auto-Rotate ---
def auto_rotate_by_orientation(img: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Detect if image is rotated 90/180/270 degrees.
    Uses a heuristic: try OCR on all 4 rotations, pick the one with highest
    text-like horizontal density. For notebooks, text is usually horizontal.
    """
    candidate = img
    method = "none"

    h, w = img.shape[:2]
    if h > w * 1.2:  # portrait = likely correct for notebook
        return img, "portrait_ok"

    # Try 90 and 270 rotation, check which gives more "text-like" rows
    best = img
    best_score = -1
    best_method = "none"
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = img
        else:
            rotated = np.rot90(img, k=angle // 90)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated
        # Horizontal projection (sum of dark pixels per row)
        proj = np.sum(gray < 128, axis=1)
        # Score: how bimodal is the projection (text rows = peaks, gaps = valleys)
        score = np.std(proj) if len(proj) > 0 else 0
        if score > best_score:
            best_score = score
            best = rotated
            best_method = f"rotated_{angle}" if angle != 0 else "original"

    return best, best_method


# --- Step 2: Page Detection ---
def detect_page_and_crop(img: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Detect the page boundary via largest contour and apply perspective transform.
    Returns cropped page (or original if detection fails).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img, "no_contours"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    h, w = img.shape[:2]

    if area < (h * w * 0.3):
        return img, "page_too_small"

    # Get bounding rect
    x, y, w_rect, h_rect = cv2.boundingRect(largest)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_rect = min(img.shape[1] - x, w_rect + 2 * margin)
    h_rect = min(img.shape[0] - y, h_rect + 2 * margin)

    cropped = img[y:y + h_rect, x:x + w_rect]
    return cropped, f"cropped_{x},{y},{w_rect},{h_rect}"


# --- Step 3: Deskew ---
def deskew_image(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Deskew using Hough line detection.
    Returns (deskewed_image, angle_degrees).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 5:
        return img, 0.0

    angle = 0.0
    try:
        # Use minAreaRect for angle estimation
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -(angle)
    except Exception:
        angle = 0.0

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle


# --- Step 4: Lighting Normalization (shadow removal) ---
def normalize_lighting(img: np.ndarray) -> np.ndarray:
    """
    Remove shadows and normalize lighting.
    Uses morphological closing on background, then divide.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Estimate background via morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Divide to remove lighting variation
    normalized = cv2.divide(gray, bg, scale=255)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)

    return normalized


# --- Step 5: Denoise ---
def denoise_image(img: np.ndarray, method: str = "fastnlmeans") -> np.ndarray:
    """
    Denoise the image.
    Options: fastnlmeans, bilateral, gaussian
    """
    if method == "fastnlmeans":
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)
    else:
        return img


# --- Step 6: Remove ruled lines ---
def remove_ruled_lines(img: np.ndarray) -> np.ndarray:
    """
    Remove horizontal ruled lines (book lines) using morphology.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Create kernel for horizontal lines
    h = gray.shape[0]
    w = gray.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))

    # Remove horizontal lines
    no_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    # Inpaint the removed regions
    mask = cv2.absdiff(gray, no_lines)
    inpainted = cv2.inpaint(img if len(img.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                            mask, 3, cv2.INPAINT_TELEA)

    return inpainted


# --- Step 7: Binarization ---
def binarize_image(img: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert to binary.
    Options: otsu, adaptive_gaussian, adaptive_mean, sauvola
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
        return binary
    elif method == "adaptive_mean":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
        return binary
    elif method == "sauvola":
        # Sauvola binarization
        binary = cv2.ximgproc.niBlackThreshold(
            gray, 255, cv2.THRESH_BINARY, 11, -0.2
        )
        return binary
    else:
        return gray


# --- Step 8: Upscale ---
def upscale_image(img: np.ndarray, target_size: int = 1600) -> np.ndarray:
    """
    Upscale image so text height is ~30-40px.
    Target max dimension = 1600 (good trade-off for speed/accuracy).
    """
    h, w = img.shape[:2]
    if max(h, w) > target_size:
        return img

    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Use cubic interpolation for best quality
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


class Preprocessor:
    """
    Chain of preprocessing steps for Warung Lupi notebook OCR.
    Each step is toggleable.

    Pipeline order:
        1. auto_rotate
        2. detect_page (crop)
        3. deskew
        4. normalize_lighting
        5. denoise
        6. remove_ruled_lines
        7. binarize
        8. upscale
    """

    DEFAULT_STEPS = [
        "normalize_lighting",
        "denoise",
        "binarize",
        "upscale",
    ]

    def __init__(self, image_path: str, debug_dir: str = None):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        self.debug_dir = debug_dir or f"eval/debug/{os.path.basename(image_path).rsplit('.', 1)[0]}"
        self.steps_applied = []

    def apply_step(self, step: str, img: np.ndarray) -> np.ndarray:
        """Apply a single preprocessing step."""
        if step == "auto_rotate":
            result, method = auto_rotate_by_orientation(img)
            self.steps_applied.append(("auto_rotate", method))
        elif step == "detect_page":
            result, method = detect_page_and_crop(img)
            self.steps_applied.append(("detect_page", method))
        elif step == "deskew":
            result, angle = deskew_image(img)
            self.steps_applied.append(("deskew", f"angle={angle:.1f}"))
        elif step == "normalize_lighting":
            result = normalize_lighting(img)
            self.steps_applied.append(("normalize_lighting", "CLAHE"))
        elif step == "denoise":
            result = denoise_image(img)
            self.steps_applied.append(("denoise", "fastnlmeans"))
        elif step == "remove_ruled_lines":
            result = remove_ruled_lines(img)
            self.steps_applied.append(("remove_ruled_lines", "horizontal_morph"))
        elif step == "binarize":
            result = binarize_image(img, "adaptive_gaussian")
            self.steps_applied.append(("binarize", "adaptive_gaussian"))
        elif step == "upscale":
            result = upscale_image(img)
            self.steps_applied.append(("upscale", f"max_dim={max(result.shape[:2])}"))
        else:
            logger.warning(f"Unknown step: {step}")
            return img

        # Save debug image
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            debug_path = os.path.join(self.debug_dir, f"{step}.png")
            try:
                cv2.imwrite(debug_path, result)
            except Exception as e:
                logger.warning(f"Could not save debug image: {e}")

        return result

    def apply_pipeline(self, steps: Optional[list] = None) -> tuple[np.ndarray, list]:
        """
        Apply a sequence of preprocessing steps.
        Returns (processed_image, list_of_step_descriptions).
        """
        steps = steps or self.DEFAULT_STEPS
        img = self.original.copy()
        self.steps_applied = []

        for step in steps:
            img = self.apply_step(step, img)

        return img, self.steps_applied

    def save_processed(self, img: np.ndarray, path: str) -> None:
        """Save processed image to file."""
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, img)
        logger.info(f"Saved processed image to {path}")
