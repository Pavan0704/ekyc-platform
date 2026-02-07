"""
Image processing utilities for in-memory handling.
All images are processed without touching disk for security.
"""

import base64
import io
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import cv2


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 string to OpenCV image array.
    Processes entirely in memory - no disk I/O.
    
    Args:
        base64_string: Base64 encoded image (with or without data URL prefix)
        
    Returns:
        OpenCV BGR image array
        
    Raises:
        ValueError: If image cannot be decoded
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
            
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def encode_image_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
    """
    Encode OpenCV image array to base64 string.
    
    Args:
        image: OpenCV BGR image array
        format: Output format (jpeg, png)
        
    Returns:
        Base64 encoded string
    """
    # Encode to bytes
    if format.lower() == "png":
        _, buffer = cv2.imencode('.png', image)
    else:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Convert to base64
    return base64.b64encode(buffer).decode('utf-8')


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image: OpenCV BGR image
        
    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better text contrast
    # This helps with varying lighting conditions on ID cards
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Apply slight sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def preprocess_for_face_detection(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for face detection.
    
    Args:
        image: OpenCV BGR image
        
    Returns:
        RGB image suitable for MediaPipe/FaceNet
    """
    # Convert BGR to RGB (MediaPipe and most ML models expect RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


def resize_image(image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: OpenCV image array
        max_dimension: Maximum width or height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_dimension:
        return image
    
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def extract_face_region(
    image: np.ndarray, 
    bbox: Tuple[int, int, int, int],
    padding: float = 0.2
) -> np.ndarray:
    """
    Extract face region from image with padding.
    
    Args:
        image: OpenCV image array
        bbox: Bounding box (x, y, width, height)
        padding: Padding ratio around face
        
    Returns:
        Cropped face region
    """
    x, y, w, h = bbox
    height, width = image.shape[:2]
    
    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    # Calculate padded coordinates with bounds checking
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(width, x + w + pad_w)
    y2 = min(height, y + h + pad_h)
    
    return image[y1:y2, x1:x2]


def align_face(image: np.ndarray, landmarks: dict) -> np.ndarray:
    """
    Align face based on eye landmarks for better verification.
    
    Args:
        image: OpenCV image with face
        landmarks: Dictionary with 'left_eye' and 'right_eye' coordinates
        
    Returns:
        Aligned face image
    """
    left_eye = landmarks.get('left_eye')
    right_eye = landmarks.get('right_eye')
    
    if not left_eye or not right_eye:
        return image
    
    # Calculate angle between eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Get center point between eyes
    center = ((left_eye[0] + right_eye[0]) // 2, 
              (left_eye[1] + right_eye[1]) // 2)
    
    # Rotate image to align eyes horizontally
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, rotation_matrix, (width, height),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    
    return aligned


def validate_image_quality(image: np.ndarray) -> dict:
    """
    Validate image quality for KYC processing.
    
    Args:
        image: OpenCV image array
        
    Returns:
        Dictionary with quality metrics and pass/fail status
    """
    height, width = image.shape[:2]
    
    # Check minimum resolution
    min_dimension = min(height, width)
    resolution_ok = min_dimension >= 480
    
    # Check brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    brightness = np.mean(gray)
    brightness_ok = 40 < brightness < 220
    
    # Check blur (using Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_ok = laplacian_var > 100
    
    # Check contrast
    contrast = gray.std()
    contrast_ok = contrast > 30
    
    return {
        "resolution": {"value": f"{width}x{height}", "ok": resolution_ok},
        "brightness": {"value": float(brightness), "ok": brightness_ok},
        "blur_score": {"value": float(laplacian_var), "ok": blur_ok},
        "contrast": {"value": float(contrast), "ok": contrast_ok},
        "overall_ok": all([resolution_ok, brightness_ok, blur_ok, contrast_ok])
    }
