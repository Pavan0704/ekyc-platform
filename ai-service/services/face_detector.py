"""
Face Detection Service using MediaPipe.
Provides face detection and landmark extraction for KYC verification.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import numpy as np

# MediaPipe is imported lazily
_face_detection = None
_face_mesh = None


def get_face_detection():
    """Lazy initialization of MediaPipe face detection."""
    global _face_detection
    if _face_detection is None:
        import mediapipe as mp
        _face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
    return _face_detection


def get_face_mesh():
    """Lazy initialization of MediaPipe face mesh."""
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _face_mesh


@dataclass
class FaceDetectionResult:
    """Result of face detection."""
    detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    confidence: float = 0.0
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    face_image: Optional[np.ndarray] = None


class FaceDetector:
    """
    Face detection and landmark extraction using MediaPipe.
    """
    
    # Key landmark indices for MediaPipe Face Mesh
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    LANDMARK_INDICES = {
        'left_eye_center': 468,      # Left iris center
        'right_eye_center': 473,     # Right iris center
        'left_eye_outer': 33,
        'left_eye_inner': 133,
        'right_eye_outer': 263,
        'right_eye_inner': 362,
        'nose_tip': 4,
        'mouth_left': 61,
        'mouth_right': 291,
        'mouth_top': 13,
        'mouth_bottom': 14,
        'left_eyebrow_outer': 46,
        'right_eyebrow_outer': 276,
        'chin': 152,
        # Eye landmarks for blink detection (EAR calculation)
        'left_eye_top': 159,
        'left_eye_bottom': 145,
        'right_eye_top': 386,
        'right_eye_bottom': 374,
    }
    
    def __init__(self):
        self.face_detection = None
        self.face_mesh = None
    
    def _ensure_models(self):
        """Ensure detection models are loaded."""
        if self.face_detection is None:
            self.face_detection = get_face_detection()
        if self.face_mesh is None:
            self.face_mesh = get_face_mesh()
    
    def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detect face in image and extract bounding box.
        
        Args:
            image: RGB image array
            
        Returns:
            FaceDetectionResult with detection status and bbox
        """
        self._ensure_models()
        
        height, width = image.shape[:2]
        
        # Run face detection
        results = self.face_detection.process(image)
        
        if not results.detections:
            return FaceDetectionResult(detected=False)
        
        # Get first (most confident) detection
        detection = results.detections[0]
        
        # Extract bounding box (normalized) and convert to pixels
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        
        # Ensure bbox is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        
        return FaceDetectionResult(
            detected=True,
            bbox=(x, y, w, h),
            confidence=detection.score[0] if detection.score else 0.0
        )
    
    def detect_landmarks(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detect face and extract detailed landmarks.
        
        Args:
            image: RGB image array
            
        Returns:
            FaceDetectionResult with landmarks
        """
        self._ensure_models()
        
        height, width = image.shape[:2]
        
        # Run face mesh
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return FaceDetectionResult(detected=False)
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract key landmarks
        landmarks = {}
        for name, idx in self.LANDMARK_INDICES.items():
            landmark = face_landmarks.landmark[idx]
            landmarks[name] = (
                int(landmark.x * width),
                int(landmark.y * height)
            )
        
        # Calculate bounding box from landmarks
        x_coords = [lm.x * width for lm in face_landmarks.landmark]
        y_coords = [lm.y * height for lm in face_landmarks.landmark]
        
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)
        
        # Add some padding
        padding = int(min(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        return FaceDetectionResult(
            detected=True,
            bbox=(x, y, w, h),
            confidence=1.0,  # Face mesh doesn't provide confidence
            landmarks=landmarks
        )
    
    def extract_face(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int] = (160, 160)
    ) -> Optional[np.ndarray]:
        """
        Extract and resize face region for face verification.
        
        Args:
            image: RGB image array
            target_size: Output face size (width, height)
            
        Returns:
            Cropped and resized face image, or None if no face detected
        """
        import cv2
        
        result = self.detect_landmarks(image)
        
        if not result.detected:
            # Try simpler detection
            result = self.detect_face(image)
            if not result.detected:
                return None
        
        x, y, w, h = result.bbox
        
        # Extract and resize face
        face_image = image[y:y+h, x:x+w]
        
        if face_image.size == 0:
            return None
            
        face_resized = cv2.resize(face_image, target_size)
        
        return face_resized
    
    def get_eye_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Get detailed eye landmarks for blink detection.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with left and right eye landmark points
        """
        result = self.detect_landmarks(image)
        
        if not result.detected or not result.landmarks:
            return None
        
        landmarks = result.landmarks
        
        return {
            'left_eye': {
                'center': landmarks.get('left_eye_center'),
                'outer': landmarks.get('left_eye_outer'),
                'inner': landmarks.get('left_eye_inner'),
                'top': landmarks.get('left_eye_top'),
                'bottom': landmarks.get('left_eye_bottom'),
            },
            'right_eye': {
                'center': landmarks.get('right_eye_center'),
                'outer': landmarks.get('right_eye_outer'),
                'inner': landmarks.get('right_eye_inner'),
                'top': landmarks.get('right_eye_top'),
                'bottom': landmarks.get('right_eye_bottom'),
            }
        }
    
    def get_head_pose_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Get landmarks needed for head pose estimation.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with key facial points for pose estimation
        """
        result = self.detect_landmarks(image)
        
        if not result.detected or not result.landmarks:
            return None
        
        landmarks = result.landmarks
        
        return {
            'nose_tip': landmarks.get('nose_tip'),
            'chin': landmarks.get('chin'),
            'left_eye': landmarks.get('left_eye_outer'),
            'right_eye': landmarks.get('right_eye_outer'),
            'mouth_left': landmarks.get('mouth_left'),
            'mouth_right': landmarks.get('mouth_right'),
        }


# Singleton instance
face_detector = FaceDetector()
