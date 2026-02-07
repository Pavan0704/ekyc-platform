"""
Face Verification Service using FaceNet.
Compares face embeddings to verify identity match between ID photo and selfie.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

# FaceNet is imported lazily
_facenet_model = None
_mtcnn = None


def get_facenet_model():
    """Lazy initialization of FaceNet model."""
    global _facenet_model
    if _facenet_model is None:
        from facenet_pytorch import InceptionResnetV1
        _facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    return _facenet_model


def get_mtcnn():
    """Lazy initialization of MTCNN for face alignment."""
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=False
        )
    return _mtcnn


@dataclass
class VerificationResult:
    """Result of face verification."""
    verified: bool
    similarity_score: float
    threshold_used: float
    embedding_distance: float
    message: str = ""


class FaceVerifier:
    """
    Face verification using FaceNet embeddings.
    Compares faces using cosine similarity of 512-dimensional embeddings.
    """
    
    # Similarity threshold for verification (higher = stricter)
    DEFAULT_THRESHOLD = 0.6
    
    # Thresholds for different security levels
    THRESHOLDS = {
        'low': 0.5,
        'medium': 0.6,
        'high': 0.7,
        'strict': 0.8
    }
    
    def __init__(self):
        self.model = None
        self.mtcnn = None
    
    def _ensure_models(self):
        """Ensure FaceNet model is loaded."""
        if self.model is None:
            self.model = get_facenet_model()
        if self.mtcnn is None:
            self.mtcnn = get_mtcnn()
    
    def _preprocess_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess face image for FaceNet.
        
        Args:
            image: RGB image array
            
        Returns:
            Preprocessed face tensor, or None if no face found
        """
        import torch
        from PIL import Image
        
        self._ensure_models()
        
        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Extract aligned face
        face_tensor = self.mtcnn(pil_image)
        
        if face_tensor is None:
            return None
        
        return face_tensor
    
    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.
        
        Args:
            image: RGB image array
            
        Returns:
            512-dimensional face embedding, or None if no face found
        """
        import torch
        
        self._ensure_models()
        
        face_tensor = self._preprocess_face(image)
        
        if face_tensor is None:
            return None
        
        # Add batch dimension if needed
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        return embedding.numpy()[0]
    
    def cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Convert from [-1, 1] to [0, 1]
        return float((similarity + 1) / 2)
    
    def euclidean_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate Euclidean distance between embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Euclidean distance (lower = more similar)
        """
        return float(np.linalg.norm(embedding1 - embedding2))
    
    def verify_faces(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        security_level: str = 'medium'
    ) -> VerificationResult:
        """
        Verify if two images contain the same person.
        
        Args:
            image1: First RGB image (e.g., ID photo)
            image2: Second RGB image (e.g., selfie)
            security_level: 'low', 'medium', 'high', or 'strict'
            
        Returns:
            VerificationResult with match status and scores
        """
        self._ensure_models()
        
        # Get threshold for security level
        threshold = self.THRESHOLDS.get(security_level, self.DEFAULT_THRESHOLD)
        
        # Extract embeddings
        embedding1 = self.get_embedding(image1)
        
        if embedding1 is None:
            return VerificationResult(
                verified=False,
                similarity_score=0.0,
                threshold_used=threshold,
                embedding_distance=float('inf'),
                message="No face detected in first image (ID document)"
            )
        
        embedding2 = self.get_embedding(image2)
        
        if embedding2 is None:
            return VerificationResult(
                verified=False,
                similarity_score=0.0,
                threshold_used=threshold,
                embedding_distance=float('inf'),
                message="No face detected in second image (selfie)"
            )
        
        # Calculate similarity
        similarity = self.cosine_similarity(embedding1, embedding2)
        distance = self.euclidean_distance(embedding1, embedding2)
        
        # Verify against threshold
        verified = similarity >= threshold
        
        return VerificationResult(
            verified=verified,
            similarity_score=similarity,
            threshold_used=threshold,
            embedding_distance=distance,
            message="Face verification successful" if verified else "Faces do not match"
        )
    
    def verify_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        security_level: str = 'medium'
    ) -> VerificationResult:
        """
        Verify if two embeddings represent the same person.
        Useful when embeddings are pre-computed.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            security_level: Security level for threshold
            
        Returns:
            VerificationResult with match status and scores
        """
        threshold = self.THRESHOLDS.get(security_level, self.DEFAULT_THRESHOLD)
        
        similarity = self.cosine_similarity(embedding1, embedding2)
        distance = self.euclidean_distance(embedding1, embedding2)
        
        verified = similarity >= threshold
        
        return VerificationResult(
            verified=verified,
            similarity_score=similarity,
            threshold_used=threshold,
            embedding_distance=distance,
            message="Face verification successful" if verified else "Faces do not match"
        )
    
    def get_quality_score(self, image: np.ndarray) -> dict:
        """
        Assess quality of face in image for verification.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with quality metrics
        """
        import cv2
        
        # Check if face can be detected
        face_tensor = self._preprocess_face(image)
        face_detected = face_tensor is not None
        
        if not face_detected:
            return {
                "face_detected": False,
                "quality_score": 0.0,
                "issues": ["No face detected"]
            }
        
        # Analyze image quality
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Check blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_ok = laplacian_var > 100
        
        # Check brightness
        brightness = np.mean(gray)
        brightness_ok = 50 < brightness < 200
        
        # Check contrast
        contrast = np.std(gray)
        contrast_ok = contrast > 40
        
        issues = []
        if not blur_ok:
            issues.append("Image is blurry")
        if not brightness_ok:
            if brightness <= 50:
                issues.append("Image is too dark")
            else:
                issues.append("Image is too bright")
        if not contrast_ok:
            issues.append("Low contrast")
        
        # Calculate overall score
        score = sum([blur_ok, brightness_ok, contrast_ok]) / 3.0
        
        return {
            "face_detected": True,
            "quality_score": score,
            "blur_score": float(laplacian_var),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "issues": issues
        }


# Singleton instance
face_verifier = FaceVerifier()
