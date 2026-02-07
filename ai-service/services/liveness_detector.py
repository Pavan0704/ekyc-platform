"""
Liveness Detection Service.
Implements anti-spoofing measures using blink detection and head pose estimation.
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random
import uuid


class ChallengeType(Enum):
    """Types of liveness challenges."""
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD = "nod"


@dataclass
class LivenessChallenge:
    """Liveness challenge configuration."""
    challenge_id: str
    challenge_type: ChallengeType
    instructions: str
    required_count: int = 1  # How many times action must be detected


@dataclass
class LivenessResult:
    """Result of liveness verification."""
    passed: bool
    score: float
    challenge_completed: bool
    details: Dict
    message: str = ""


class LivenessDetector:
    """
    Liveness detection using eye aspect ratio (EAR) and head pose.
    Prevents photo-of-photo and video replay attacks.
    """
    
    # Eye Aspect Ratio threshold for blink detection
    EAR_THRESHOLD = 0.21
    
    # Head rotation thresholds (in degrees)
    HEAD_ROTATION_THRESHOLD = 15.0
    HEAD_NOD_THRESHOLD = 10.0
    
    # Challenge instructions
    CHALLENGE_INSTRUCTIONS = {
        ChallengeType.BLINK: "Please blink your eyes naturally 2-3 times",
        ChallengeType.TURN_LEFT: "Please slowly turn your head to the left",
        ChallengeType.TURN_RIGHT: "Please slowly turn your head to the right",
        ChallengeType.NOD: "Please nod your head up and down",
    }
    
    def __init__(self):
        # Store active challenges (in production, use Redis or similar)
        self._active_challenges: Dict[str, LivenessChallenge] = {}
    
    def create_challenge(
        self, 
        challenge_type: Optional[ChallengeType] = None
    ) -> LivenessChallenge:
        """
        Create a new liveness challenge.
        
        Args:
            challenge_type: Specific challenge type, or random if None
            
        Returns:
            LivenessChallenge with unique ID
        """
        if challenge_type is None:
            # Default to blink as it's most reliable
            challenge_type = ChallengeType.BLINK
        
        challenge = LivenessChallenge(
            challenge_id=str(uuid.uuid4()),
            challenge_type=challenge_type,
            instructions=self.CHALLENGE_INSTRUCTIONS[challenge_type],
            required_count=2 if challenge_type == ChallengeType.BLINK else 1
        )
        
        self._active_challenges[challenge.challenge_id] = challenge
        return challenge
    
    def get_challenge(self, challenge_id: str) -> Optional[LivenessChallenge]:
        """Get an active challenge by ID."""
        return self._active_challenges.get(challenge_id)
    
    def calculate_ear(self, eye_landmarks: Dict) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Where points are:
        - p1: outer corner, p4: inner corner
        - p2, p3: top points, p5, p6: bottom points
        
        Args:
            eye_landmarks: Dictionary with eye corner and lid points
            
        Returns:
            Eye aspect ratio (lower = more closed)
        """
        outer = eye_landmarks.get('outer')
        inner = eye_landmarks.get('inner')
        top = eye_landmarks.get('top')
        bottom = eye_landmarks.get('bottom')
        
        if not all([outer, inner, top, bottom]):
            return 0.3  # Default open eye value
        
        # Horizontal distance
        horizontal = np.sqrt(
            (inner[0] - outer[0]) ** 2 + 
            (inner[1] - outer[1]) ** 2
        )
        
        # Vertical distance (simplified - using just one pair)
        vertical = np.sqrt(
            (top[0] - bottom[0]) ** 2 + 
            (top[1] - bottom[1]) ** 2
        )
        
        if horizontal == 0:
            return 0.3
        
        ear = vertical / horizontal
        return ear
    
    def detect_blink(self, frames_ear: List[float]) -> Tuple[bool, int]:
        """
        Detect blinks in a sequence of EAR values.
        
        A blink is detected when EAR drops below threshold and returns.
        
        Args:
            frames_ear: List of EAR values for each frame
            
        Returns:
            Tuple of (blink_detected, blink_count)
        """
        if len(frames_ear) < 3:
            return False, 0
        
        blink_count = 0
        in_blink = False
        
        for ear in frames_ear:
            if ear < self.EAR_THRESHOLD:
                if not in_blink:
                    in_blink = True
            else:
                if in_blink:
                    # Completed a blink
                    blink_count += 1
                    in_blink = False
        
        return blink_count > 0, blink_count
    
    def estimate_head_pose(
        self, 
        landmarks: Dict,
        image_size: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Estimate head pose (yaw, pitch, roll) from facial landmarks.
        
        Uses a simplified geometric approach based on key landmark positions.
        
        Args:
            landmarks: Dictionary with nose, eyes, mouth landmarks
            image_size: (width, height) of the image
            
        Returns:
            Dictionary with yaw, pitch, roll estimates in degrees
        """
        width, height = image_size
        
        nose = landmarks.get('nose_tip')
        left_eye = landmarks.get('left_eye')
        right_eye = landmarks.get('right_eye')
        mouth_left = landmarks.get('mouth_left')
        mouth_right = landmarks.get('mouth_right')
        chin = landmarks.get('chin')
        
        if not all([nose, left_eye, right_eye]):
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        # Estimate YAW (left-right rotation)
        # Based on nose position relative to eye centers
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset_x = nose[0] - eye_center_x
        eye_distance = abs(right_eye[0] - left_eye[0])
        
        if eye_distance > 0:
            yaw = np.degrees(np.arcsin(
                np.clip(nose_offset_x / (eye_distance * 0.5), -1, 1)
            ))
        else:
            yaw = 0.0
        
        # Estimate PITCH (up-down rotation)
        if chin and nose:
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            face_height = chin[1] - eye_center_y
            nose_vertical = nose[1] - eye_center_y
            
            if face_height > 0:
                pitch_ratio = nose_vertical / face_height
                # Normalize: typical ratio is ~0.3-0.4
                pitch = (pitch_ratio - 0.35) * 100  # Approximate degrees
            else:
                pitch = 0.0
        else:
            pitch = 0.0
        
        # Estimate ROLL (head tilt)
        roll = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))
        
        return {
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll)
        }
    
    def detect_head_turn(
        self,
        poses: List[Dict[str, float]],
        direction: str  # 'left' or 'right'
    ) -> bool:
        """
        Detect if head was turned in specified direction.
        
        Args:
            poses: List of head pose estimates for each frame
            direction: 'left' or 'right'
            
        Returns:
            True if turn was detected
        """
        if len(poses) < 5:
            return False
        
        yaw_values = [p['yaw'] for p in poses]
        
        if direction == 'left':
            # Look for significant negative yaw
            min_yaw = min(yaw_values)
            return min_yaw < -self.HEAD_ROTATION_THRESHOLD
        else:
            # Look for significant positive yaw
            max_yaw = max(yaw_values)
            return max_yaw > self.HEAD_ROTATION_THRESHOLD
    
    def detect_nod(self, poses: List[Dict[str, float]]) -> bool:
        """
        Detect if head nodded (up-down movement).
        
        Args:
            poses: List of head pose estimates for each frame
            
        Returns:
            True if nod was detected
        """
        if len(poses) < 5:
            return False
        
        pitch_values = [p['pitch'] for p in poses]
        
        # Look for pitch variation
        pitch_range = max(pitch_values) - min(pitch_values)
        return pitch_range > self.HEAD_NOD_THRESHOLD * 2
    
    def verify_liveness(
        self,
        challenge_id: str,
        eye_data: List[Dict],  # EAR values per frame
        pose_data: List[Dict[str, float]],  # Head poses per frame
        image_size: Tuple[int, int]
    ) -> LivenessResult:
        """
        Verify liveness based on captured frames data.
        
        Args:
            challenge_id: ID of the active challenge
            eye_data: Eye aspect ratios for each frame
            pose_data: Head pose estimates for each frame
            image_size: Size of the captured frames
            
        Returns:
            LivenessResult with verification outcome
        """
        challenge = self.get_challenge(challenge_id)
        
        if not challenge:
            return LivenessResult(
                passed=False,
                score=0.0,
                challenge_completed=False,
                details={"error": "Invalid challenge ID"},
                message="Challenge not found or expired"
            )
        
        # Verify based on challenge type
        if challenge.challenge_type == ChallengeType.BLINK:
            ear_values = [e.get('avg_ear', 0.3) for e in eye_data]
            blink_detected, blink_count = self.detect_blink(ear_values)
            
            passed = blink_count >= challenge.required_count
            score = min(1.0, blink_count / challenge.required_count)
            
            return LivenessResult(
                passed=passed,
                score=score,
                challenge_completed=passed,
                details={
                    "blinks_detected": blink_count,
                    "blinks_required": challenge.required_count,
                    "ear_values_sample": ear_values[:10]
                },
                message=f"Detected {blink_count} blinks" if passed else "Blink not detected"
            )
        
        elif challenge.challenge_type == ChallengeType.TURN_LEFT:
            turn_detected = self.detect_head_turn(pose_data, 'left')
            yaw_values = [p['yaw'] for p in pose_data]
            
            return LivenessResult(
                passed=turn_detected,
                score=1.0 if turn_detected else 0.0,
                challenge_completed=turn_detected,
                details={
                    "direction": "left",
                    "max_rotation": min(yaw_values),
                    "threshold": -self.HEAD_ROTATION_THRESHOLD
                },
                message="Head turn detected" if turn_detected else "Please turn your head more"
            )
        
        elif challenge.challenge_type == ChallengeType.TURN_RIGHT:
            turn_detected = self.detect_head_turn(pose_data, 'right')
            yaw_values = [p['yaw'] for p in pose_data]
            
            return LivenessResult(
                passed=turn_detected,
                score=1.0 if turn_detected else 0.0,
                challenge_completed=turn_detected,
                details={
                    "direction": "right",
                    "max_rotation": max(yaw_values),
                    "threshold": self.HEAD_ROTATION_THRESHOLD
                },
                message="Head turn detected" if turn_detected else "Please turn your head more"
            )
        
        elif challenge.challenge_type == ChallengeType.NOD:
            nod_detected = self.detect_nod(pose_data)
            pitch_values = [p['pitch'] for p in pose_data]
            
            return LivenessResult(
                passed=nod_detected,
                score=1.0 if nod_detected else 0.0,
                challenge_completed=nod_detected,
                details={
                    "pitch_range": max(pitch_values) - min(pitch_values),
                    "threshold": self.HEAD_NOD_THRESHOLD * 2
                },
                message="Nod detected" if nod_detected else "Please nod your head more"
            )
        
        return LivenessResult(
            passed=False,
            score=0.0,
            challenge_completed=False,
            details={"error": "Unknown challenge type"},
            message="Invalid challenge type"
        )
    
    def cleanup_challenge(self, challenge_id: str):
        """Remove completed challenge from memory."""
        if challenge_id in self._active_challenges:
            del self._active_challenges[challenge_id]


# Singleton instance
liveness_detector = LivenessDetector()
