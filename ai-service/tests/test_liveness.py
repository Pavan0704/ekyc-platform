"""
Tests for Liveness Detection.
"""

import pytest
import numpy as np
from services.liveness_detector import (
    LivenessDetector, 
    liveness_detector,
    ChallengeType,
    LivenessChallenge
)


class TestLivenessDetector:
    """Test suite for liveness detection."""
    
    def test_create_challenge(self):
        """Test challenge creation."""
        detector = LivenessDetector()
        
        # Create blink challenge
        challenge = detector.create_challenge(ChallengeType.BLINK)
        assert challenge.challenge_id is not None
        assert challenge.challenge_type == ChallengeType.BLINK
        assert "blink" in challenge.instructions.lower()
        
        # Create turn challenge
        challenge = detector.create_challenge(ChallengeType.TURN_LEFT)
        assert challenge.challenge_type == ChallengeType.TURN_LEFT
    
    def test_get_challenge(self):
        """Test challenge retrieval."""
        detector = LivenessDetector()
        
        # Create and retrieve
        challenge = detector.create_challenge()
        retrieved = detector.get_challenge(challenge.challenge_id)
        
        assert retrieved is not None
        assert retrieved.challenge_id == challenge.challenge_id
        
        # Non-existent challenge
        assert detector.get_challenge("invalid-id") is None
    
    def test_calculate_ear(self):
        """Test Eye Aspect Ratio calculation."""
        detector = LivenessDetector()
        
        # Simulated eye landmarks (open eye)
        open_eye = {
            'outer': (0, 50),
            'inner': (100, 50),
            'top': (50, 40),
            'bottom': (50, 60),
        }
        ear_open = detector.calculate_ear(open_eye)
        assert 0.2 < ear_open < 0.4, f"Open eye EAR: {ear_open}"
        
        # Simulated closed eye
        closed_eye = {
            'outer': (0, 50),
            'inner': (100, 50),
            'top': (50, 48),
            'bottom': (50, 52),
        }
        ear_closed = detector.calculate_ear(closed_eye)
        assert ear_closed < ear_open, "Closed eye should have lower EAR"
    
    def test_detect_blink(self):
        """Test blink detection from EAR sequence."""
        detector = LivenessDetector()
        
        # Simulated EAR values with a blink (EAR drops below threshold)
        ear_with_blink = [
            0.30, 0.30, 0.28, 0.25,  # Open
            0.20, 0.15, 0.12,         # Blinking (below threshold)
            0.18, 0.22, 0.28, 0.30    # Open again
        ]
        
        blink_detected, count = detector.detect_blink(ear_with_blink)
        assert blink_detected, "Blink should be detected"
        assert count >= 1, f"Expected at least 1 blink, got {count}"
        
        # No blink (EAR stays above threshold)
        ear_no_blink = [0.30, 0.29, 0.31, 0.28, 0.30, 0.29]
        blink_detected, count = detector.detect_blink(ear_no_blink)
        assert not blink_detected, "No blink should be detected"
    
    def test_estimate_head_pose(self):
        """Test head pose estimation."""
        detector = LivenessDetector()
        image_size = (640, 480)
        
        # Centered face (neutral pose)
        neutral_landmarks = {
            'nose_tip': (320, 280),
            'left_eye': (280, 200),
            'right_eye': (360, 200),
            'chin': (320, 400),
            'mouth_left': (290, 340),
            'mouth_right': (350, 340),
        }
        
        pose = detector.estimate_head_pose(neutral_landmarks, image_size)
        assert abs(pose['yaw']) < 10, f"Neutral yaw should be near 0, got {pose['yaw']}"
        assert abs(pose['roll']) < 5, f"Neutral roll should be near 0, got {pose['roll']}"
        
        # Head turned left (nose moves left relative to eyes)
        left_turn_landmarks = {
            'nose_tip': (260, 280),  # Nose moved left
            'left_eye': (280, 200),
            'right_eye': (360, 200),
            'chin': (280, 400),
            'mouth_left': (250, 340),
            'mouth_right': (310, 340),
        }
        
        pose_left = detector.estimate_head_pose(left_turn_landmarks, image_size)
        assert pose_left['yaw'] < -5, f"Left turn should have negative yaw, got {pose_left['yaw']}"
    
    def test_detect_head_turn(self):
        """Test head turn detection."""
        detector = LivenessDetector()
        
        # Sequence with left turn
        poses_left_turn = [
            {'yaw': 0, 'pitch': 0, 'roll': 0},
            {'yaw': -5, 'pitch': 0, 'roll': 0},
            {'yaw': -15, 'pitch': 0, 'roll': 0},
            {'yaw': -25, 'pitch': 0, 'roll': 0},
            {'yaw': -20, 'pitch': 0, 'roll': 0},
        ]
        
        assert detector.detect_head_turn(poses_left_turn, 'left'), "Left turn should be detected"
        assert not detector.detect_head_turn(poses_left_turn, 'right'), "Right turn should not be detected"
        
        # Sequence with right turn
        poses_right_turn = [
            {'yaw': 0, 'pitch': 0, 'roll': 0},
            {'yaw': 10, 'pitch': 0, 'roll': 0},
            {'yaw': 20, 'pitch': 0, 'roll': 0},
            {'yaw': 25, 'pitch': 0, 'roll': 0},
            {'yaw': 15, 'pitch': 0, 'roll': 0},
        ]
        
        assert detector.detect_head_turn(poses_right_turn, 'right'), "Right turn should be detected"
    
    def test_detect_nod(self):
        """Test head nod detection."""
        detector = LivenessDetector()
        
        # Sequence with nod
        poses_with_nod = [
            {'yaw': 0, 'pitch': 0, 'roll': 0},
            {'yaw': 0, 'pitch': 15, 'roll': 0},
            {'yaw': 0, 'pitch': 25, 'roll': 0},
            {'yaw': 0, 'pitch': 10, 'roll': 0},
            {'yaw': 0, 'pitch': -5, 'roll': 0},
        ]
        
        assert detector.detect_nod(poses_with_nod), "Nod should be detected"
        
        # No significant pitch change
        poses_no_nod = [
            {'yaw': 0, 'pitch': 0, 'roll': 0},
            {'yaw': 5, 'pitch': 2, 'roll': 0},
            {'yaw': -3, 'pitch': 3, 'roll': 0},
            {'yaw': 2, 'pitch': 1, 'roll': 0},
            {'yaw': 0, 'pitch': 2, 'roll': 0},
        ]
        
        assert not detector.detect_nod(poses_no_nod), "No nod should be detected"
    
    def test_verify_liveness_blink(self):
        """Test full liveness verification for blink challenge."""
        detector = LivenessDetector()
        
        # Create challenge
        challenge = detector.create_challenge(ChallengeType.BLINK)
        
        # Simulated eye data with blinks
        eye_data = [
            {'avg_ear': 0.30},
            {'avg_ear': 0.28},
            {'avg_ear': 0.15},  # Blink
            {'avg_ear': 0.12},
            {'avg_ear': 0.25},
            {'avg_ear': 0.30},
            {'avg_ear': 0.14},  # Another blink
            {'avg_ear': 0.11},
            {'avg_ear': 0.26},
            {'avg_ear': 0.30},
        ]
        
        pose_data = [{'yaw': 0, 'pitch': 0, 'roll': 0}] * 10
        
        result = detector.verify_liveness(
            challenge.challenge_id,
            eye_data,
            pose_data,
            (640, 480)
        )
        
        assert result.passed, f"Liveness should pass: {result.message}"
        assert result.score > 0.5
    
    def test_cleanup_challenge(self):
        """Test challenge cleanup."""
        detector = LivenessDetector()
        
        challenge = detector.create_challenge()
        assert detector.get_challenge(challenge.challenge_id) is not None
        
        detector.cleanup_challenge(challenge.challenge_id)
        assert detector.get_challenge(challenge.challenge_id) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
