"""
Tests for Face Verification accuracy.
"""

import pytest
import numpy as np
from services.face_verifier import FaceVerifier, face_verifier


class TestFaceVerifier:
    """Test suite for face verification."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        verifier = FaceVerifier()
        
        # Identical embeddings should have similarity = 1
        embedding = np.random.randn(512)
        similarity = verifier.cosine_similarity(embedding, embedding)
        assert 0.99 < similarity <= 1.0, f"Identical embeddings should have sim ~1, got {similarity}"
        
        # Opposite embeddings should have similarity = 0
        opposite = -embedding
        similarity = verifier.cosine_similarity(embedding, opposite)
        assert 0.0 <= similarity < 0.1, f"Opposite embeddings should have sim ~0, got {similarity}"
        
        # Orthogonal embeddings
        orth1 = np.zeros(512)
        orth1[0] = 1
        orth2 = np.zeros(512)
        orth2[1] = 1
        similarity = verifier.cosine_similarity(orth1, orth2)
        assert 0.49 < similarity < 0.51, f"Orthogonal should have sim ~0.5, got {similarity}"
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        verifier = FaceVerifier()
        
        # Same embedding = distance 0
        embedding = np.random.randn(512)
        distance = verifier.euclidean_distance(embedding, embedding)
        assert distance < 0.001, f"Same embedding should have distance ~0, got {distance}"
        
        # Different embeddings should have positive distance
        other = np.random.randn(512)
        distance = verifier.euclidean_distance(embedding, other)
        assert distance > 0, "Different embeddings should have positive distance"
    
    def test_verify_embeddings_same_person(self):
        """Test verification with similar embeddings (same person)."""
        verifier = FaceVerifier()
        
        # Simulate same person with slight variation
        base_embedding = np.random.randn(512)
        # Add small noise to simulate lighting/angle changes
        noise = np.random.randn(512) * 0.1
        similar_embedding = base_embedding + noise
        
        # Normalize
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
        
        result = verifier.verify_embeddings(
            base_embedding,
            similar_embedding,
            security_level='medium'
        )
        
        # With small noise, similarity should be high
        assert result.similarity_score > 0.7, f"Similar embeddings should have high score: {result.similarity_score}"
    
    def test_verify_embeddings_different_person(self):
        """Test verification with different embeddings (different person)."""
        verifier = FaceVerifier()
        
        # Two random embeddings (different people)
        embedding1 = np.random.randn(512)
        embedding2 = np.random.randn(512)
        
        # Normalize
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        result = verifier.verify_embeddings(
            embedding1,
            embedding2,
            security_level='medium'
        )
        
        # Random embeddings should generally not match
        # Due to randomness, we check the score is reasonable (around 0.5)
        assert 0.3 < result.similarity_score < 0.7, f"Random embeddings score: {result.similarity_score}"
    
    def test_security_levels(self):
        """Test different security level thresholds."""
        verifier = FaceVerifier()
        
        assert verifier.THRESHOLDS['low'] == 0.5
        assert verifier.THRESHOLDS['medium'] == 0.6
        assert verifier.THRESHOLDS['high'] == 0.7
        assert verifier.THRESHOLDS['strict'] == 0.8
        
        # Same embedding at different security levels
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)
        
        # With identical embeddings, all levels should pass
        for level in ['low', 'medium', 'high', 'strict']:
            result = verifier.verify_embeddings(embedding, embedding, level)
            assert result.verified, f"Same embedding should pass {level} security"


class TestFaceVerificationAccuracy:
    """Accuracy benchmarks for face verification."""
    
    def test_same_person_detection_rate(self):
        """Test true positive rate for same person."""
        verifier = FaceVerifier()
        
        true_positives = 0
        trials = 100
        
        for _ in range(trials):
            # Simulate same person with variation
            base = np.random.randn(512)
            base = base / np.linalg.norm(base)
            
            # Add realistic noise (about 15% variation)
            noise = np.random.randn(512) * 0.15
            variant = base + noise
            variant = variant / np.linalg.norm(variant)
            
            result = verifier.verify_embeddings(base, variant, 'medium')
            if result.verified:
                true_positives += 1
        
        tpr = true_positives / trials
        # We expect at least 70% TPR with medium security
        assert tpr >= 0.70, f"True positive rate {tpr} below threshold"
    
    def test_different_person_rejection_rate(self):
        """Test true negative rate for different people."""
        verifier = FaceVerifier()
        
        true_negatives = 0
        trials = 100
        
        for _ in range(trials):
            # Two completely random embeddings
            emb1 = np.random.randn(512)
            emb1 = emb1 / np.linalg.norm(emb1)
            
            emb2 = np.random.randn(512)
            emb2 = emb2 / np.linalg.norm(emb2)
            
            result = verifier.verify_embeddings(emb1, emb2, 'medium')
            if not result.verified:
                true_negatives += 1
        
        tnr = true_negatives / trials
        # We expect at least 80% TNR to avoid false positives
        assert tnr >= 0.80, f"True negative rate {tnr} below threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
