"""
Pydantic schemas for e-KYC API request/response validation.
All data models use strict validation for security.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import datetime
import base64
import re


class DocumentUploadRequest(BaseModel):
    """Request schema for document upload and OCR extraction."""
    image: str = Field(..., description="Base64 encoded image of the ID document")
    document_type: Literal["national_id", "passport", "drivers_license"] = Field(
        ..., description="Type of identity document"
    )
    
    @field_validator('image')
    @classmethod
    def validate_base64_image(cls, v: str) -> str:
        """Validate that image is valid base64."""
        # Remove data URL prefix if present
        if v.startswith('data:image'):
            v = v.split(',')[1]
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 encoded image')
        return v


class ExtractedDocumentData(BaseModel):
    """Extracted data from ID document."""
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    id_number: Optional[str] = None
    expiry_date: Optional[str] = None
    nationality: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    raw_text: str = ""


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""
    session_id: str
    extracted_data: ExtractedDocumentData
    confidence: float = Field(..., ge=0.0, le=1.0)
    face_detected: bool
    document_valid: bool
    message: str = ""


class LivenessStartRequest(BaseModel):
    """Request to start liveness challenge."""
    session_id: str


class LivenessStartResponse(BaseModel):
    """Response with liveness challenge details."""
    challenge_id: str
    challenge_type: Literal["blink", "turn_left", "turn_right", "nod"]
    instructions: str
    timeout_seconds: int = 30


class LivenessVerifyRequest(BaseModel):
    """Request to verify liveness with captured frames."""
    session_id: str
    challenge_id: str
    frames: list[str] = Field(..., min_length=5, max_length=30, 
                               description="Base64 encoded video frames")
    
    @field_validator('frames')
    @classmethod
    def validate_frames(cls, v: list[str]) -> list[str]:
        """Validate all frames are valid base64."""
        validated = []
        for frame in v:
            if frame.startswith('data:image'):
                frame = frame.split(',')[1]
            try:
                base64.b64decode(frame)
                validated.append(frame)
            except Exception:
                raise ValueError('Invalid base64 encoded frame')
        return validated


class LivenessVerifyResponse(BaseModel):
    """Response for liveness verification."""
    passed: bool
    score: float = Field(..., ge=0.0, le=1.0)
    challenge_completed: bool
    details: dict = Field(default_factory=dict)
    message: str = ""


class FaceVerifyRequest(BaseModel):
    """Request for face verification (selfie vs ID)."""
    session_id: str
    selfie_image: str = Field(..., description="Base64 encoded selfie image")
    
    @field_validator('selfie_image')
    @classmethod
    def validate_selfie(cls, v: str) -> str:
        """Validate selfie is valid base64."""
        if v.startswith('data:image'):
            v = v.split(',')[1]
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 encoded selfie')
        return v


class FaceVerifyResponse(BaseModel):
    """Response for face verification."""
    verified: bool
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float
    kyc_status: Literal["pending", "approved", "rejected", "manual_review"]
    message: str = ""


class KYCStatusResponse(BaseModel):
    """Full KYC session status."""
    session_id: str
    status: Literal["pending", "document_uploaded", "liveness_passed", 
                    "verified", "rejected", "expired"]
    document_verified: bool = False
    liveness_verified: bool = False
    face_verified: bool = False
    ocr_confidence: Optional[float] = None
    liveness_score: Optional[float] = None
    face_similarity_score: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    extracted_data: Optional[ExtractedDocumentData] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    code: str


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    services: dict = Field(default_factory=dict)
