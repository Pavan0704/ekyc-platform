"""
KYC API Routes.
Handles document upload, liveness verification, and face matching.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict
import uuid
from datetime import datetime

from models.schemas import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    ExtractedDocumentData,
    LivenessStartRequest,
    LivenessStartResponse,
    LivenessVerifyRequest,
    LivenessVerifyResponse,
    FaceVerifyRequest,
    FaceVerifyResponse,
    KYCStatusResponse,
    ErrorResponse
)
from services import ocr_engine, face_detector, liveness_detector, face_verifier, ChallengeType
from utils import (
    decode_base64_image,
    preprocess_for_ocr,
    preprocess_for_face_detection,
    resize_image,
    validate_image_quality
)

router = APIRouter(prefix="/api/kyc", tags=["KYC"])

# In-memory session storage (use Redis/PostgreSQL in production)
_sessions: Dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    """Get session or raise 404."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return _sessions[session_id]


@router.post(
    "/document",
    response_model=DocumentUploadResponse,
    responses={400: {"model": ErrorResponse}}
)
async def upload_document(request: DocumentUploadRequest):
    """
    Upload and process ID document.
    
    Extracts text using OCR and detects face in the document.
    Returns extracted data and creates a new KYC session.
    """
    try:
        # Decode image (in-memory processing)
        image = decode_base64_image(request.image)
        
        # Validate image quality
        quality = validate_image_quality(image)
        if not quality["overall_ok"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image quality issues: {quality}"
            )
        
        # Resize for processing
        image = resize_image(image, max_dimension=1024)
        
        # Preprocess for OCR
        ocr_image = preprocess_for_ocr(image)
        
        # Extract text
        ocr_result = ocr_engine.parse_id_document(
            ocr_image, 
            document_type=request.document_type
        )
        
        # Detect face in document (for later comparison)
        rgb_image = preprocess_for_face_detection(image)
        face_result = face_detector.detect_face(rgb_image)
        
        # Extract face embedding if detected
        id_face_embedding = None
        if face_result.detected:
            id_face_embedding = face_verifier.get_embedding(rgb_image)
        
        # Create session
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "id": session_id,
            "status": "document_uploaded",
            "document_type": request.document_type,
            "extracted_data": {
                "full_name": ocr_result.full_name,
                "date_of_birth": ocr_result.date_of_birth,
                "id_number": ocr_result.id_number,
                "expiry_date": ocr_result.expiry_date,
                "nationality": ocr_result.nationality,
                "gender": ocr_result.gender,
                "address": ocr_result.address,
                "raw_text": ocr_result.raw_text
            },
            "ocr_confidence": ocr_result.confidence,
            "face_detected": face_result.detected,
            "id_face_embedding": id_face_embedding.tolist() if id_face_embedding is not None else None,
            "liveness_verified": False,
            "face_verified": False,
            "created_at": datetime.now().isoformat()
        }
        
        return DocumentUploadResponse(
            session_id=session_id,
            extracted_data=ExtractedDocumentData(
                full_name=ocr_result.full_name,
                date_of_birth=ocr_result.date_of_birth,
                id_number=ocr_result.id_number,
                expiry_date=ocr_result.expiry_date,
                nationality=ocr_result.nationality,
                gender=ocr_result.gender,
                address=ocr_result.address,
                raw_text=ocr_result.raw_text
            ),
            confidence=ocr_result.confidence,
            face_detected=face_result.detected,
            document_valid=bool(ocr_result.id_number or ocr_result.full_name),
            message="Document processed successfully" if face_result.detected 
                    else "Document processed but no face detected"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.post(
    "/liveness/start",
    response_model=LivenessStartResponse
)
async def start_liveness_challenge(request: LivenessStartRequest):
    """
    Start a liveness challenge for the session.
    
    Returns challenge type and instructions for the user.
    """
    session = get_session(request.session_id)
    
    # Create blink challenge (most reliable)
    challenge = liveness_detector.create_challenge(ChallengeType.BLINK)
    
    # Store challenge ID in session
    session["current_challenge_id"] = challenge.challenge_id
    
    return LivenessStartResponse(
        challenge_id=challenge.challenge_id,
        challenge_type=challenge.challenge_type.value,
        instructions=challenge.instructions,
        timeout_seconds=30
    )


@router.post(
    "/liveness/verify",
    response_model=LivenessVerifyResponse,
    responses={400: {"model": ErrorResponse}}
)
async def verify_liveness(request: LivenessVerifyRequest):
    """
    Verify liveness using captured frames.
    
    Analyzes frames for blink detection or head movement based on challenge type.
    """
    session = get_session(request.session_id)
    
    if session.get("current_challenge_id") != request.challenge_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Challenge ID mismatch or expired"
        )
    
    try:
        # Process frames
        eye_data = []
        pose_data = []
        image_size = None
        
        for frame_b64 in request.frames:
            # Decode frame
            frame = decode_base64_image(frame_b64)
            frame_rgb = preprocess_for_face_detection(frame)
            
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
            
            # Get eye landmarks for EAR
            eye_landmarks = face_detector.get_eye_landmarks(frame_rgb)
            if eye_landmarks:
                left_ear = liveness_detector.calculate_ear(eye_landmarks['left_eye'])
                right_ear = liveness_detector.calculate_ear(eye_landmarks['right_eye'])
                avg_ear = (left_ear + right_ear) / 2
                eye_data.append({"left_ear": left_ear, "right_ear": right_ear, "avg_ear": avg_ear})
            
            # Get head pose
            pose_landmarks = face_detector.get_head_pose_landmarks(frame_rgb)
            if pose_landmarks:
                pose = liveness_detector.estimate_head_pose(pose_landmarks, image_size)
                pose_data.append(pose)
        
        # Verify liveness
        result = liveness_detector.verify_liveness(
            request.challenge_id,
            eye_data,
            pose_data,
            image_size or (640, 480)
        )
        
        # Update session
        if result.passed:
            session["liveness_verified"] = True
            session["liveness_score"] = result.score
            session["status"] = "liveness_passed"
        
        # Cleanup challenge
        liveness_detector.cleanup_challenge(request.challenge_id)
        
        return LivenessVerifyResponse(
            passed=result.passed,
            score=result.score,
            challenge_completed=result.challenge_completed,
            details=result.details,
            message=result.message
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Liveness verification failed: {str(e)}"
        )


@router.post(
    "/verify",
    response_model=FaceVerifyResponse,
    responses={400: {"model": ErrorResponse}}
)
async def verify_face(request: FaceVerifyRequest):
    """
    Verify selfie matches the ID document photo.
    
    Compares face embeddings using FaceNet and returns similarity score.
    """
    session = get_session(request.session_id)
    
    # Check prerequisites
    if not session.get("face_detected"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in ID document. Please re-upload."
        )
    
    if not session.get("liveness_verified"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Liveness verification required before face matching."
        )
    
    try:
        # Decode selfie
        selfie = decode_base64_image(request.selfie_image)
        selfie_rgb = preprocess_for_face_detection(selfie)
        
        # Check selfie quality
        quality = face_verifier.get_quality_score(selfie_rgb)
        if not quality["face_detected"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in selfie"
            )
        
        # Get stored ID embedding
        id_embedding = session.get("id_face_embedding")
        if id_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID face embedding not found. Please re-upload document."
            )
        
        import numpy as np
        id_embedding = np.array(id_embedding)
        
        # Get selfie embedding
        selfie_embedding = face_verifier.get_embedding(selfie_rgb)
        if selfie_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract face from selfie"
            )
        
        # Verify
        result = face_verifier.verify_embeddings(
            id_embedding,
            selfie_embedding,
            security_level='medium'
        )
        
        # Update session
        session["face_verified"] = result.verified
        session["face_similarity_score"] = result.similarity_score
        session["status"] = "verified" if result.verified else "rejected"
        session["completed_at"] = datetime.now().isoformat()
        
        # Determine KYC status
        if result.verified and session.get("liveness_verified"):
            kyc_status = "approved"
        elif result.similarity_score >= 0.5:
            kyc_status = "manual_review"
        else:
            kyc_status = "rejected"
        
        session["kyc_status"] = kyc_status
        
        return FaceVerifyResponse(
            verified=result.verified,
            similarity_score=result.similarity_score,
            threshold_used=result.threshold_used,
            kyc_status=kyc_status,
            message=result.message
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face verification failed: {str(e)}"
        )


@router.get(
    "/status/{session_id}",
    response_model=KYCStatusResponse
)
async def get_kyc_status(session_id: str):
    """
    Get the current status of a KYC session.
    """
    session = get_session(session_id)
    
    return KYCStatusResponse(
        session_id=session["id"],
        status=session.get("status", "pending"),
        document_verified=bool(session.get("extracted_data", {}).get("id_number")),
        liveness_verified=session.get("liveness_verified", False),
        face_verified=session.get("face_verified", False),
        ocr_confidence=session.get("ocr_confidence"),
        liveness_score=session.get("liveness_score"),
        face_similarity_score=session.get("face_similarity_score"),
        created_at=datetime.fromisoformat(session["created_at"]),
        completed_at=datetime.fromisoformat(session["completed_at"]) if session.get("completed_at") else None,
        extracted_data=ExtractedDocumentData(**session.get("extracted_data", {})) if session.get("extracted_data") else None
    )
