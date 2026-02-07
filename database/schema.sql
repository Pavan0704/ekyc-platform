-- E-KYC Platform Database Schema
-- PostgreSQL schema for storing KYC verification data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table (basic user info)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(50),
    full_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_verified BOOLEAN DEFAULT FALSE,
    verification_count INTEGER DEFAULT 0
);

-- KYC Sessions table (main verification records)
CREATE TABLE kyc_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Document information
    document_type VARCHAR(50) NOT NULL,
    document_country VARCHAR(3),
    
    -- Extracted data (encrypted JSON)
    extracted_data_encrypted BYTEA,
    
    -- Verification scores
    ocr_confidence FLOAT,
    liveness_score FLOAT,
    face_similarity_score FLOAT,
    
    -- Status tracking
    status VARCHAR(30) DEFAULT 'pending' CHECK (
        status IN ('pending', 'document_uploaded', 'liveness_passed', 
                   'verified', 'rejected', 'expired', 'manual_review')
    ),
    
    -- Challenge tracking
    liveness_challenge_type VARCHAR(30),
    liveness_attempts INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    document_uploaded_at TIMESTAMP WITH TIME ZONE,
    liveness_verified_at TIMESTAMP WITH TIME ZONE,
    face_verified_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 hour'),
    
    -- Metadata
    ip_address INET,
    user_agent TEXT,
    device_fingerprint VARCHAR(255)
);

-- Verification results (detailed breakdown)
CREATE TABLE verification_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES kyc_sessions(id) ON DELETE CASCADE,
    
    -- Verification type
    verification_type VARCHAR(30) NOT NULL CHECK (
        verification_type IN ('ocr', 'liveness', 'face_match', 'document_validity')
    ),
    
    -- Results
    passed BOOLEAN NOT NULL,
    score FLOAT,
    threshold_used FLOAT,
    
    -- Details (JSON)
    details JSONB,
    
    -- Error tracking
    error_message TEXT,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit logs (for security and compliance)
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES kyc_sessions(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Action details
    action VARCHAR(100) NOT NULL,
    action_category VARCHAR(50) NOT NULL CHECK (
        action_category IN ('document', 'liveness', 'verification', 'security', 'admin')
    ),
    
    -- Actor information
    actor_type VARCHAR(30) CHECK (actor_type IN ('user', 'system', 'admin')),
    actor_id VARCHAR(255),
    
    -- Request details (sanitized - NO PII)
    ip_address INET,
    request_path VARCHAR(255),
    request_method VARCHAR(10),
    
    -- Response
    response_status INTEGER,
    success BOOLEAN,
    
    -- Additional metadata
    metadata JSONB,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Encrypted face embeddings (for re-verification)
-- NOTE: Embeddings are encrypted at rest for security
CREATE TABLE face_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES kyc_sessions(id) ON DELETE CASCADE,
    
    -- Embedding type
    embedding_type VARCHAR(20) CHECK (embedding_type IN ('document', 'selfie')),
    
    -- Encrypted embedding (512-dimensional FaceNet vector)
    embedding_encrypted BYTEA NOT NULL,
    
    -- Metadata
    quality_score FLOAT,
    face_detected_confidence FLOAT,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_kyc_sessions_user_id ON kyc_sessions(user_id);
CREATE INDEX idx_kyc_sessions_status ON kyc_sessions(status);
CREATE INDEX idx_kyc_sessions_created_at ON kyc_sessions(created_at);
CREATE INDEX idx_kyc_sessions_expires_at ON kyc_sessions(expires_at);

CREATE INDEX idx_verification_results_session ON verification_results(session_id);
CREATE INDEX idx_verification_results_type ON verification_results(verification_type);

CREATE INDEX idx_audit_logs_session ON audit_logs(session_id);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);

CREATE INDEX idx_face_embeddings_session ON face_embeddings(session_id);

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for users table
CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to clean expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    UPDATE kyc_sessions 
    SET status = 'expired' 
    WHERE expires_at < NOW() 
    AND status NOT IN ('verified', 'rejected', 'expired');
END;
$$ LANGUAGE plpgsql;

-- View for KYC statistics (admin dashboard)
CREATE VIEW kyc_statistics AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as total_sessions,
    COUNT(*) FILTER (WHERE status = 'verified') as verified_count,
    COUNT(*) FILTER (WHERE status = 'rejected') as rejected_count,
    COUNT(*) FILTER (WHERE status = 'manual_review') as manual_review_count,
    AVG(ocr_confidence) FILTER (WHERE ocr_confidence IS NOT NULL) as avg_ocr_confidence,
    AVG(liveness_score) FILTER (WHERE liveness_score IS NOT NULL) as avg_liveness_score,
    AVG(face_similarity_score) FILTER (WHERE face_similarity_score IS NOT NULL) as avg_face_similarity
FROM kyc_sessions
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date DESC;

-- Comments for documentation
COMMENT ON TABLE users IS 'Basic user information for KYC verification';
COMMENT ON TABLE kyc_sessions IS 'Main KYC verification session records';
COMMENT ON TABLE verification_results IS 'Detailed breakdown of each verification step';
COMMENT ON TABLE audit_logs IS 'Security and compliance audit trail';
COMMENT ON TABLE face_embeddings IS 'Encrypted face embeddings for verification';
COMMENT ON COLUMN kyc_sessions.extracted_data_encrypted IS 'AES-256 encrypted JSON containing PII';
COMMENT ON COLUMN face_embeddings.embedding_encrypted IS 'AES-256 encrypted 512-dim FaceNet vector';
