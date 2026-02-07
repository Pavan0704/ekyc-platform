"""
OCR Engine for ID Document Text Extraction.
Uses EasyOCR with preprocessing for high accuracy on identity documents.
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

# EasyOCR is imported lazily to allow the server to start quickly
_reader = None


def get_ocr_reader():
    """Lazy initialization of EasyOCR reader."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if CUDA available
    return _reader


@dataclass
class OCRResult:
    """Structured OCR extraction result."""
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    id_number: Optional[str] = None
    expiry_date: Optional[str] = None
    nationality: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    raw_text: str = ""
    confidence: float = 0.0


class OCREngine:
    """
    OCR Engine for extracting text from ID documents.
    Supports national IDs, passports, and driver's licenses.
    """
    
    # Patterns for common ID document fields
    DATE_PATTERNS = [
        r'\d{2}[/\-\.]\d{2}[/\-\.]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
        r'\d{4}[/\-\.]\d{2}[/\-\.]\d{2}',  # YYYY/MM/DD
        r'\d{2}\s+\w+\s+\d{4}',             # DD Month YYYY
    ]
    
    ID_NUMBER_PATTERNS = [
        r'[A-Z]{1,3}\d{6,12}',              # Passport format
        r'\d{9,15}',                         # National ID numbers
        r'[A-Z0-9]{8,20}',                   # Generic ID format
    ]
    
    NAME_KEYWORDS = ['name', 'surname', 'given', 'first', 'last', 'full']
    DOB_KEYWORDS = ['birth', 'dob', 'born', 'date of birth', 'birthday']
    EXPIRY_KEYWORDS = ['expiry', 'expires', 'exp', 'valid until', 'validity']
    ID_KEYWORDS = ['id', 'number', 'no.', 'no:', 'document', 'passport']
    
    def __init__(self):
        self.reader = None
    
    def _ensure_reader(self):
        """Ensure OCR reader is initialized."""
        if self.reader is None:
            self.reader = get_ocr_reader()
    
    def extract_text(self, image: np.ndarray) -> Tuple[List[dict], float]:
        """
        Extract raw text from image using EasyOCR.
        
        Args:
            image: Preprocessed image array (grayscale or RGB)
            
        Returns:
            Tuple of (list of detection results, average confidence)
        """
        self._ensure_reader()
        
        # Run OCR
        results = self.reader.readtext(image)
        
        if not results:
            return [], 0.0
        
        # Calculate average confidence
        confidences = [r[2] for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Structure results
        detections = []
        for bbox, text, conf in results:
            detections.append({
                "text": text,
                "confidence": conf,
                "bbox": bbox
            })
        
        return detections, avg_confidence
    
    def parse_id_document(
        self, 
        image: np.ndarray,
        document_type: str = "national_id"
    ) -> OCRResult:
        """
        Extract and parse ID document fields.
        
        Args:
            image: Preprocessed image array
            document_type: Type of document (national_id, passport, drivers_license)
            
        Returns:
            OCRResult with extracted fields
        """
        detections, confidence = self.extract_text(image)
        
        if not detections:
            return OCRResult(confidence=0.0)
        
        # Combine all text
        raw_text = "\n".join([d["text"] for d in detections])
        
        # Extract structured fields
        result = OCRResult(
            raw_text=raw_text,
            confidence=confidence
        )
        
        # Extract each field type
        result.full_name = self._extract_name(detections, raw_text)
        result.date_of_birth = self._extract_date(detections, raw_text, self.DOB_KEYWORDS)
        result.expiry_date = self._extract_date(detections, raw_text, self.EXPIRY_KEYWORDS)
        result.id_number = self._extract_id_number(detections, raw_text, document_type)
        result.gender = self._extract_gender(raw_text)
        result.nationality = self._extract_nationality(raw_text)
        
        return result
    
    def _extract_name(self, detections: List[dict], raw_text: str) -> Optional[str]:
        """Extract full name from OCR results."""
        text_lower = raw_text.lower()
        
        # Look for name field indicators
        for keyword in self.NAME_KEYWORDS:
            if keyword in text_lower:
                # Find the line containing the keyword and get the value
                lines = raw_text.split('\n')
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        # Check if value is on same line (after colon/space)
                        if ':' in line:
                            name = line.split(':', 1)[1].strip()
                            if name and len(name) > 2:
                                return self._clean_name(name)
                        # Check next line
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line and len(next_line) > 2:
                                return self._clean_name(next_line)
        
        # Fallback: Look for capitalized word sequences (likely names)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        matches = re.findall(name_pattern, raw_text)
        if matches:
            # Return the longest match (most likely full name)
            return max(matches, key=len)
        
        return None
    
    def _clean_name(self, name: str) -> str:
        """Clean extracted name string."""
        # Remove common prefixes/suffixes
        name = re.sub(r'^(mr|mrs|ms|dr|miss)\.?\s*', '', name, flags=re.IGNORECASE)
        # Remove non-letter characters except spaces
        name = re.sub(r'[^a-zA-Z\s]', '', name)
        # Normalize whitespace
        name = ' '.join(name.split())
        return name.title()
    
    def _extract_date(
        self, 
        detections: List[dict], 
        raw_text: str,
        keywords: List[str]
    ) -> Optional[str]:
        """Extract date field based on keywords."""
        text_lower = raw_text.lower()
        
        # Find dates in the text
        all_dates = []
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            all_dates.extend(matches)
        
        if not all_dates:
            return None
        
        # Try to find date near keyword
        lines = raw_text.split('\n')
        for keyword in keywords:
            for i, line in enumerate(lines):
                if keyword in line.lower():
                    # Check same line
                    for date in all_dates:
                        if date in line:
                            return date
                    # Check next line
                    if i + 1 < len(lines):
                        for date in all_dates:
                            if date in lines[i + 1]:
                                return date
        
        # Return first date if no keyword match
        return all_dates[0] if all_dates else None
    
    def _extract_id_number(
        self, 
        detections: List[dict], 
        raw_text: str,
        document_type: str
    ) -> Optional[str]:
        """Extract ID number based on document type."""
        # Find all potential ID numbers
        candidates = []
        
        for pattern in self.ID_NUMBER_PATTERNS:
            matches = re.findall(pattern, raw_text)
            candidates.extend(matches)
        
        if not candidates:
            return None
        
        # Filter by document type expectations
        if document_type == "passport":
            # Passports usually have alphanumeric IDs starting with letters
            passport_ids = [c for c in candidates if re.match(r'^[A-Z]{1,2}\d+', c)]
            if passport_ids:
                return passport_ids[0]
        
        # Return the longest numeric/alphanumeric sequence
        return max(candidates, key=len) if candidates else None
    
    def _extract_gender(self, raw_text: str) -> Optional[str]:
        """Extract gender from text."""
        text_lower = raw_text.lower()
        
        if 'female' in text_lower or ' f ' in text_lower or '/f/' in text_lower:
            return 'Female'
        elif 'male' in text_lower or ' m ' in text_lower or '/m/' in text_lower:
            return 'Male'
        return None
    
    def _extract_nationality(self, raw_text: str) -> Optional[str]:
        """Extract nationality from text."""
        # Common nationality keywords
        nationality_keywords = ['nationality', 'citizen', 'country']
        
        lines = raw_text.split('\n')
        for keyword in nationality_keywords:
            for i, line in enumerate(lines):
                if keyword in line.lower():
                    # Check for value after colon
                    if ':' in line:
                        nationality = line.split(':', 1)[1].strip()
                        if nationality:
                            return nationality
                    # Check next line
                    if i + 1 < len(lines):
                        return lines[i + 1].strip()
        
        return None


# Singleton instance
ocr_engine = OCREngine()
