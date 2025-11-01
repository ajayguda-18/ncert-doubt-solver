# backend/services/chunking_service.py
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import hashlib

class ChunkingService:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
    
    def extract_metadata(self, filename: str) -> Dict:
        """Extract metadata from filename"""
        # Expected format: grade_X_subject_language.pdf
        parts = filename.replace(".pdf", "").split("_")
        
        metadata = {
            "source_file": filename,
            "grade": None,
            "subject": None,
            "language": None,
        }
        
        # Parse grade
        for part in parts:
            if part.startswith("grade") or part.isdigit():
                try:
                    metadata["grade"] = int(re.search(r'\d+', part).group())
                except:
                    pass
        
        # Common subjects
        subjects = ["math", "science", "social", "english", "hindi"]
        for subject in subjects:
            if subject in filename.lower():
                metadata["subject"] = subject
                break
        
        # Languages
        languages = ["english", "hindi", "urdu"]
        for lang in languages:
            if lang in filename.lower():
                metadata["language"] = lang
                break
        
        return metadata
    
    def create_chunks(
        self, 
        pages: List[Dict], 
        metadata: Dict
    ) -> List[Dict]:
        """Create chunks with metadata"""
        chunks = []
        
        for page in pages:
            text = page["text"]
            page_number = page["page_number"]
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            for idx, chunk_text in enumerate(text_chunks):
                # Generate unique chunk ID
                chunk_id = hashlib.md5(
                    f"{metadata['source_file']}-{page_number}-{idx}".encode()
                ).hexdigest()
                
                chunk = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_index": idx,
                    **metadata
                }
                chunks.append(chunk)
        
        return chunks
    
    def filter_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter out low-quality chunks"""
        filtered = []
        
        for chunk in chunks:
            text = chunk["text"].strip()
            
            # Skip empty or very short chunks
            if len(text) < 50:
                continue
            
            # Skip chunks with too many special characters (likely noise)
            special_char_ratio = sum(
                not c.isalnum() and not c.isspace() for c in text
            ) / len(text)
            
            if special_char_ratio > 0.5:
                continue
            
            filtered.append(chunk)
        
        return filtered