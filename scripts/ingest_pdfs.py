import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import settings
from backend.services.ocr_service import OCRService
from backend.services.chunking_service import ChunkingService
from backend.services.embedding_service import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ingest_pdfs():
    """Ingest NCERT PDFs - Windows compatible"""
    
    logger.info("Starting PDF ingestion...")
    logger.info(f"PDF Path: {settings.NCERT_PDF_PATH}")
    logger.info(f"Vector DB Path: {settings.VECTOR_DB_PATH}")
    
    # Initialize services
    ocr_service = OCRService(language=settings.OCR_LANGUAGE)
    chunking_service = ChunkingService(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
        vector_db_path=settings.VECTOR_DB_PATH
    )
    embedding_service.create_collection(settings.COLLECTION_NAME)
    
    # Find all PDF files (Windows-compatible path handling)
    pdf_path = Path(settings.NCERT_PDF_PATH)
    pdf_files = list(pdf_path.rglob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_path}")
        logger.error("Please place PDFs in backend\\data\\ncert_pdfs\\grade_X\\subject\\")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    total_chunks = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract text
            pages = ocr_service.extract_text_from_pdf(str(pdf_file))
            
            if not pages:
                logger.warning(f"No content extracted from {pdf_file.name}")
                continue
            
            # Extract metadata
            metadata = chunking_service.extract_metadata(pdf_file.name)
            
            # Parse grade from directory structure
            parts = pdf_file.parts
            for part in parts:
                if 'grade' in part.lower():
                    try:
                        grade_num = int(''.join(filter(str.isdigit, part)))
                        metadata["grade"] = grade_num
                        break
                    except:
                        pass
            
            # Parse subject from directory
            if pdf_file.parent.name in ["math", "science", "social_science", "english", "hindi"]:
                metadata["subject"] = pdf_file.parent.name
            
            logger.info(f"Metadata: Grade={metadata.get('grade')}, Subject={metadata.get('subject')}")
            
            # Create chunks
            chunks = chunking_service.create_chunks(pages, metadata)
            chunks = chunking_service.filter_chunks(chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
            
            if chunks:
                # Generate embeddings
                chunks = embedding_service.embed_chunks(chunks)
                
                # Store in vector database
                embedding_service.store_chunks(chunks)
                
                total_chunks += len(chunks)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"✅ Ingestion complete! Total chunks stored: {total_chunks}")

if __name__ == "__main__":
    ingest_pdfs()