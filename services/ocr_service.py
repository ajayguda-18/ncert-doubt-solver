import pytesseract
from pdf2image import convert_from_path
import pdfplumber
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app.config import settings

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, language: str = "eng+hin+urd"):
        self.language = language
        self.poppler_path = settings.POPPLER_PATH
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(thresh)
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF using both text extraction and OCR"""
        pages = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:
                        pages.append({
                            "page_number": page_num,
                            "text": text,
                            "method": "text_extraction"
                        })
                    else:
                        logger.info(f"Using OCR for page {page_num}")
                        ocr_text = self.ocr_page(pdf_path, page_num)
                        pages.append({
                            "page_number": page_num,
                            "text": ocr_text,
                            "method": "ocr"
                        })
        except Exception as e:
            logger.error(f"Error with pdfplumber: {e}, falling back to OCR")
            pages = self.ocr_full_pdf(pdf_path)
        
        return pages
    
    def ocr_page(self, pdf_path: str, page_number: int) -> str:
        """OCR a specific page - Windows compatible"""
        try:
            images = convert_from_path(
                pdf_path, 
                first_page=page_number, 
                last_page=page_number,
                dpi=300,
                poppler_path=r"C:\poppler\bin"  # Windows: specify poppler path
            )
            
            if images:
                preprocessed = self.preprocess_image(images[0])
                text = pytesseract.image_to_string(
                    preprocessed, 
                    lang=self.language,
                    config='--psm 6'
                )
                return text
        except Exception as e:
            logger.error(f"OCR error on page {page_number}: {e}")
        
        return ""
    
    def ocr_full_pdf(self, pdf_path: str) -> List[Dict]:
        """OCR entire PDF - Windows compatible"""
        try:
            images = convert_from_path(
                pdf_path, 
                dpi=300,
                poppler_path=r"C:\poppler\bin")
                
            
            
            pages = []
            for page_num, image in enumerate(images, 1):
                preprocessed = self.preprocess_image(image)
                text = pytesseract.image_to_string(
                    preprocessed, 
                    lang=self.language,
                    config='--psm 6'
                )
                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "method": "ocr"
                })
            
            return pages
        except Exception as e:
            logger.error(f"Full PDF OCR error: {e}")
            return []