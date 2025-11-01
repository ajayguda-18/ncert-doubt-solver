from langdetect import detect, LangDetectException
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class LanguageService:
    def __init__(self):
        # Map langdetect codes to our language names
        self.language_map = {
            "en": "english",
            "hi": "hindi",
            "ur": "urdu",
            "mr": "marathi",
            "ta": "tamil",
            "te": "telugu",
            "bn": "bengali",
            "gu": "gujarati",
            "kn": "kannada",
            "ml": "malayalam"
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            detected_code = detect(text)
            language = self.language_map.get(detected_code, "english")
            logger.info(f"Detected language: {language}")
            return language
        except LangDetectException:
            logger.warning("Language detection failed, defaulting to English")
            return "english"
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return "english"
    
    def get_language_prompts(self, language: str) -> Dict[str, str]:
        """Get language-specific prompts"""
        prompts = {
            "english": {
                "system": "You are a helpful teacher assistant for NCERT textbooks.",
                "citation": "Based on the following context from NCERT textbooks:",
                "no_answer": "I don't have enough information in the NCERT textbooks to answer this question."
            },
            "hindi": {
                "system": "आप NCERT पाठ्यपुस्तकों के लिए एक सहायक शिक्षक सहायक हैं।",
                "citation": "NCERT पाठ्यपुस्तकों से निम्नलिखित संदर्भ के आधार पर:",
                "no_answer": "मेरे पास इस प्रश्न का उत्तर देने के लिए NCERT पाठ्यपुस्तकों में पर्याप्त जानकारी नहीं है।"
            },
            "urdu": {
                "system": "آپ NCERT کتابوں کے لیے ایک مددگار استاد معاون ہیں۔",
                "citation": "NCERT کتابوں سے درج ذیل سیاق و سباق کی بنیاد پر:",
                "no_answer": "میرے پاس NCERT کتابوں میں اس سوال کا جواب دینے کے لیے کافی معلومات نہیں ہیں۔"
            }
        }
        
        return prompts.get(language, prompts["english"])