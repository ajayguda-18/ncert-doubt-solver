from groq import Groq
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RAGServiceGroq:
    def __init__(
        self, 
        groq_api_key: str,
        groq_model: str,
        embedding_service,
        language_service,
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        self.embedding_service = embedding_service
        self.language_service = language_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.groq_model = groq_model
        
        # Initialize Groq client
        logger.info(f"Initializing Groq with model: {groq_model}")
        self.client = Groq(api_key=groq_api_key)
        logger.info("✅ Groq client initialized successfully")
    
    def retrieve_context(
        self, 
        query: str, 
        grade: Optional[int] = None,
        subject: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant chunks from vector database (simplified version)"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.model.encode(
                query,
                convert_to_numpy=True
            ).tolist()
            
            # Query without filters for now (will filter after retrieval)
            logger.info(f"Querying vector database for top {top_k} results...")
            
            results = self.embedding_service.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2  # Get more results to filter later
            )
            
            # Format results
            contexts = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    
                    # Filter by grade and subject in post-processing
                    if grade and metadata.get("grade") != str(grade):
                        continue
                    if subject and metadata.get("subject") != subject:
                        continue
                    
                    context = {
                        "text": doc,
                        "metadata": metadata,
                        "distance": results["distances"][0][i] if "distances" in results else None
                    }
                    contexts.append(context)
                    
                    # Stop when we have enough
                    if len(contexts) >= top_k:
                        break
            
            logger.info(f"Retrieved {len(contexts)} contexts")
            return contexts
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def check_relevance(self, contexts: List[Dict], threshold: float = 0.7) -> bool:
        """Check if retrieved contexts are relevant enough"""
        if not contexts:
            return False
        
        if contexts[0].get("distance"):
            similarity = 1 - contexts[0]["distance"]
            return similarity >= threshold
        
        return True
    
    def get_language_prompts(self, language: str) -> Dict[str, str]:
        """Get language-specific prompts with fallback"""
        # Default prompts
        default_prompts = {
            "system": "You are a helpful teacher assistant for NCERT textbooks.",
            "citation": "Based on the following context from NCERT textbooks:",
            "no_answer": "I don't have enough information in the NCERT textbooks to answer this question."
        }
        
        # Try to get from language service
        if self.language_service:
            try:
                return self.language_service.get_language_prompts(language)
            except Exception as e:
                logger.warning(f"Could not get language prompts: {e}")
        
        return default_prompts
    
    def format_prompt(self, query: str, contexts: List[Dict], language: str) -> str:
        """Format prompt for Groq"""
        prompts = self.get_language_prompts(language)
        
        # Build context string with citations
        context_str = ""
        for i, ctx in enumerate(contexts[:5], 1):
            metadata = ctx["metadata"]
            source = f"Grade {metadata.get('grade', 'N/A')}, {metadata.get('subject', 'N/A')}, Page {metadata.get('page_number', 'N/A')}"
            context_str += f"\n[Source {i}: {source}]\n{ctx['text'][:500]}\n"
        
        prompt = f"""{prompts['system']}

{prompts['citation']}
{context_str}

Student Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using [Source X] notation
3. If the context doesn't contain the answer, say: "{prompts['no_answer']}"
4. Explain step-by-step for math problems
5. Use simple language appropriate for students
6. Be concise but complete

Answer:"""
        
        return prompt
    
    def detect_language(self, text: str) -> str:
        """Detect language with fallback"""
        if self.language_service:
            try:
                return self.language_service.detect_language(text)
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        return "english"
    
    def generate_answer(
        self, 
        query: str,
        grade: Optional[int] = None,
        subject: Optional[str] = None,
        language: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """Generate answer using Groq API"""
        try:
            # Detect language if not provided
            if not language:
                language = self.detect_language(query)
            
            logger.info(f"Processing query for Grade {grade}, Subject: {subject}, Language: {language}")
            
            # Retrieve relevant context
            contexts = self.retrieve_context(query, grade, subject, top_k=5)
            
            logger.info(f"Retrieved {len(contexts)} contexts")
            
            # Check relevance
            if not self.check_relevance(contexts):
                prompts = self.get_language_prompts(language)
                return {
                    "answer": prompts["no_answer"],
                    "contexts": [],
                    "language": language,
                    "confidence": "low"
                }
            
            # Format prompt
            prompt = self.format_prompt(query, contexts, language)
            
            # Call Groq API
            logger.info(f"Calling Groq API with model: {self.groq_model}")
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.groq_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            answer = chat_completion.choices[0].message.content
            
            logger.info(f"✅ Got response from Groq (length: {len(answer)} chars)")
            
            return {
                "answer": answer,
                "contexts": contexts[:5],  # Return top 5 for citations
                "language": language,
                "confidence": "high" if contexts else "low"
            }
        
        except Exception as e:
            logger.error(f"❌ Error generating answer with Groq: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": f"Error: {str(e)}",
                "contexts": [],
                "language": language or "english",
                "confidence": "low"
            }