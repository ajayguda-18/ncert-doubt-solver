# backend/opea/microservices.py
"""
OPEA-based microservices architecture
Each component is a separate microservice that can be scaled independently
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class OPEAMicroservice:
    """Base class for OPEA microservices"""
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized microservice: {name}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class EmbeddingMicroservice(OPEAMicroservice):
    """Microservice for text embedding"""
    
    def __init__(self, embedding_service):
        super().__init__("embedding")
        self.embedding_service = embedding_service
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data.get("text")
        embedding = self.embedding_service.model.encode(text)
        return {"embedding": embedding.tolist()}

class RetrievalMicroservice(OPEAMicroservice):
    """Microservice for document retrieval"""
    
    def __init__(self, embedding_service):
        super().__init__("retrieval")
        self.embedding_service = embedding_service
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query")
        grade = input_data.get("grade")
        subject = input_data.get("subject")
        top_k = input_data.get("top_k", 5)
        
        # Generate embedding
        query_embedding = self.embedding_service.model.encode(query).tolist()
        
        # Retrieve documents
        where_filter = {}
        if grade:
            where_filter["grade"] = str(grade)
        if subject:
            where_filter["subject"] = subject
        
        results = self.embedding_service.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        
        return {"results": results}

class RerankingMicroservice(OPEAMicroservice):
    """Microservice for result reranking"""
    
    def __init__(self):
        super().__init__("reranking")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        results = input_data.get("results")
        query = input_data.get("query")
        
        # Simple reranking based on distance scores
        # In production, use a cross-encoder model for better reranking
        
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        # Combine and sort
        combined = list(zip(documents, distances, metadatas))
        combined.sort(key=lambda x: x[1])  # Sort by distance (lower is better)
        
        reranked = {
            "documents": [[item[0] for item in combined]],
            "distances": [[item[1] for item in combined]],
            "metadatas": [[item[2] for item in combined]]
        }
        
        return {"reranked_results": reranked}

class LLMMicroservice(OPEAMicroservice):
    """Microservice for LLM generation"""
    
    def __init__(self, llm_generator):
        super().__init__("llm")
        self.generator = llm_generator
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt")
        
        response = self.generator(prompt)[0]["generated_text"]
        answer = response.split("Answer:")[-1].strip()
        
        return {"answer": answer}

# backend/opea/megaservice.py
"""
OPEA Megaservice - Orchestrates multiple microservices
"""

class OPEAMegaservice:
    """Orchestrates the RAG pipeline using OPEA microservices"""
    
    def __init__(
        self,
        embedding_microservice,
        retrieval_microservice,
        reranking_microservice,
        llm_microservice,
        language_service
    ):
        self.embedding_ms = embedding_microservice
        self.retrieval_ms = retrieval_microservice
        self.reranking_ms = reranking_microservice
        self.llm_ms = llm_microservice
        self.language_service = language_service
        
        logger.info("OPEA Megaservice initialized")
    
    async def process_query(
        self,
        query: str,
        grade: int = None,
        subject: str = None,
        language: str = None
    ) -> Dict[str, Any]:
        """Process query through the RAG pipeline"""
        
        # Step 1: Language detection
        if not language:
            language = self.language_service.detect_language(query)
        
        # Step 2: Retrieval
        retrieval_input = {
            "query": query,
            "grade": grade,
            "subject": subject,
            "top_k": 10
        }
        retrieval_output = await self.retrieval_ms.process(retrieval_input)
        
        # Step 3: Reranking
        reranking_input = {
            "results": retrieval_output["results"],
            "query": query
        }
        reranking_output = await self.reranking_ms.process(reranking_input)
        
        # Step 4: Format prompt
        contexts = self._format_contexts(reranking_output["reranked_results"])
        prompt = self._build_prompt(query, contexts, language)
        
        # Step 5: LLM generation
        llm_input = {"prompt": prompt}
        llm_output = await self.llm_ms.process(llm_input)
        
        return {
            "answer": llm_output["answer"],
            "contexts": contexts[:5],  # Top 5 for citations
            "language": language
        }
    
    def _format_contexts(self, results: Dict) -> list:
        """Format retrieval results into context list"""
        contexts = []
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            contexts.append({
                "text": doc,
                "metadata": metadata,
                "distance": distance
            })
        
        return contexts
    
    def _build_prompt(self, query: str, contexts: list, language: str) -> str:
        """Build prompt with context"""
        prompts = self.language_service.get_language_prompts(language)
        
        context_str = ""
        for i, ctx in enumerate(contexts[:5], 1):
            metadata = ctx["metadata"]
            source = f"Grade {metadata.get('grade', 'N/A')}, {metadata.get('subject', 'N/A')}, Page {metadata.get('page_number', 'N/A')}"
            context_str += f"\n[Source {i}: {source}]\n{ctx['text']}\n"
        
        prompt = f"""{prompts['system']}

{prompts['citation']}
{context_str}

Student Question: {query}

Answer:"""
        
        return prompt