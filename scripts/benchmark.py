# scripts/benchmark.py
"""
Benchmark RAG system performance
"""

import sys
import os
import time
import json
from typing import List, Dict
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.rag_service import RAGService
from backend.services.embedding_service import EmbeddingService
from backend.services.Language_service import LanguageService
from backend.app.config import settings

class RAGBenchmark:
    def __init__(self):
        # Initialize services
        self.embedding_service = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL,
            vector_db_path=settings.VECTOR_DB_PATH
        )
        self.embedding_service.create_collection(settings.COLLECTION_NAME)
        
        self.language_service = LanguageService()
        
        self.rag_service = RAGService(
            model_name=settings.LLM_MODEL,
            embedding_service=self.embedding_service,
            language_service=self.language_service
        )
    
    def load_evaluation_data(self, file_path: str) -> List[Dict]:
        """Load evaluation dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        
        total = len(results)
        latencies = [r["latency_ms"] for r in results]
        
        # Latency metrics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Check latency target
        within_target = sum(1 for l in latencies if l <= settings.MAX_LATENCY_SECONDS * 1000)
        latency_success_rate = (within_target / total) * 100
        
        # Citation metrics
        with_citations = sum(1 for r in results if r["has_citations"])
        citation_rate = (with_citations / total) * 100
        
        # Confidence metrics
        high_confidence = sum(1 for r in results if r["confidence"] == "high")
        confidence_rate = (high_confidence / total) * 100
        
        return {
            "total_queries": total,
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "min_latency_ms": min_latency,
            "latency_success_rate": latency_success_rate,
            "citation_rate": citation_rate,
            "confidence_rate": confidence_rate,
        }
    
    def run_benchmark(self, eval_data: List[Dict]) -> List[Dict]:
        """Run benchmark on evaluation data"""
        
        results = []
        
        for item in eval_data:
            start_time = time.time()
            
            try:
                response = self.rag_service.generate_answer(
                    query=item["question"],
                    grade=item.get("grade"),
                    subject=item.get("subject"),
                    language=item.get("language")
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                result = {
                    "question": item["question"],
                    "answer": response["answer"],
                    "expected": item["expected_answer"],
                    "latency_ms": latency_ms,
                    "has_citations": len(response["contexts"]) > 0,
                    "confidence": response["confidence"],
                    "language": response["language"],
                    "grade": item.get("grade"),
                    "subject": item.get("subject")
                }
                
                results.append(result)
                
                print(f"✓ Processed: {item['question'][:50]}... ({latency_ms:.2f}ms)")
            
            except Exception as e:
                print(f"✗ Error processing question: {e}")
                results.append({
                    "question": item["question"],
                    "error": str(e),
                    "latency_ms": (time.time() - start_time) * 1000
                })
        
        return results
    
    def generate_report(self, results: List[Dict], metrics: Dict):
        """Generate benchmark report"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          NCERT Doubt-Solver Benchmark Report                 ║
╚══════════════════════════════════════════════════════════════╝

PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Queries:           {metrics['total_queries']}
Average Latency:         {metrics['avg_latency_ms']:.2f}ms
Max Latency:             {metrics['max_latency_ms']:.2f}ms
Min Latency:             {metrics['min_latency_ms']:.2f}ms

TARGET COMPLIANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Latency ≤ 5s:            {metrics['latency_success_rate']:.1f}% ({'✓' if metrics['latency_success_rate'] >= 90 else '✗'})
Citation Rate:           {metrics['citation_rate']:.1f}% ({'✓' if metrics['citation_rate'] >= 85 else '✗'})
High Confidence:         {metrics['confidence_rate']:.1f}%

TARGET STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Latency Target (≤5s):    {'✓ PASS' if metrics['latency_success_rate'] >= 90 else '✗ FAIL'}
Citation Target (≥85%):  {'✓ PASS' if metrics['citation_rate'] >= 85 else '✗ FAIL'}
"""
        
        print(report)
        
        # Save detailed results
        os.makedirs("benchmark_results", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        with open(f"benchmark_results/results_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 Detailed results saved to: benchmark_results/results_{timestamp}.json")

def main():
    print("Starting RAG System Benchmark...\n")
    
    benchmark = RAGBenchmark()
    
    # Load evaluation data
    eval_data = benchmark.load_evaluation_data("data/evaluation/eval_qa.json")
    print(f"Loaded {len(eval_data)} evaluation examples\n")
    
    # Run benchmark
    results = benchmark.run_benchmark(eval_data)
    
    # Calculate metrics
    metrics = benchmark.calculate_metrics(results)
    
    # Generate report
    benchmark.generate_report(results, metrics)

if __name__ == "__main__":
    main()