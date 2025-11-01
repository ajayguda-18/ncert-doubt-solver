# scripts/create_evaluation_dataset.py
"""
Create evaluation dataset for testing RAG system
"""

import json
import os
from typing import List, Dict

def create_evaluation_dataset() -> List[Dict]:
    """
    Create evaluation Q&A pairs across different subjects and grades
    """
    
    evaluation_data = [
        # Grade 5 - Math
        {
            "grade": 5,
            "subject": "math",
            "language": "english",
            "question": "What is the formula for the area of a rectangle?",
            "expected_answer": "Area = length × breadth",
            "context": "NCERT Grade 5 Math, Chapter on Mensuration"
        },
        {
            "grade": 5,
            "subject": "math",
            "language": "hindi",
            "question": "आयत का क्षेत्रफल क्या है?",
            "expected_answer": "क्षेत्रफल = लंबाई × चौड़ाई",
            "context": "NCERT Grade 5 Math, Chapter on Mensuration"
        },
        # Grade 7 - Social Science
        {
            "grade": 7,
            "subject": "social_science",
            "language": "english",
            "question": "Who was Akbar?",
            "expected_answer": "Akbar was the third Mughal emperor who ruled India from 1556 to 1605",
            "context": "NCERT Grade 7 History, Chapter on Mughal Empire"
        },
        
        # Grade 8 - Math
        {
            "grade": 8,
            "subject": "math",
            "language": "english",
            "question": "What is the Pythagorean theorem?",
            "expected_answer": "In a right-angled triangle, the square of the hypotenuse is equal to the sum of squares of the other two sides (a² + b² = c²)",
            "context": "NCERT Grade 8 Math, Chapter on Pythagoras Theorem"
        },
        
        # Out of scope questions
        {
            "grade": None,
            "subject": None,
            "language": "english",
            "question": "What is the capital of France?",
            "expected_answer": "I don't know / Out of scope",
            "context": "None - Geography question not in NCERT scope"
        },
    ]
    
    return evaluation_data

def save_dataset():
    """Save evaluation dataset to JSON files"""
    
    data = create_evaluation_dataset()
    
    # Split into train/eval
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    # Create directory
    os.makedirs("data/evaluation", exist_ok=True)
    
    # Save files
    with open("data/evaluation/train_qa.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("data/evaluation/eval_qa.json", 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    with open("data/evaluation/full_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Created evaluation dataset with {len(data)} examples")
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

if __name__ == "__main__":
    save_dataset()