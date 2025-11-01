import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.finetuning_service import FinetuningService
from backend.app.config import settings
import json

def load_qa_data(file_path: str):
    """Load Q&A pairs from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Load training data
    train_data = load_qa_data("data/evaluation/train_qa.json")
    eval_data = load_qa_data("data/evaluation/eval_qa.json")
    
    # Initialize finetuning service
    finetuning_service = FinetuningService(
        base_model="ai4bharat/indic-bert",  # or any Indic language model
        output_dir=os.path.join(settings.MODEL_PATH, "finetuned_hindi")
    )
    
    # Prepare datasets
    train_dataset = finetuning_service.prepare_dataset(train_data)
    eval_dataset = finetuning_service.prepare_dataset(eval_data)
    
    # Finetune
    finetuning_service.finetune(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=3,
        batch_size=4
    )

if __name__ == "__main__":
    main()