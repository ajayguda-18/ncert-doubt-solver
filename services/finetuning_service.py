# backend/services/finetuning_service.py
"""
Finetuning service for regional language models
"""

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FinetuningService:
    def __init__(self, base_model: str, output_dir: str):
        self.base_model = base_model
        self.output_dir = output_dir
        
    def prepare_dataset(
        self, 
        qa_pairs: List[Dict[str, str]]
    ) -> Dataset:
        """
        Prepare Q&A dataset for finetuning
        
        Format:
        [
            {"question": "...", "answer": "...", "context": "..."},
            ...
        ]
        """
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        def format_example(example):
            prompt = f"""Context: {example['context']}

Question: {example['question']}

Answer: {example['answer']}"""
            
            return {"text": prompt}
        
        formatted_data = [format_example(qa) for qa in qa_pairs]
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def finetune(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5
    ):
        """Finetune the model"""
        
        logger.info(f"Starting finetuning of {self.base_model}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=True,
            gradient_accumulation_steps=4,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Finetuning complete. Model saved to {self.output_dir}")


