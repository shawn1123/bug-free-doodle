import torch
from datasets import Dataset
import pandas as pd

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def prepare_data(data_path):
    # Read the text file
    with open(data_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    
    # Format should be: one example per line
    # Each line should be in a conversation format or plain text format
    texts = raw_text.split('\n')
    
    # Create dataset dictionary
    dataset_dict = {'text': [text for text in texts if text.strip()]}
    return Dataset.from_dict(dataset_dict)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

def main():
    # Paths
    model_path = r"C:\Users\daisy\Downloads\mistralocr\llama"
    data_path = r"C:\Users\daisy\Downloads\mistralocr\PyWhatKit_DB.txt"
    
    # Initialize tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    
    # Prepare dataset
    dataset = prepare_data(data_path)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=2e-5,
    )
    
    # Initialize trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model("./fine_tuned_llama")

if __name__ == "__main__":
    main()

"""
Required data format in PyWhatKit_DB.txt should be:
Each line should contain one training example.
Example format:

Question: What is Python?
Answer: Python is a high-level programming language.

Question: How to declare a variable in Python?
Answer: In Python, you can declare a variable by simply assigning a value to it.

OR simple text format:

Python is a high-level programming language known for its simplicity.
Variables in Python are dynamically typed and don't need explicit declaration.

Make sure the text file is properly formatted with one example per line and
contains clean, well-structured text data for better training results.
"""
