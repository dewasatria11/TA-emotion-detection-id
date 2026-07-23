from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from src.evaluate import compute_metrics

def build_trainer(model_name, tokenized_dataset, output_dir, learning_rate, batch_size, num_epochs, seed=42, max_steps=-1):
    """
    Builds and configures a Hugging Face Trainer instance for sequence classification.
    
    Parameters:
        model_name (str): Pre-trained model name/identifier.
        tokenized_dataset (datasets.DatasetDict): Tokenized dataset containing train and validation splits.
        output_dir (str): Directory where checkpoints and logs are saved.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size per device for training and evaluation.
        num_epochs (int/float): Total number of training epochs.
        seed (int): Random seed for reproducibility.
        max_steps (int): Total number of training steps. If > 0, overrides num_epochs (useful for smoke tests).
        
    Returns:
        transformers.Trainer: Fully configured Hugging Face Trainer instance.
    """
    # Load model for sequence classification with 5 emotion classes
    # (sadness, anger, love, fear, happy)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=5, 
        ignore_mismatched_sizes=True
    )
    
    # Load tokenizer corresponding to the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize DataCollator to pad batch sequences dynamically
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        max_steps=max_steps,
        report_to="none"  # Disable external integrations like WandB/TensorBoard during baseline run
    )
    
    # Early stopping callback: stop training if validation metric does not improve for 2 epochs
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    
    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    return trainer