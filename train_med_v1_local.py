import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def main():
    model_name = "roberta-base"
    data_file = "data/pubmed_v1_50000.jsonl"
    output_dir = "./outputs/med-v1"

    seed = 42
    max_length = 256
    num_train_epochs = 2
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.06
    logging_steps = 50
    dataloader_num_workers = 4
    mlm_probability = 0.15

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    print("=" * 60)
    print("Starting Med-V1 continued pretraining")
    print("Model:", model_name)
    print("Data file:", data_file)
    print("Output dir:", output_dir)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")
    print("=" * 60)

    dataset = load_dataset("json", data_files=data_file, split="train")
    print("Loaded dataset size:", len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing local PubMed corpus",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    train_result = trainer.train()

    print("Training finished.")
    print("Metrics:", train_result.metrics)

    print("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("=" * 60)
    print("Done.")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

