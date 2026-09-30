from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import os

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./fine_tuned_model"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files="dataset.jsonl"
)

def format_example(example):
    text = (
        "### Instruction:\n"
        + example["instruction"]
        + "\n\n"
        + "### Response:\n"
        + example["response"]
        + tokenizer.eos_token
    )
    return {"text": text}

dataset = dataset.map(format_example)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    fp16=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

print("\nStarting fine-tuning...\n")

trainer.train()

print("\nSaving LoRA adapter...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFine-tuning completed!")
print("Saved to:", os.path.abspath(OUTPUT_DIR))