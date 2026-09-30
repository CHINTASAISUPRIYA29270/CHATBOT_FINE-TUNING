from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTER_PATH = "./fine_tuned_model"

print("Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)

print("Loading fine-tuned LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("Model loaded successfully!")
print("Java Fine-Tuning Chatbot")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    prompt = (
        "### Instruction:\n"
        + question
        + "\n\n"
        + "### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Response:" in answer:
        answer = answer.split("### Response:")[-1].strip()

    print("Chatbot:", answer)
    print()