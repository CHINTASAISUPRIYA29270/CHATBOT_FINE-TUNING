from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTER_PATH = os.path.join(BASE_DIR, "fine_tuned_model")

print("Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

print("Loading fine-tuned LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("Model loaded successfully!")


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })

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

    return jsonify({
        "answer": answer
    })


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":

    print("Starting Fine-Tuning Chatbot...")
    print("Open: http://127.0.0.1:5003")

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5003
    )