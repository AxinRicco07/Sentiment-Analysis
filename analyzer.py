from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os

# Load ONNX model
onnx_model_path = os.path.join("model", "emotion_model.onnx")
session = ort.InferenceSession(onnx_model_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Define FastAPI app
app = FastAPI()

# Define input data structure
class TextRequest(BaseModel):
    text: str

# Emotion Labels
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def softmax(logits):
    """Apply softmax to convert logits into probabilities"""
    exp_logits = np.exp(logits - np.max(logits))  # Stability trick
    return exp_logits / np.sum(exp_logits)

@app.post("/analyze")
async def sentiment_analyzer(request: TextRequest):
    """
    Process user input, tokenize it, run inference using ONNX model,
    apply softmax, and return the predicted sentiment/emotion.
    """
    # Tokenize input text
    inputs = tokenizer(
        request.text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="np"
    )

    # Prepare input dictionary for ONNX
    input_dict = {"input_ids": inputs["input_ids"].astype(np.int64)}

    # Include attention mask if the model expects it
    model_inputs = [inp.name for inp in session.get_inputs()]
    if "attention_mask" in model_inputs:
        input_dict["attention_mask"] = inputs["attention_mask"].astype(np.int64)

    # Run inference
    outputs = session.run(None, input_dict)[0]  # Raw logits from model

    # Apply softmax to get probabilities
    probs = softmax(outputs[0])  # Convert logits to probability distribution

    # Get the highest probability emotion
    predicted_index = np.argmax(probs)
    sentiment = LABELS[predicted_index]

    # Debugging Output (Check Scores)
    print(f"Text: {request.text}")
    print(f"Logits: {outputs[0]}")
    print(f"Probabilities: {probs}")
    print(f"Predicted Emotion: {sentiment}")

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": float(probs[predicted_index]),  # Show confidence score
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}  # Show all probabilities
    }
