import torch
from transformers import AutoModelForSequenceClassification

# Load the original PyTorch model
model_name = "j-hartmann/emotion-english-distilroberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Dummy input for ONNX conversion
dummy_input = {
    "input_ids": torch.ones(1, 128, dtype=torch.int64),
    "attention_mask": torch.ones(1, 128, dtype=torch.int64),
}


# Export to ONNX format with opset version 14
onnx_model_path = "model/emotion_model.onnx"
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_model_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=14  # Change from 11 to 14
)

print(f"Model converted to ONNX and saved at: {onnx_model_path}")
