import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from huggingface_hub import hf_hub_download

# Set your device (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model
model_name = "Dafilab/ai-image-detector"
processor = ViTImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)

# Your labels
LABEL_MAPPING = {0: "ai", 1: "human"}

# Load your image
image_path = "/Users/fysiki_mac/Desktop/ML OS IIT/college_pro/Data/6863578.jpg"
img = Image.open(image_path).convert("RGB")

# Process the image and make a prediction
with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    logits = model(**inputs).logits

    # Get probabilities (the confidence scores!)
    probs = torch.nn.functional.softmax(logits, dim=1)

    # Get the top prediction
    predicted_class_idx = torch.argmax(probs, dim=1).item()
    predicted_class = LABEL_MAPPING[predicted_class_idx]
    confidence = probs[0, predicted_class_idx].item()

print(f"Predicted Class: {predicted_class}")
print(f"Confidence Score: {confidence * 100:.2f}%")