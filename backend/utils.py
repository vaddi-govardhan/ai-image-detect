# utils.py
from ultralytics import YOLO
import os
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()



# ✅ Load .env from this backend folder
load_dotenv(Path(__file__).resolve().parent / ".env")

# ✅ Now fetch your Gemini key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("❌ GOOGLE_API_KEY not found! Check your .env file path.")
else:
    print("✅ GOOGLE_API_KEY loaded successfully.")

# ========================================
# YOLO Object Detection
# ========================================
yolo_model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    """Detect objects in image using YOLO"""
    results = yolo_model(image_path)
    detected_objects = []

    for result in results:
        for cls in result.boxes.cls:
            class_idx = int(cls)
            class_name = result.names[class_idx]
            detected_objects.append(class_name)

    # Remove duplicates
    return list(set(detected_objects))


# ========================================
# ViT Model for AI Detection (used in main.py)
# ========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vit-ai-vs-real-model")
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# ========================================
# Grad-CAM Visualization
# ========================================
def generate_gradcam(image_path, output_path="gradcam_result.jpg"):
    """
    Generates Grad-CAM heatmap for the ViT model and saves it as an image file.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Enable gradient computation
    inputs["pixel_values"].requires_grad_(True)
    
    # Lists to capture attention weights and gradients
    attention_weights = []
    attention_grads = []
    
    def forward_hook(module, input, output):
        """Capture attention weights during forward pass"""
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights.append(output[1])
        else:
            attention_weights.append(None)
    
    def backward_hook(module, grad_input, grad_output):
        """Capture gradients during backward pass"""
        if grad_output[0] is not None:
            attention_grads.append(grad_output[0])
    
    # Register hooks on the last attention layer
    last_attn = model.vit.encoder.layer[-1].attention.attention
    fwd_handle = last_attn.register_forward_hook(forward_hook)
    bwd_handle = last_attn.register_full_backward_hook(backward_hook)
    
    try:
        print("🔍 Running forward pass...")
        # Forward pass
        outputs = model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=-1).item()
        score = outputs.logits[:, pred_class]
        
        print(f"🎯 Predicted class: {pred_class}, Score: {score.item():.4f}")
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        print(f"📊 Captured {len(attention_weights)} attention weights")
        print(f"📊 Captured {len(attention_grads)} gradients")
        
        # Check if we captured valid attention data
        if len(attention_weights) == 0 or attention_weights[0] is None:
            print("⚠️ No attention weights captured - using alternative method")
            
            # Alternative: Use the last hidden state's gradients
            patch_grad = inputs["pixel_values"].grad
            
            if patch_grad is not None:
                # Average over channels and batch
                cam = patch_grad.abs().mean(dim=[0, 1]).detach().cpu().numpy()
                
                # Resize to reasonable spatial dimensions
                import torch.nn.functional as F
                h, w = cam.shape
                target_size = 14  # ViT typically uses 14x14 patches
                cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
                cam = F.interpolate(cam_tensor, size=(target_size, target_size), 
                                   mode='bilinear', align_corners=False)
                cam = cam.squeeze().numpy()
            else:
                print("⚠️ No gradients found - using center-focused fallback")
                # Create a simple center-focused heatmap as fallback
                cam = np.ones((14, 14))
                center = 7
                for i in range(14):
                    for j in range(14):
                        dist = np.sqrt((i - center)**2 + (j - center)**2)
                        cam[i, j] = max(0, 1 - dist / 10)
        else:
            print("✅ Processing attention weights...")
            # Get attention weights and gradients
            attn = attention_weights[0]  # [batch, heads, seq_len, seq_len]
            
            if len(attention_grads) > 0:
                grad = attention_grads[0]
                
                # Average over heads
                weights = grad.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
                attn_map = attn.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
                
                # Weight the attention map by gradients
                cam = (weights * attn_map).sum(dim=0).detach().cpu().numpy()
            else:
                # Just use attention without gradients
                attn_map = attn.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
                cam = attn_map.mean(dim=0).detach().cpu().numpy()
            
            # Remove CLS token (first position) and reshape to spatial dimensions
            cam = cam[1:]  # Remove CLS token
            size = int(np.sqrt(len(cam)))
            cam = cam[:size*size].reshape(size, size)
        
        # Normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.ones_like(cam) * 0.5
        
        print(f"📏 CAM shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}")
    
    except Exception as e:
        print(f"❌ Error during Grad-CAM computation: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: create simple heatmap
        print("🔄 Using fallback heatmap")
        cam = np.ones((14, 14)) * 0.5
    
    finally:
        # Remove hooks
        fwd_handle.remove()
        bwd_handle.remove()
    
    # Resize CAM to match input image size
    cam = cv2.resize(cam, (image.size[0], image.size[1]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    overlay = np.array(image) * 0.6 + heatmap * 0.4
    overlay = np.uint8(overlay)
    
    # Save Grad-CAM image
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print("🟢 Grad-CAM generated!")
    print(f"💾 Saved at: {os.path.abspath(output_path)}")
    
    return output_path

def generate_gemini_summary(
    original_image_path: str = None,
    grad_cam_image_path: str = None,
    classification: str = None,
    probability_percent: float = None,
    objects: list = None
) -> str:
    """
    Unified Gemini summary generator for both vision and text-only models.
    Works for Gemini 1.5 (vision) and text-only Gemini models.
    """

    try:
        if not GEMINI_MODEL:
            raise ValueError("No Gemini model available")

        print(f"🤖 Using Gemini model: {GEMINI_MODEL}")
        gemini_model_instance = genai.GenerativeModel(GEMINI_MODEL)

        # Detect if model supports vision
        is_vision_model = any(x in GEMINI_MODEL.lower() for x in ["1.5", "2.0", "2.5", "vision"])

        # Vision model → send both images
        if original_image_path and grad_cam_image_path and is_vision_model:
            with open(original_image_path, "rb") as f1, open(grad_cam_image_path, "rb") as f2:
                img_original = {"mime_type": "image/jpeg", "data": f1.read()}
                img_gradcam = {"mime_type": "image/jpeg", "data": f2.read()}

            prompt = f"""
            You are an expert in AI image forensics and explainable AI (XAI).

            Two images are provided:
            1. The original image.
            2. The Grad-CAM heatmap (red/yellow = strong focus regions).

            The detection model's result:
            • Classification: {classification or "Unknown"}
            • Probability: {probability_percent or 0:.2f}%

            Write a concise 2–3 sentence explanation describing:
            - Which regions were highlighted by Grad-CAM.
            - Why those regions indicate the model’s decision.
            - How the confidence reflects these cues.
            """

            response = gemini_model_instance.generate_content([prompt, img_original, img_gradcam])
            return response.text.strip()

        # Text-only fallback
        else:
            prompt = f"""
            The model detected these objects: {', '.join(objects or [])}.
            Classification result: {classification or "Unknown"} ({probability_percent or 0:.2f}% confidence).
            Write a short 2–3 sentence explanation of why the image might appear {classification.lower() if classification else 'AI-generated or real'}.
            """
            response = gemini_model_instance.generate_content(prompt)
            return response.text.strip()

    except Exception as e:
        print(f"⚠️ Gemini summary error: {e}")
        err = str(e)
        if "429" in err or "quota" in err.lower():
            print("💡 Quota exceeded — fallback summary used.")
            if not objects:
                return "This image likely contains a few simple visual features, but detailed analysis is unavailable."
            elif len(objects) == 1:
                return f"This image features a {objects[0]}."
            elif len(objects) == 2:
                return f"This image shows a {objects[0]} and a {objects[1]}."
            else:
                return f"This image contains {', '.join(objects[:-1])}, and {objects[-1]}."
        else:
            return f"Unable to generate Gemini summary. Possible reason: {err}"
