from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel  # <-- NEW ADDITION: For request body
import shutil
import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from .utils import detect_objects, generate_gradcam
from dotenv import load_dotenv
import google.generativeai as genai



load_dotenv()


app = FastAPI()


# ========================================
# CORS Configuration
# ========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    allow_credentials=True,
)


# ✅ Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = None

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Find available model
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Try models in order of preference
        model_priority = [
            'models/gemini-2.5-flash',
            'models/gemini-flash-latest',
            'models/gemini-2.0-flash',
            'models/gemini-2.5-flash-lite',
            'models/gemini-flash-lite-latest',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro'
        ]
        
        for preferred in model_priority:
            if preferred in available_models:
                GEMINI_MODEL = preferred
                break
        
        if not GEMINI_MODEL and available_models:
            GEMINI_MODEL = available_models[0]
        
        if GEMINI_MODEL:
            print(f"✅ Gemini API configured with model: {GEMINI_MODEL}")
        else:
            print("⚠️ No Gemini models available for content generation")
            
    except Exception as e:
        print(f"⚠️ Gemini API configuration error: {e}")
else:
    print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables")

# ========================================
# Model Setup
# ========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vit-ai-vs-real-model")
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def detect_ai_image(image_path: str):
    """
    Detect whether an image is AI-generated or real using the ViT model.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    ai_prob = probs[0][0].item()  # 'AI-generated'
    real_prob = probs[0][1].item()  # 'Real'
    
    return {
        "AI-generated": ai_prob,
        "Real": real_prob,
        "prediction": "AI-generated" if ai_prob > real_prob else "Real"
    }

def generate_gemini_summary(
    original_image_path: str = None,
    grad_cam_image_path: str = None,
    classification: str = None,
    probability_percent: float = None,
    objects: list = None
) -> str:
    """
    Unified Gemini summary generator for both vision and text-only models.
    Automatically handles:
    - Vision model input (original + Grad-CAM)
    - Text-only fallback
    - Quota or API errors
    """

    try:
        if not GEMINI_MODEL:
            raise ValueError("No Gemini model available")

        print(f"🤖 Using Gemini model: {GEMINI_MODEL}")
        gemini_model_instance = genai.GenerativeModel(GEMINI_MODEL)

        # Check if this model supports image input
        is_vision_model = any(x in GEMINI_MODEL.lower() for x in ["1.5", "2.0", "2.5", "vision"])

        # ============ CASE 1: Vision Model (Uses Original + Grad-CAM) ============
        if original_image_path and grad_cam_image_path and is_vision_model:
            img_original = Image.open(original_image_path)
            img_grad_cam = Image.open(grad_cam_image_path)

            prompt = f"""
            You are an expert in AI image forensics and explainable AI (XAI).

            Two images are provided:
            1. The **original image**
            2. A **Grad-CAM heatmap** showing where an AI detection model focused (red/yellow = strong attention).

            The model's result:
            • **Classification:** {classification or "Unknown"}
            • **Confidence:** {probability_percent or 0:.2f}%

            Your task:
            Write a short (2–3 sentence) explanation describing why the model reached this conclusion.
            Specifically:
            - What regions were highlighted by Grad-CAM?
            - Why might those regions indicate this classification?
            - How does the confidence relate to the visual clues?
            """

            response = gemini_model_instance.generate_content([prompt, img_original, img_grad_cam])
            return response.text.strip()

        # ============ CASE 2: Text-Only Model (No Vision Support) ============
        else:
            if objects:
                prompt = f"""
                The AI model detected these objects: {', '.join(objects)}.
                Its classification result was: {classification or "Unknown"} ({probability_percent or 0:.2f}%).
                Write a short, 2–3 sentence explanation of why the image might appear {classification.lower() if classification else 'AI-generated or real'}.
                """
            else:
                prompt = f"""
                The model predicted: {classification or "Unknown"} ({probability_percent or 0:.2f}% confidence).
                Provide a short summary of possible visual cues that led to this result.
                """

            response = gemini_model_instance.generate_content(prompt)
            return response.text.strip()

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Gemini API error: {error_msg}")

        # Handle quota errors gracefully
        if "429" in error_msg or "quota" in error_msg.lower():
            print("💡 Quota exceeded — using fallback summary.")
            if not objects:
                return "This image likely contains a few simple objects or visual elements, but the API quota has been reached."
            elif len(objects) == 1:
                return f"This image features a {objects[0]}."
            elif len(objects) == 2:
                return f"This image shows a {objects[0]} and a {objects[1]}."
            else:
                return f"This image contains {', '.join(objects[:-1])}, and {objects[-1]}."
        else:
            # Generic fallback summary
            return f"Unable to generate Gemini summary. Possible reason: {error_msg}"

# ========================================
# Upload Folder Setup
# ========================================
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve uploaded images as static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# ========================================
# Endpoints
# ========================================

# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {"status": "running", "message": "FastAPI server is running"}


# ========================================
# Endpoints
# ========================================

# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {"status": "running", "message": "FastAPI server is running"}




@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image for objects, AI detection, Grad-CAM, and Gemini summary."""

    # 1️⃣ Save uploaded file to backend/uploads/
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print(f"✅ Image saved: {file_path}")

    # 2️⃣ Run object detection (YOLO)
    objects = detect_objects(file_path)
    print(f"🧠 Detected objects: {objects}")

    # 3️⃣ Run AI vs Real detection (ViT)
    ai_result = detect_ai_image(file_path)
    ai_probability = ai_result.get("AI-generated", 0.0)
    print(f"🤖 AI Probability: {ai_probability:.2f}")

    # 4️⃣ Generate Grad-CAM visualization
    gradcam_filename = f"gradcam_{file.filename}"
    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    try:
        generate_gradcam(file_path, gradcam_path)
        print(f"🔥 Grad-CAM saved: {gradcam_path}")
    except Exception as e:
        print(f"⚠️ Grad-CAM generation failed: {e}")
        gradcam_path = None

    # 5️⃣ Generate Gemini summary
    gemini_summary = generate_gemini_summary(
        original_image_path=file_path,
        grad_cam_image_path=gradcam_path,
        classification="AI-Generated" if ai_probability > 0.5 else "Real",
        probability_percent=ai_probability * 100,
        objects=objects
    )
    print(f"🪶 Gemini summary: {gemini_summary[:100]}...")

    # 6️⃣ Return everything as JSON
    return JSONResponse({
        "filename": file.filename,
        "objects": objects,
        "ai_probability": float(ai_probability),
        "gradcam_image": f"/uploads/{gradcam_filename}" if gradcam_path else None,
        "summary": gemini_summary
    })


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """Generate Grad-CAM visualization for uploaded image"""
    temp_file_path = None
    
    try:
        print(f"📥 Received file: {file.filename}")
        
        temp_file_path = os.path.join(UPLOAD_FOLDER, f"temp_{file.filename}")
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        print(f"💾 Saved temp file to: {temp_file_path}")
        
        output_filename = f"gradcam_{file.filename}"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        print(f"🎨 Generating Grad-CAM...")
        gradcam_path = generate_gradcam(temp_file_path, output_path)
        
        if not os.path.exists(gradcam_path):
            raise FileNotFoundError(f"Grad-CAM file was not created at {gradcam_path}")
        
        rel_path = f"/uploads/{output_filename}"
        print(f"✅ Grad-CAM generated successfully: {rel_path}")
        
        return JSONResponse({
            "success": True,
            "gradcam_image_url": rel_path,
            "message": "Grad-CAM generated successfully"
        })
    
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": f"Failed to generate Grad-CAM: {str(e)}"
            }
        )
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"🗑️ Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                print(f"⚠️ Could not delete temp file: {e}")

# ========================================
# <-- NEW ADDITION: Endpoint for Classification Analysis -->
# ========================================

class ExplanationRequest(BaseModel):
    """Defines the request body for the explanation endpoint."""
    original_image_url: str
    grad_cam_image_url: str
    classification: str
    probability_percent: float

def get_path_from_url(url: str) -> str:
    """Helper to convert /uploads/filename.jpg to uploads/filename.jpg"""
    # This is a simple helper. Assumes all URLs are relative.
    if url.startswith('/'):
        url = url[1:]
    return url

@app.post("/explain-classification")
async def explain_classification(request: ExplanationRequest):
    """
    Generates a Gemini-powered explanation for a classification.
    """
    try:
        print(f"🧠 Generating XAI explanation for: {request.original_image_url}")
        
        # Convert URLs to local file paths
        # Assumes UPLOAD_FOLDER is the root for these URLs
        original_path = get_path_from_url(request.original_image_url)
        grad_cam_path = get_path_from_url(request.grad_cam_image_url)
        
        # Safety check
        if not os.path.exists(original_path) or not os.path.exists(grad_cam_path):
            print(f"❌ File not found. Tried: {original_path} and {grad_cam_path}")
            return JSONResponse(status_code=404, content={"error": "Image file not found on server."})
        
        # Call the new Gemini function
        explanation = generate_gemini_summary(
            original_image_path=original_path,
            grad_cam_image_path=grad_cam_path,
            classification=request.classification,
            probability_percent=request.probability_percent
        )
        
        print(f"✅ XAI explanation generated.")
        
        return JSONResponse({
            "success": True,
            "explanation": explanation
        })

    except Exception as e:
        print(f"❌ Error in explain-classification: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": f"Failed to generate explanation: {str(e)}"
            }
        )


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
# ========================================
# Run Server
# ========================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print(f"📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🤖 Model path: {os.path.abspath(MODEL_PATH)}")
    print(f"💻 Device: {device}")
    uvicorn.run(app, host="0.0.0.0", port=7860)