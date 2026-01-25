---
title: AI Image Detector
emoji: 📸
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

-----

\

# 🚀 AI Image Analyzer

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-%F0%9F%A4%97%20Spaces-blue)](https://huggingface.co/spaces/fysiki/ai-image-detector)

A full-stack web application to detect AI-generated images, identify objects, and provide an explainable AI (XAI) summary of the analysis.

**[Try the live demo here!](https://huggingface.co/spaces/fysiki/ai-image-detector)**

---


## 📖 About This Project

This project is a complete AI-powered tool built for a university presentation. It provides a simple web interface where a user can upload an image and receive a multi-part analysis. The goal is to not only classify an image but to explain *why* it was classified, using modern AI and XAI techniques.

## ✨ Core Features

* **AI vs. Real Classification:** Uses a **Vision Transformer (ViT)** model fine-tuned on the `ateeqq/ai-vs-human-image-detector` dataset to classify an image as "AI-generated" or "Real" and provide a confidence score.
* **Object Detection:** Employs **YOLOv8** to identify and list common objects found in the image.
* **Explainable AI (XAI):** Generates a **Grad-CAM** (Gradient-weighted Class Activation Mapping) visualization. This heatmap shows exactly which parts of the image the ViT model focused on to make its decision.
* **AI-Powered Summary:** Uses the **Google Gemini API** to generate a human-readable summary that syntesizes all the findings (classification, objects, and Grad-CAM) into a coherent explanation.
* **Full-Stack & Deployed:** A complete frontend (HTML/CSS/JS) and backend (FastAPI) application, containerized with **Docker** and deployed on **Hugging Face Spaces**.

---

## 🛠 Tech Stack

| Area | Technology |
| :--- | :--- |
| **Backend** | Python, FastAPI, Uvicorn |
| **Machine Learning** | PyTorch, Hugging Face Transformers, Ultralytics (YOLOv8) |
| **Generative AI** | Google Generative AI (Gemini) |
| **Frontend** | HTML, CSS, JavaScript (Fetch API, localStorage) |
| **Deployment** | Docker, Git LFS, Hugging Face Spaces |

---

## 🔧 How It Works: The Analysis Pipeline

1.  **Frontend:** A user uploads an image. The browser's `fetch` API sends this file to the `/analyze-image` endpoint.
2.  **Backend (FastAPI):**
    * The file is saved to a temporary `/tmp/uploads` directory (necessary for the read-only file system on Hugging Face).
    * **YOLOv8** runs `detect_objects()` on the file.
    * The **ViT Model** runs `detect_ai_image()` on the file.
    * The **Grad-CAM** utility runs `generate_gradcam()` on the file.
    * The **Gemini API** is called with all this data to `generate_gemini_summary()`.
3.  **Frontend:** The backend returns a single JSON object. The frontend parses this, saves it to `localStorage`, and redirects to `result.html` to display all the information.

---

## Chrome Extension

A Chrome extension is included that allows you to analyze images directly from your browser. **Works out of the box with the hosted API** - no setup required!

### Installation

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `chrome-extension` directory
5. Right-click any image on the web and select "Analyze with AI Image Detector"

The extension uses the hosted API by default (`https://fysiki-ai-image-detector.hf.space`). You can change this in the extension settings if you want to use a local server.

---

## 💻 How to Run Locally

### 1. Prerequisites
* Python 3.10+
* Git
* Git LFS (Git Large File Storage)

### 2. Clone the Repository
```bash
git clone [https://github.com/142502022/ai-image-detect.git](https://github.com/142502022/ai-image-detect.git)
cd ai-image-detect
````

### 3\. Download the 300MB Model File

This project uses Git LFS for the large ViT model.

```bash
# Install LFS
git lfs install

# Pull the large files
git lfs pull
```

You should now see `model.safetensors` inside the `backend/vit-ai-vs-real-model` folder.

### 4\. Set Up the Environment

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all Python dependencies
pip install -r backend/requirements.txt
```

### 5\. Configure Environment Variables

Create a file named `.env` in the root of the project:

```
# Create this file
nano .env
```

Add your Google API key to it:

```
GOOGLE_API_KEY=AIza...
```

### 6\. Run the Server

The FastAPI app is configured to serve both the API and the frontend files.

```bash
uvicorn backend.main:app --reload --port 7860
```

Open your browser to **`http://127.0.0.1:7860`**

```
```
