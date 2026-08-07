# ⚡ GadgetSense AI
> **Because the real review is always in the comments.**
 GadgetSense is an AI-powered sentiment analysis platform designed for tech products.
 By aggregating viewer comments from YouTube reviews, GadgetSense filters out spam, runs multi-stage NLP classification,
 and calculates a weighted sentiment score (0–100) alongside actionable pros, cons, and buyer verdicts.

# 🌟 Key Features

Automated Data Scraping & Filtering
Retrieves review videos and comments via multi-instance fallback APIs (Piped & YouTube Data API v3).

Multi-Stage AI Pipeline
Uses ModernBERT-zeroshot to remove non-product chatter (creator praise, shipping complaints, editing talk). 
Classifies product-specific feedback (Positive, Neutral, Negative) using fine-tuned ModernBERT. Employs google/gemma-3-1b-it (4-bit quantized) to summarize
key pros, cons, and an overall buyer recommendation ("BUY", "WAIT", "AVOID").

Weighted Sentiment Index
Weights comments based on review video popularity (log10(views)) to deliver accurate aggregate scores.

Dual Interface
FastAPI Backend exposes REST endpoints (/api/analyze, /api/health) for API integrations. Interactive Gradio UI presents real-time gauge visualizers, 
sentiment breakdown metrics, and video source lists.


#📂 Project Structure

gadgetsense/
├── app.py              # FastAPI server + Gradio web interface entry point
├── sentiment.py        # ModernBERT zero-shot filter, classification & Gemma-3 synthesis
├── youtube_service.py  # YouTube comment scraper with Piped failover & text cleaning
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


# 🛠️ Installation & Quickstart

## Prerequisites

Python 3.10+
An NVIDIA GPU with CUDA support is recommended for execution (e.g., T4/V100/A100).
A Hugging Face user access token.

---

## Option 1: Running in Google Colab (Recommended for Free GPU)

1. Mount Google Drive & Navigate to Workspace:
   from google.colab import drive
   drive.mount('/content/drive')

   %cd /content/drive/MyDrive/Colab_Notebooks/gadgetsense-master

2. Install Required Packages:
   !pip install -q gradio fastapi uvicorn transformers torch bitsandbytes accelerate httpx

3. Authenticate Hugging Face Token:
   from huggingface_hub import login
   login(token="YOUR_HUGGINGFACE_TOKEN")

4. Launch the Application:
   !python app.py

   > Follow the generated public Gradio link (https://xxxx.gradio.live) to open the web application.

---

### Option 2: Local Installation (Linux / Windows / macOS)

1. Clone the Repository:
   git clone https://github.com/YOUR_USERNAME/gadgetsense.git
   cd gadgetsense

2. Create and Activate a Virtual Environment:
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Log into Hugging Face CLI:
   huggingface-cli login

5. Run the Server:
   python app.py

   Open http://localhost:7860 in your web browser.

---

## 🚀 API Endpoint Usage

GadgetSense exposes API endpoints via FastAPI alongside the web UI.

### Health Check
GET /api/health

Response:
{
  "status": "ok"
}

### Analyze Product
POST /api/analyze
Content-Type: application/json

{
  "gadget": "Sony WH-1000XM5"
}

---

