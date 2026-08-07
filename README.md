# GadgetSense AI: View-Weighted Sentiment Analysis Engine

GadgetSense is an end-to-end NLP intelligence pipeline that quantifies consumer sentiment from YouTube review comment sections. It filters out irrelevancies, classifies sentiment using a fine-tuned encoder, and synthesizes structured buying recommendations using a 4-bit quantized Causal LLM.

---

## 🏗️ System Architecture

                              +-----------------------+
                              | YouTube Video Comments|
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Text Cleaning Pipeline|
                              | (HTML, Regex, Dupes)  |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              |  Zero-Shot Bouncer    |
                              |  (ModernBERT-v2.0)    |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              |  Sentiment Engine     |
                              | (Fine-Tuned ModernBERT|
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Weighted Score Engine |
                              |  Score Calculation    |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Synthesis & Report    |
                              |  (Gemma-3-1b-it 4-Bit)|
                              +-----------------------+

---

## 🔬 Tech Stack & ML Methodology

### Data Ingestion & Fallback Layer
* Primary API: Multi-instance Piped API rotation for comment scraping without rate limits.
* Secondary Fallback: YouTube Data API v3 (`google-api-python-client`).
* Cleaning Filters: Regex stripping for URLs, HTML entities, length boundaries (5-500 characters), and emoji-only noise.

### NLP Pipeline Stages
1. Filtering (`MoritzLaurer/ModernBERT-base-zeroshot-v2.0`): Filters non-product comments (creator praise, video editing, delivery feedback) with a confidence threshold >= 0.70.
2. Classification (`karlsefni/gadgetsense-modernbert`): Batched inferencing (Batch Size = 64) classifying comments into Positive, Neutral, or Negative labels.
3. Weighted Scoring Index: Calculates aggregate product score using logarithmic view weighting:

   Score = ((Sum(Sentiment_i * log10(Views_i + 1)) / Sum(log10(Views_i + 1))) + 1) / 2 * 100

4. Synthesis (`google/gemma-3-1b-it`): Generates structured JSON reports (5 distinct Pros, 5 Cons, and a 2-3 sentence Verdict) using NF4 BitsAndBytes 4-bit quantization.

---

## 📂 Repository Structure

gadgetsense/
├── app.py              # FastAPI server + Gradio web application entry point
├── sentiment.py        # Inference pipeline (Zero-Shot, ModernBERT, Gemma-3)
├── youtube_service.py  # Asynchronous Piped/YouTube API client & text cleaner
├── requirements.txt    # Fixed dependency lock file
└── README.md           # Technical documentation

---

## ⚙️ Technical Requirements & Dependencies

* Hardware: NVIDIA GPU with CUDA support (Minimum 8 GB VRAM for Gemma-3-1b 4-bit + ModernBERT inference).
* Runtime: Python 3.10+
* Key Libraries: `transformers`, `torch`, `bitsandbytes`, `accelerate`, `fastapi`, `gradio`, `httpx`

---

## ⚡ Quickstart Guide

### Running in Google Colab

1. Mount Drive & Navigate to Repository:
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/Colab_Notebooks/gadgetsense-master

2. Install Dependencies:
   !pip install -q gradio fastapi uvicorn transformers torch bitsandbytes accelerate httpx

3. Authenticate Hugging Face:
   from huggingface_hub import login
   login(token="YOUR_HF_TOKEN")

4. Launch Server & UI:
   !python app.py

---

### Local Setup

1. Clone & Set Up Environment:
   git clone https://github.com/YOUR_USERNAME/gadgetsense.git
   cd gadgetsense
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install Packages & Login:
   pip install -r requirements.txt
   huggingface-cli login

3. Optional Environment Variables:
   export YOUTUBE_API_KEY="your_optional_youtube_v3_key"

4. Execute:
   python app.py

---

## 📡 API Specification

### Health Check
* GET `/api/health`
* Response: `{"status": "ok"}`

### Sentiment Analysis Query
* POST `/api/analyze`
* Payload:
  {
    "gadget": "MacBook Pro M3"
  }
* Response Schema:
  {
    "score": 82,
    "total_comments": 420,
    "positive_count": 310,
    "neutral_count": 70,
    "negative_count": 40,
    "pros": ["Display Quality", "Battery Efficiency"],
    "cons": ["High Base Price", "Limited Port Selection"],
    "verdict": "BUY",
    "verdict_summary": "Highly favored for performance and display upgrades...",
    "videos": [...]
  }

---
