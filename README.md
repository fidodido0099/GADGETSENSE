# ⚡ GadgetSense AI: End-to-End Sentiment Intelligence Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Because the real review is always in the comments.**

GadgetSense AI is a full-stack Natural Language Processing (NLP) intelligence platform designed to extract, filter, and quantify real consumer sentiment from YouTube tech product reviews. By combining multi-instance data collection, zero-shot filtering, fine-tuned transformer classification, and 4-bit quantized Causal LLM synthesis, GadgetSense converts thousands of unstructured comments into actionable buying recommendations.

---

## 🎯 Executive Summary & Impact

When making purchase decisions, buyers spend hours watching long-form reviews where the most candid user feedback is hidden inside thousands of video comments. GadgetSense automates product research by:

* **Eliminating Noise**: Automatically filters out creator praise, channel feedback, and shipping complaints using zero-shot classification.
* **Weighted Data Aggregation**: Weights user feedback dynamically based on video popularity to deliver unbiased sentiment scores (0–100).
* **Automated Buying Advice**: Synthesizes 5 distinct strengths, 5 key weaknesses, and an actionable buyer verdict ("BUY", "WAIT", "AVOID").

---

## 🏗️ System Architecture & Data Pipeline

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

### 1. Data Ingestion & Resiliency Layer
* **Primary Source**: Multi-instance Piped API rotation for rate-limit-free comment scraping.
* **Secondary Fallback**: YouTube Data API v3 (`httpx` asynchronous fetching).
* **Preprocessing Pipeline**: Custom regex filtering stripping HTML entities, URLs, length boundaries (5–500 chars), emoji-only noise, and near-duplicates.

### 2. Multi-Stage NLP Engine
* **Noise Reduction (Zero-Shot Bouncer)**: `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` identifies and discards irrelevant channel noise with a confidence threshold >= 0.70.
* **Sentiment Classification**: Fine-tuned `karlsefni/gadgetsense-modernbert` classifies product-specific feedback (Positive, Neutral, Negative) in GPU-accelerated batches (Batch Size = 64).
* **Logarithmic Weighted Scoring**: Calculates product index based on video view counts:

  Score = ((Sum(Sentiment_i * log10(Views_i + 1)) / Sum(log10(Views_i + 1))) + 1) / 2 * 100

* **Insight Generation (LLM Synthesis)**: `google/gemma-3-1b-it` (NF4 BitsAndBytes 4-bit quantization) generates structured JSON outputs covering key product highlights, critiques, and summary verdicts.

---

## 📂 Repository Structure

gadgetsense/
├── app.py              # FastAPI server backend + Gradio UI frontend entry point
├── sentiment.py        # NLP pipeline (Zero-shot filter, ModernBERT classifier, Gemma-3 synthesis)
├── youtube_service.py  # Asynchronous Piped/YouTube API scraper & text sanitization engine
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation

---

## 🛠️ Installation & Setup Guide

### Prerequisites

* Python **3.10+**
* NVIDIA GPU with CUDA support recommended (e.g., T4/V100/A100).
* A Hugging Face user access token.

---

### Option 1: Quickstart in Google Colab (Free GPU)

1. **Mount Drive & Navigate to Workspace**:
   from google.colab import drive
   drive.mount('/content/drive')

   %cd /content/drive/MyDrive/Colab_Notebooks/gadgetsense-master

2. **Install Dependencies**:
   !pip install -q gradio fastapi uvicorn transformers torch bitsandbytes accelerate httpx

3. **Authenticate Hugging Face**:
   from huggingface_hub import login
   login(token="YOUR_HUGGINGFACE_TOKEN")

4. **Launch Application**:
   !python app.py

---

### Option 2: Local Environment Setup

1. **Clone Repository & Set Up Virtual Environment**:
   git clone https://github.com/YOUR_USERNAME/gadgetsense.git
   cd gadgetsense
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. **Install Dependencies**:
   pip install -r requirements.txt

3. **Log into Hugging Face CLI**:
   huggingface-cli login

4. **Execute**:
   python app.py
   # Access UI at http://localhost:7860

---

## 🚀 API Specification

GadgetSense provides production-ready REST API endpoints alongside its web interface.

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

Response:
{
  "score": 85,
  "total_comments": 350,
  "positive_count": 260,
  "neutral_count": 50,
  "negative_count": 40,
  "pros": ["Active Noise Cancellation", "Sound Stage Quality", "Comfortable Fit"],
  "cons": ["Non-Folding Hinge Design", "High Launch Price"],
  "verdict": "BUY",
  "verdict_summary": "Highly recommended for ANC quality and audio clarity...",
  "videos": [...]
}

---
