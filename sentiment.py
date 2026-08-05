"""
sentiment.py — Sentiment analysis pipeline using google/gemma-3-1b-it (4-bit quantized).

Provides:
  - Batch comment classification (positive / neutral / negative)
  - Structured report generation (pros, cons, buyer verdict)
  - Weighted sentiment score (0–100)
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, pipeline

from youtube_service import CommentData, VideoWithComments

logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-3-1b-it"
BATCH_SIZE = 20  # comments per classification prompt


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ClassifiedComment:
    text: str
    sentiment: str  # "POSITIVE", "NEUTRAL", "NEGATIVE"
    video_id: str = ""
    confidence: float = 0.0


@dataclass
class SentimentReport:
    score: int  # 0–100
    total_comments: int
    positive_count: int
    neutral_count: int
    negative_count: int
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    verdict: str = ""  # "BUY", "WAIT", "AVOID"
    verdict_summary: str = ""
    classified_comments: list[ClassifiedComment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None

_bert_model = None
_bert_tokenizer = None

_zero_shot = None


def _load_zero_shot():
    """Lazy-load ModernBERT zero-shot classifier."""
    global _zero_shot
    if _zero_shot is not None:
        return

    logger.info("Loading ModernBERT zero-shot classifier...")
    _zero_shot = pipeline(
        "zero-shot-classification", 
        model="MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info("ModernBERT zero-shot loaded successfully.")


def _load_bert():
    """Lazy-load ModernBERT."""
    global _bert_model, _bert_tokenizer
    if _bert_model is not None:
        return

    logger.info("Loading ModernBERT sequence classifier...")
    model_id = "karlsefni/gadgetsense-modernbert"
    _bert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    _bert_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    _bert_model.eval()
    if torch.cuda.is_available():
        _bert_model.to("cuda")
    logger.info("ModernBERT loaded successfully.")


def _load_model():
    """Lazy-load model."""
    global _model, _tokenizer
    if _model is not None:
        return

    logger.info("Loading %s with 4-bit quantization...", MODEL_ID)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    logger.info("Model loaded successfully.")


def _generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate text from a prompt using the loaded model."""
    _load_model()

    messages = [{"role": "user", "content": prompt}]
    out = _tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **out,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0][out.input_ids.shape[1]:]
    return _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict | list | None:
    """Try to extract a JSON object or array from model output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
        r"(\{[\s\S]*\})",
        r"(\[[\s\S]*\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = """You are a sentiment classifier. Classify each numbered comment about a product as POSITIVE, NEUTRAL, or NEGATIVE.

Return ONLY a JSON array of objects with "id" and "sentiment" keys. No explanation.

Example output:
[{{"id": 1, "sentiment": "POSITIVE"}}, {{"id": 2, "sentiment": "NEGATIVE"}}]

Comments:
{comments}"""


def _classify_batch(comments: list[CommentData]) -> list[ClassifiedComment]:
    """Classify a batch of comments using the model."""
    numbered = "\n".join(f"{i+1}. {c.text}" for i, c in enumerate(comments))
    prompt = _CLASSIFY_PROMPT.format(comments=numbered)

    raw_output = _generate(prompt, max_new_tokens=512)
    parsed = _extract_json(raw_output)

    results: list[ClassifiedComment] = []

    if isinstance(parsed, list):
        # Map parsed results to comments
        sentiment_map: dict[int, str] = {}
        for item in parsed:
            if isinstance(item, dict):
                idx = item.get("id", 0)
                sent = str(item.get("sentiment", "NEUTRAL")).upper()
                if sent in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
                    sentiment_map[idx] = sent

        for i, c in enumerate(comments):
            sentiment = sentiment_map.get(i + 1, "NEUTRAL")
            results.append(ClassifiedComment(text=c.text, sentiment=sentiment, video_id=c.video_id))
    else:
        # Fallback: keyword-based classification
        logger.warning("JSON parse failed for batch, using keyword fallback. Raw: %s", raw_output[:200])
        results = _keyword_classify(comments)

    return results


def _keyword_classify(comments: list[CommentData]) -> list[ClassifiedComment]:
    """Simple keyword-based fallback classifier."""
    positive_words = {
        "love", "great", "amazing", "excellent", "awesome", "fantastic", "best",
        "perfect", "wonderful", "impressive", "recommend", "worth", "good", "nice",
        "happy", "solid", "beautiful", "premium", "superb", "outstanding",
    }
    negative_words = {
        "hate", "terrible", "awful", "worst", "bad", "horrible", "waste",
        "disappointed", "poor", "broken", "useless", "cheap", "overpriced",
        "regret", "avoid", "garbage", "junk", "defective", "sucks", "annoying",
    }

    results: list[ClassifiedComment] = []
    for c in comments:
        words = set(c.text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            sentiment = "POSITIVE"
        elif neg_count > pos_count:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        results.append(ClassifiedComment(text=c.text, sentiment=sentiment, video_id=c.video_id))
    return results


def classify_all_comments(videos_with_comments: list[VideoWithComments]) -> list[ClassifiedComment]:
    """Classify all comments across all videos in batches using ModernBERT."""
    all_comments: list[CommentData] = []
    for vwc in videos_with_comments:
        all_comments.extend(vwc.comments)

    if not all_comments:
        return []

    # --- Stage 3: Zero-shot Bouncer ---
    _load_zero_shot()
    logger.info("Running zero-shot filter on %d comments...", len(all_comments))
    
    candidate_labels = [
        "smartphone hardware specs, battery, camera, and device performance",
        "youtube video feedback, channel praise, or creator editing",
        "shipping, delivery, and customer service complaints"
    ]
    target_label = candidate_labels[0]
    
    texts_to_filter = [c.text for c in all_comments]
    zs_results = _zero_shot(texts_to_filter, candidate_labels, batch_size=32)
    
    filtered_comments = []
    for c, res in zip(all_comments, zs_results):
        if res["labels"][0] == target_label and res["scores"][0] >= 0.7:
            filtered_comments.append(c)
            
    logger.info("Zero-shot filter kept %d/%d product-focused comments.", len(filtered_comments), len(all_comments))

    if not filtered_comments:
        return []

    # --- Stage 4: Sentiment Engine ---
    _load_bert()
    device = _bert_model.device

    logger.info("Classifying %d comments using ModernBERT in batches...", len(filtered_comments))
    classified: list[ClassifiedComment] = []
    batch_size = 64

    for i in range(0, len(filtered_comments), batch_size):
        batch = filtered_comments[i : i + batch_size]
        texts = [c.text for c in batch]

        inputs = _bert_tokenizer(
            texts, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = _bert_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for j, c in enumerate(batch):
            sentiment_probs = probs[j]
            confidence, predicted_id = torch.max(sentiment_probs, dim=-1)
            
            # Map predicted ID back to LABEL (0: negative, 1: neutral, 2: positive)
            label = _bert_model.config.id2label[predicted_id.item()].upper()

            classified.append(
                ClassifiedComment(
                    text=c.text, 
                    sentiment=label, 
                    video_id=c.video_id,
                    confidence=confidence.item() * 100
                )
            )

        if (i // batch_size + 1) % 10 == 0:
            logger.info("  Classified batch %d/%d", i // batch_size + 1, math.ceil(len(filtered_comments) / batch_size))

    return classified


# ---------------------------------------------------------------------------
# Weighted score calculation
# ---------------------------------------------------------------------------
def calculate_weighted_score(
    classified: list[ClassifiedComment], videos: list[VideoWithComments]
) -> int:
    """
    Calculate weighted sentiment score (0–100).
    Weight = log10(views + 1) per video.
    Positive=+1, Neutral=0, Negative=−1.
    """
    # Build view count map
    view_map: dict[str, int] = {}
    for vwc in videos:
        view_map[vwc.video.video_id] = vwc.video.views

    weighted_sum = 0.0
    weight_total = 0.0

    for c in classified:
        views = view_map.get(c.video_id, 1)
        weight = math.log10(views + 1)

        if c.sentiment == "POSITIVE":
            score_val = 1.0
        elif c.sentiment == "NEGATIVE":
            score_val = -1.0
        else:
            score_val = 0.0

        weighted_sum += score_val * weight
        weight_total += weight

    if weight_total == 0:
        return 50  # neutral default

    raw = weighted_sum / weight_total  # range [-1, 1]
    return round((raw + 1) / 2 * 100)


# ---------------------------------------------------------------------------
# Batched report generation prompts (one task per LLM call)
# ---------------------------------------------------------------------------
_VERDICT_PROMPT = """You are a product analyst. Based on the user comments below about the '{gadget}', write a 2-3 sentence overall verdict summarizing what users think about this product.

POSITIVE COMMENTS:
{positive_samples}

NEGATIVE COMMENTS:
{negative_samples}

Write ONLY 2-3 sentences. No bullet points, no JSON. Just the verdict paragraph:"""

_BULLET_PROMPT = """Identify the 5 main {sentiment_label} themes about the '{gadget}' from the comments below.

Rules:
- Write each theme as a short label (3-8 words).
- NEVER quote or copy from comments. Extract the underlying THEME only.
- All 5 themes MUST be different. Do NOT repeat the same topic.
- Cover 5 DISTINCT aspects such as: design, performance, display, price, software, features, accessories, etc.

COMMENTS:
{samples}

Return a JSON array of exactly 5 UNIQUE theme strings.
JSON array:"""


def _generate_verdict(gadget: str, pos_samples: str, neg_samples: str) -> str:
    """Step 1: Generate a verdict summary."""
    prompt = _VERDICT_PROMPT.format(
        gadget=gadget,
        positive_samples=pos_samples,
        negative_samples=neg_samples,
    )
    return _generate(prompt, max_new_tokens=256).strip()


def _generate_bullets(gadget: str, samples: str, sentiment_label: str) -> list[str]:
    """Step 2/3: Generate bullet point summaries for one sentiment category."""
    prompt = _BULLET_PROMPT.format(
        gadget=gadget,
        samples=samples,
        sentiment_label=sentiment_label,
    )
    raw = _generate(prompt, max_new_tokens=256)
    parsed = _extract_json(raw)

    if isinstance(parsed, list):
        return [str(item) for item in parsed][:5]

    # Fallback: try to extract lines that look like bullet points
    lines = [line.strip().lstrip("-•*").strip().strip('"').strip("'") for line in raw.split("\n") if line.strip()]
    if lines:
        return lines[:5]

    logger.warning("Bullet generation failed for %s. Raw: %s", sentiment_label, raw[:200])
    return []


def generate_report(
    gadget: str,
    classified: list[ClassifiedComment],
    score: int,
) -> SentimentReport:
    """Generate the full sentiment report using 3 focused LLM calls."""
    pos = [c for c in classified if c.sentiment == "POSITIVE"]
    neu = [c for c in classified if c.sentiment == "NEUTRAL"]
    neg = [c for c in classified if c.sentiment == "NEGATIVE"]

    # Sort to prioritize highest confidence comments
    pos_sorted = sorted(pos, key=lambda x: x.confidence, reverse=True)
    neg_sorted = sorted(neg, key=lambda x: x.confidence, reverse=True)

    # Sample top 15 comments, truncated to keep prompts compact
    top_pos = pos_sorted[:15]
    top_neg = neg_sorted[:15]

    def _truncate(text: str, limit: int = 250) -> str:
        return text[:limit] + "..." if len(text) > limit else text

    pos_samples = "\n".join(f"- {_truncate(c.text)}" for c in top_pos) or "- None"
    neg_samples = "\n".join(f"- {_truncate(c.text)}" for c in top_neg) or "- None"

    # --- Step 1: Generate verdict summary ---
    logger.info("Generating verdict summary...")
    verdict_text = _generate_verdict(gadget, pos_samples, neg_samples)

    # --- Step 2: Generate pro bullet points ---
    logger.info("Generating pro highlights...")
    pros = _generate_bullets(gadget, pos_samples, "POSITIVE")

    # --- Step 3: Generate con bullet points ---
    logger.info("Generating con highlights...")
    cons = _generate_bullets(gadget, neg_samples, "NEGATIVE")

    # Calculate verdict deterministically from score
    if score >= 65:
        deterministic_verdict = "BUY"
    elif score >= 40:
        deterministic_verdict = "WAIT"
    else:
        deterministic_verdict = "AVOID"

    report = SentimentReport(
        score=score,
        total_comments=len(classified),
        positive_count=len(pos),
        neutral_count=len(neu),
        negative_count=len(neg),
        verdict=deterministic_verdict,
        verdict_summary=verdict_text,
        pros=pros,
        cons=cons,
        classified_comments=classified,
    )

    # Fallbacks
    if not report.pros:
        report.pros = ["Users mentioned positive aspects, but highlights could not be summarized."]
    if not report.cons:
        report.cons = ["Users mentioned negative aspects, but highlights could not be summarized."]
    if not report.verdict_summary:
        report.verdict_summary = f"Based on {len(classified)} comments across YouTube reviews, this product scores {score}/100."

    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def analyze_sentiment(
    gadget: str, videos_with_comments: list[VideoWithComments]
) -> SentimentReport:
    """
    Full sentiment analysis pipeline:
    1. Classify all comments
    2. Calculate weighted score
    3. Generate report
    """
    classified = classify_all_comments(videos_with_comments)

    if not classified:
        return SentimentReport(
            score=50,
            total_comments=0,
            positive_count=0,
            neutral_count=0,
            negative_count=0,
            verdict="WAIT",
            verdict_summary="No comments were found to analyze.",
        )

    score = calculate_weighted_score(classified, videos_with_comments)
    report = generate_report(gadget, classified, score)

    logger.info(
        "Analysis complete: score=%d, verdict=%s, %d comments (%d+/%d~/%d-)",
        report.score, report.verdict, report.total_comments,
        report.positive_count, report.neutral_count, report.negative_count,
    )
    return report
