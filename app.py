"""
app.py — GadgetSense entry point: FastAPI backend + Gradio frontend.

Run:  python app.py
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sentiment import SentimentReport, analyze_sentiment
from youtube_service import YouTubeService, VideoWithComments

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gadgetsense")

# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------
_yt_service: YouTubeService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _yt_service
    _yt_service = YouTubeService()
    logger.info("GadgetSense started.")
    yield
    await _yt_service.close()
    logger.info("GadgetSense shut down.")


fastapi_app = FastAPI(title="GadgetSense API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@fastapi_app.get("/api/health")
async def health():
    return {"status": "ok"}


@fastapi_app.post("/api/analyze")
async def api_analyze(payload: dict):
    gadget = payload.get("gadget", "").strip()
    if not gadget:
        return JSONResponse({"error": "gadget name is required"}, status_code=400)
    try:
        report = await _run_analysis(gadget)
        return _report_to_dict(report)
    except Exception as exc:
        logger.exception("Analysis failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Core analysis logic (shared by API and Gradio)
# ---------------------------------------------------------------------------
async def _run_analysis(gadget: str) -> tuple[SentimentReport, list[VideoWithComments]]:
    """Run full pipeline: search → comments → sentiment."""
    global _yt_service
    if _yt_service is None:
        _yt_service = YouTubeService()

    videos_with_comments = await _yt_service.search_and_collect(gadget)

    if not videos_with_comments:
        raise ValueError(f"No review videos found for '{gadget}'. Try a different product name.")

    total = sum(len(vwc.comments) for vwc in videos_with_comments)
    if total == 0:
        raise ValueError(f"Found videos but no comments for '{gadget}'. Try a more popular product.")

    # Run sentiment analysis (CPU/GPU intensive, run in thread)
    report = await asyncio.to_thread(analyze_sentiment, gadget, videos_with_comments)
    return report, videos_with_comments


def _report_to_dict(result: tuple[SentimentReport, list[VideoWithComments]]) -> dict:
    report, videos_with_comments = result
    return {
        "score": report.score,
        "total_comments": report.total_comments,
        "positive_count": report.positive_count,
        "neutral_count": report.neutral_count,
        "negative_count": report.negative_count,
        "pros": report.pros,
        "cons": report.cons,
        "verdict": report.verdict,
        "verdict_summary": report.verdict_summary,
        "videos": [
            {
                "title": vwc.video.title,
                "channel": vwc.video.channel,
                "views": vwc.video.views,
                "url": vwc.video.url,
                "thumbnail": vwc.video.thumbnail,
                "comment_count": len(vwc.comments),
            }
            for vwc in videos_with_comments
        ],
    }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
VERDICT_EMOJIS = {"BUY": "✅", "WAIT": "⏳", "AVOID": "❌"}
VERDICT_COLORS = {"BUY": "#22c55e", "WAIT": "#f59e0b", "AVOID": "#ef4444"}


def _build_score_html(score: int) -> str:
    """Generate an HTML gauge for the sentiment score."""
    if score >= 65:
        color = "#10b981"  # Emerald 500
        gradient = "url(#emeraldGrad)"
        label = "Positive"
        subtext = "Favorable Market Sentiment"
    elif score >= 40:
        color = "#f59e0b"  # Amber 500
        gradient = "url(#amberGrad)"
        label = "Mixed"
        subtext = "Divided Market Sentiment"
    else:
        color = "#ef4444"  # Red 500
        gradient = "url(#redGrad)"
        label = "Negative"
        subtext = "Unfavorable Market Sentiment"

    # ARC Math: Dash array for a 180-degree semi-circle
    # circumference of r=70 is 2*pi*70 = 439.8
    # A semi-circle is 220
    fill = (score / 100) * 220
    empty = 440 - fill

    return f"""
    <div class="stat-card score-card">
        <div class="score-header">
            <h3 class="card-title">Sentiment Index</h3>
            <span class="card-subtitle">Aggregate Score</span>
        </div>
        <div style="position:relative; width:100%; max-width:240px; height:140px; margin:24px auto 0;">
            <svg viewBox="0 0 160 110" style="width:100%; height:100%; overflow:visible;">
                <defs>
                    <linearGradient id="emeraldGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#34d399" />
                        <stop offset="100%" stop-color="#059669" />
                    </linearGradient>
                    <linearGradient id="amberGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#fbbf24" />
                        <stop offset="100%" stop-color="#d97706" />
                    </linearGradient>
                    <linearGradient id="redGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#f87171" />
                        <stop offset="100%" stop-color="#dc2626" />
                    </linearGradient>
                </defs>
                <!-- Background Arc -->
                <path d="M 20 90 A 60 60 0 0 1 140 90" fill="none" class="gauge-bg" stroke-width="16" stroke-linecap="round"/>
                <!-- Foreground Arc -->
                <path d="M 20 90 A 60 60 0 0 1 140 90" fill="none" stroke="{gradient}" stroke-width="16"
                    stroke-dasharray="{fill} {empty}"
                    stroke-linecap="round" style="transition: stroke-dasharray 1.5s cubic-bezier(0.4, 0, 0.2, 1);"/>
            </svg>
            <div class="gauge-text">
                <span style="color: var(--body-text-color); font-size:4rem;">{score}</span>
            </div>
        </div>
        <div class="score-footer">
            <div class="score-label" style="color:{color};">{label}</div>
            <div class="score-subtext">{subtext}</div>
        </div>
    </div>
    """


def _build_verdict_html(verdict: str, summary: str) -> str:
    color = VERDICT_COLORS.get(verdict, "#94a3b8")
    bg_color = f"{color}10"  # Very subtle background

    return f"""
    <div class="stat-card verdict-card" style="border-top: 4px solid {color}; background-color: var(--background-fill-secondary);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
            <div>
                <h3 class="card-title">Recommendation</h3>
                <span class="card-subtitle">AI Final Verdict</span>
            </div>
            <div class="verdict-badge" style="background-color: {bg_color}; color: {color}; border: 1px solid {color}30;">
                {verdict}
            </div>
        </div>
        <div class="verdict-summary">
            "{summary}"
        </div>
    </div>
    """


def _build_breakdown_html(pos: int, neu: int, neg: int, total: int) -> str:
    if total == 0:
        return "<p class='empty-text'>No comments analyzed.</p>"
    pp = round(pos / total * 100)
    np_ = round(neu / total * 100)
    ngp = round(neg / total * 100)
    return f"""
    <div class="stat-card">
        <div class="score-header" style="margin-bottom:24px;">
            <h3 class="card-title">Response Distribution</h3>
            <span class="card-subtitle">{total:,} verified sources</span>
        </div>
        
        <div class="breakdown-metrics">
            <div class="metric-col">
                <span class="metric-dot" style="background:#10b981;"></span>
                <div class="metric-data">
                    <span class="metric-val">{pp}%</span>
                    <span class="metric-count">{pos:,} Pos</span>
                </div>
            </div>
            <div class="metric-col">
                <span class="metric-dot" style="background:#f59e0b;"></span>
                <div class="metric-data">
                    <span class="metric-val">{np_}%</span>
                    <span class="metric-count">{neu:,} Neu</span>
                </div>
            </div>
            <div class="metric-col">
                <span class="metric-dot" style="background:#ef4444;"></span>
                <div class="metric-data">
                    <span class="metric-val">{ngp}%</span>
                    <span class="metric-count">{neg:,} Neg</span>
                </div>
            </div>
        </div>

        <div class="stacked-bar-container">
            <div style="width:{pp}%; background:linear-gradient(90deg, #34d399, #10b981);"></div>
            <div style="width:{np_}%; background:linear-gradient(90deg, #fbbf24, #f59e0b);"></div>
            <div style="width:{ngp}%; background:linear-gradient(90deg, #f87171, #ef4444);"></div>
        </div>
    </div>
    """


def _build_pros_cons_md(pros: list[str], cons: list[str]) -> str:
    md = """
<div class="insights-container">
    <div class="insight-column pros-column">
        <h4 class="insight-title"><span class="insight-icon">✅</span> Highlights & Strengths</h4>
        <ul class="insight-list">
"""
    for p in pros:
        md += f"            <li>{p}</li>\n"
    md += """
        </ul>
    </div>
    <div class="insight-column cons-column">
        <h4 class="insight-title"><span class="insight-icon">⚠️</span> Critiques & Weaknesses</h4>
        <ul class="insight-list">
"""
    for c in cons:
        md += f"            <li>{c}</li>\n"
    md += """
        </ul>
    </div>
</div>
"""
    return md


def _build_videos_md(videos: list[VideoWithComments]) -> str:
    if not videos:
        return "*No sources found.*"
    md = ""
    for i, vwc in enumerate(videos, 1):
        v = vwc.video
        views_str = f"{v.views:,}" if v.views else "N/A"
        md += f"#### {i}. [{v.title}]({v.url})\n"
        md += f"**{v.channel}** · {views_str} views · {len(vwc.comments)} comments analyzed\n\n---\n\n"
    return md


def _progress_html(title: str, subtext: str) -> str:
    return f"""
    <div class="stat-card" style="text-align:center; display:flex; flex-direction:column; align-items:center; justify-content:center; padding: 48px 24px;">
        <svg style="margin-bottom:24px; animation:spin 1s linear infinite;" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--body-text-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        <h3 class="card-title" style="margin-bottom:8px !important; color:var(--body-text-color); font-size:18px;">{title}</h3>
        <span class="card-subtitle">{subtext}</span>
        <style>@keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}</style>
    </div>
    """


async def gradio_analyze(gadget: str):
    """Gradio handler — yields loading state steps then final outputs."""
    gadget = gadget.strip()
    if not gadget:
        raise gr.Error("Please enter a product name.")

    # Show loader, hide results
    yield (
        gr.update(value=_progress_html(f"Locating '{gadget}' data...", "Searching YouTube for product reviews..."), visible=True),
        gr.update(visible=False),
        "", "", "", "", ""
    )

    try:
        global _yt_service
        if _yt_service is None:
            _yt_service = YouTubeService()
        
        videos_with_comments = await _yt_service.search_and_collect(gadget)
        if not videos_with_comments:
            raise ValueError(f"No review videos found for '{gadget}'. Try a different product name.")

        total = sum(len(vwc.comments) for vwc in videos_with_comments)
        if total == 0:
            raise ValueError(f"Found videos but no comments for '{gadget}'. Try a more popular product.")

        yield (
            gr.update(value=_progress_html(f"Analyzing {total:,} verified sources...", "Feeding viewer sentiments into the neural engine..."), visible=True),
            gr.update(visible=False),
            "", "", "", "", ""
        )

        from sentiment import analyze_sentiment
        report = await asyncio.to_thread(analyze_sentiment, gadget, videos_with_comments)

        yield (
            gr.update(value=_progress_html("Finalizing intelligence report...", "Synthesizing scores, pros, and cons..."), visible=True),
            gr.update(visible=False),
            "", "", "", "", ""
        )

        score_html = _build_score_html(report.score)
        verdict_html = _build_verdict_html(report.verdict, report.verdict_summary)
        breakdown_html = _build_breakdown_html(
            report.positive_count, report.neutral_count, report.negative_count, report.total_comments
        )
        pros_cons_html = _build_pros_cons_md(report.pros, report.cons)
        videos_md = _build_videos_md(videos_with_comments)

        # Hide loader, show results
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            score_html, verdict_html, breakdown_html, pros_cons_html, videos_md
        )

    except ValueError as e:
        yield gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""
        raise gr.Error(str(e))
    except Exception as e:
        logger.exception("Analysis failed in Gradio handler")
        yield gr.update(visible=False), gr.update(visible=False), "", "", "", "", ""
        raise gr.Error(f"Analysis failed: {e}")


# ---------------------------------------------------------------------------
# Build Gradio Blocks
# ---------------------------------------------------------------------------
css = """
/* Sophisticated Type & Colors */
body {
    background-color: var(--background-fill-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Base Card Styling */
.stat-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 4px 24px -6px rgba(0,0,0,0.03), 0 0 1px rgba(0,0,0,0.05);
    height: 100%;
    display: flex;
    flex-direction: column;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    box-shadow: 0 12px 32px -8px rgba(0,0,0,0.06), 0 0 1px rgba(0,0,0,0.05);
    transform: translateY(-2px);
}

/* Headers */
.score-header { display: flex; flex-direction: column; gap: 4px; }
.card-title {
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--body-text-color);
    margin: 0 !important;
}
.card-subtitle {
    font-size: 13px;
    font-weight: 500;
    color: var(--body-text-color-subdued);
}

/* Gauge specific */
.gauge-bg { stroke: var(--background-fill-primary); border: 1px solid var(--border-color-primary); }
.score-card { text-align: center; }
.score-card .score-header { text-align: left; }
.gauge-text {
    position: absolute;
    bottom: -5px; left: 50%;
    transform: translateX(-50%);
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1;
}
.score-footer { margin-top: 24px; text-align: center; }
.score-label {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.score-subtext {
    font-size: 13px;
    color: var(--body-text-color-subdued);
    font-weight: 500;
}

/* Verdict */
.verdict-card { justify-content: flex-start; }
.verdict-badge {
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.verdict-summary {
    font-size: 18px;
    line-height: 1.6;
    font-weight: 400;
    color: var(--body-text-color);
    font-style: italic;
    margin-top: auto;
    margin-bottom: auto;
}

/* Breakdown */
.breakdown-metrics {
    display: flex;
    justify-content: space-between;
    margin-bottom: 32px;
    margin-top: 16px;
}
.metric-col { display: flex; align-items: flex-start; gap: 10px; }
.metric-dot { width: 10px; height: 10px; border-radius: 5px; margin-top: 6px; }
.metric-data { display: flex; flex-direction: column; }
.metric-val { font-size: 22px; font-weight: 800; line-height: 1; margin-bottom: 4px; color: var(--body-text-color); }
.metric-count { font-size: 12px; font-weight: 600; color: var(--body-text-color-subdued); text-transform: uppercase; letter-spacing: 0.05em; }
.stacked-bar-container {
    background: var(--background-fill-primary);
    border-radius: 999px;
    overflow: hidden;
    height: 12px;
    display: flex;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
}

/* Pros and Cons Columns */
.insights-container {
    display: flex;
    gap: 24px;
    margin-top: 16px;
}
.insight-column {
    flex: 1;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 4px 24px -6px rgba(0,0,0,0.03);
}
.pros-column { border-top: 4px solid #10b981; }
.cons-column { border-top: 4px solid #ef4444; }
.insight-title {
    font-size: 18px !important;
    font-weight: 800 !important;
    margin: 0 0 24px 0 !important;
    color: var(--body-text-color) !important;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: none !important;
}
.insight-icon { font-size: 20px; }
.insight-list {
    list-style: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.insight-list li {
    font-size: 15px;
    line-height: 1.6;
    color: var(--body-text-color);
    margin-bottom: 16px;
    padding-left: 24px;
    position: relative;
    font-weight: 500;
}
.insight-list li::before {
    content: "•";
    color: var(--body-text-color-subdued);
    font-size: 20px;
    position: absolute;
    left: 4px;
    top: -2px;
}

/* Headings & Search */
.app-header {
    text-align: center;
    padding: 60px 0 40px;
}
.app-logo {
    font-size: 48px;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: var(--body-text-color);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}
.app-logo .accent {
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-tagline {
    font-size: 18px;
    color: var(--body-text-color-subdued);
    font-weight: 500;
    letter-spacing: -0.01em;
}

/* Hide Gradio default footer */
footer { display: none !important; }
"""

_gradio_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_text_color="var(--body-text-color)",
    border_color_primary="var(--border-color-primary)",
    block_background_fill="transparent",
    block_border_width="0px",
    input_background_fill="var(--background-fill-secondary)",
    input_border_color="var(--border-color-primary)",
    input_padding="20px",
    input_text_size="18px",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca, #6d28d9)",
    button_border_width="0px",
)

with gr.Blocks(title="GadgetSense AI", theme=_gradio_theme, css=css) as demo:
    
    with gr.Column(elem_classes="app-header"):
        gr.HTML("""
        <div class="app-logo">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="url(#logoGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <defs>
                    <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#4f46e5" />
                        <stop offset="100%" stop-color="#9333ea" />
                    </linearGradient>
                </defs>
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <path d="M11 8v6"></path>
                <path d="M8 11h6"></path>
            </svg>
            <span>Gadget<span class="accent">Sense</span></span>
        </div>
        <div class="app-tagline">Because the real review is always in the comments.</div>
        """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            gadget_input = gr.Textbox(
                placeholder="Search any product (e.g., MacBook Pro M3, Sony WH-1000XM5)...",
                lines=1,
                max_lines=1,
                show_label=False,
                container=True,
            )
        with gr.Column(scale=1, min_width=180):
            analyze_btn = gr.Button("Analyze Data", variant="primary", size="lg")

    gr.HTML('<div style="height: 40px;"></div>')
    
    status_output = gr.HTML()

    with gr.Group(visible=False) as results_container:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                score_output = gr.HTML()
            with gr.Column(scale=1):
                verdict_output = gr.HTML()
            with gr.Column(scale=1):
                breakdown_output = gr.HTML()

        gr.HTML('<div style="height: 16px;"></div>')

        with gr.Row():
            with gr.Column():
                pros_cons_output = gr.HTML()

        gr.HTML('<div style="height: 24px;"></div>')
        with gr.Accordion("Sources & Analyzed Raw Data", open=False):
            videos_output = gr.Markdown()

    # Wire events
    analyze_btn.click(
        fn=gradio_analyze,
        inputs=[gadget_input],
        outputs=[status_output, results_container, score_output, verdict_output, breakdown_output, pros_cons_output, videos_output],
        show_progress="hidden"
    )
    gadget_input.submit(
        fn=gradio_analyze,
        inputs=[gadget_input],
        outputs=[status_output, results_container, score_output, verdict_output, breakdown_output, pros_cons_output, videos_output],
        show_progress="hidden"
    )

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True, debug=True)
