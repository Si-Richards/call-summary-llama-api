import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm_client import (
    LLMConfig,
    OpenAICompatLLM,
    chunk_text,
    clean_text,
    extract_json_object,
)

app = FastAPI(title="Call Summary API (Llama via vLLM)", version="1.1.0")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
TIMEOUT = env_float("REQUEST_TIMEOUT_SECS", 180.0)
MAX_INPUT_CHARS = env_int("MAX_INPUT_CHARS", 300000)
CHUNK_CHARS = env_int("CHUNK_CHARS", 12000)

llm = OpenAICompatLLM(LLMConfig(base_url=LLM_BASE_URL, model=LLM_MODEL, timeout_secs=TIMEOUT))


class CallSummaryRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Full call transcript (plain text)")
    agent: Optional[str] = Field(None, description="Agent name")
    customer: Optional[str] = Field(None, description="Customer name")
    call_reason: Optional[str] = Field(None, description="What the call is about (if known)")
    style: str = Field("bullets", pattern="^(bullets|short|detailed)$")
    max_tokens: int = Field(700, ge=200, le=2000)
    temperature: float = Field(0.2, ge=0.0, le=1.2)


class SentimentScore(BaseModel):
    label: str = "neutral"   # positive|neutral|negative
    score: float = 0.0       # -1.0..1.0


class SentimentTimelinePoint(BaseModel):
    index: int
    start_char: int
    end_char: int
    overall: SentimentScore = SentimentScore()
    customer: SentimentScore = SentimentScore()
    agent: SentimentScore = SentimentScore()
    drivers: List[str] = []
    escalation_risk: str = "low"  # low|medium|high


class CallSummaryResponse(BaseModel):
    summary: str
    key_points: List[str] = []
    decisions: List[str] = []
    action_items: List[str] = []
    risks: List[str] = []
    follow_ups: List[str] = []

    sentiment_overall: SentimentScore = SentimentScore()
    sentiment_customer: SentimentScore = SentimentScore()
    sentiment_agent: SentimentScore = SentimentScore()
    sentiment_trend: str = "steady"
    sentiment_drivers: List[str] = []
    escalation_risk: str = "low"

    sentiment_timeline: List[SentimentTimelinePoint] = []

    model: str


def strip_ivr(transcript: str) -> str:
    """
    Drop IVR/menu lines if you have a dedicated IVR speaker label.
    This is optional, but helps sentiment accuracy.
    """
    lines = []
    for line in transcript.splitlines():
        if line.strip().startswith("[SPEAKER_00]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _score_from(obj: Any) -> SentimentScore:
    if isinstance(obj, dict):
        try:
            return SentimentScore(
                label=str(obj.get("label", "neutral")),
                score=float(obj.get("score", 0.0)),
            )
        except Exception:
            return SentimentScore()
    return SentimentScore()


def build_messages(req: CallSummaryRequest, text: str) -> List[Dict[str, str]]:
    style_hint = {
        "short": "Return a short paragraph summary (3–5 sentences) + a short bullet list of action items.",
        "detailed": "Return a structured summary with headings and bullets, include decisions, actions, and any risks.",
        "bullets": "Return a concise bullet-point summary. Capture decisions and actions.",
    }[req.style]

    meta_bits: List[str] = []
    if req.agent:
        meta_bits.append(f"Agent: {req.agent}")
    if req.customer:
        meta_bits.append(f"Customer: {req.customer}")
    if req.call_reason:
        meta_bits.append(f"Call reason: {req.call_reason}")

    meta = "\n".join(meta_bits).strip()
    meta_block = f"Metadata:\n{meta}\n" if meta else ""

    system = (
        "You are a call summarisation assistant for a telecoms service desk. "
        "Be accurate. Do not invent facts. If something is unclear, say 'unclear'. "
        "Keep names, numbers, dates exactly as stated."
    )

    user = f"""
{style_hint}

Return output as JSON with these keys:
- summary (string)
- key_points (array of strings)
- decisions (array of strings)
- action_items (array of strings)
- risks (array of strings)
- follow_ups (array of strings)

Sentiment keys:
- sentiment_overall: {{ "label": "positive|neutral|negative", "score": -1.0..1.0 }}
- sentiment_customer: {{ "label": "positive|neutral|negative", "score": -1.0..1.0 }}
- sentiment_agent: {{ "label": "positive|neutral|negative", "score": -1.0..1.0 }}
- sentiment_trend: "improving|steady|worsening"
- sentiment_drivers (array of strings)
- escalation_risk: "low|medium|high"

Rules:
- Do not invent facts.
- If speaker roles are unclear, treat SPEAKER_01 as agent and SPEAKER_02 as customer (best effort).
- Keep it concise.

{meta_block}TRANSCRIPT:
{text}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def summarize_one_pass(req: CallSummaryRequest, text: str) -> Dict[str, Any]:
    content = llm.chat(
        build_messages(req, text),
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    data = extract_json_object(content)
    if data is None:
        # Fallback: raw summary only
        return {
            "summary": content,
            "key_points": [],
            "decisions": [],
            "action_items": [],
            "risks": [],
            "follow_ups": [],
            "sentiment_overall": SentimentScore().model_dump(),
            "sentiment_customer": SentimentScore().model_dump(),
            "sentiment_agent": SentimentScore().model_dump(),
            "sentiment_trend": "steady",
            "sentiment_drivers": [],
            "escalation_risk": "low",
        }

    return {
        "summary": clean_text(str(data.get("summary", ""))),
        "key_points": data.get("key_points", []) or [],
        "decisions": data.get("decisions", []) or [],
        "action_items": data.get("action_items", []) or [],
        "risks": data.get("risks", []) or [],
        "follow_ups": data.get("follow_ups", []) or [],

        "sentiment_overall": _score_from(data.get("sentiment_overall")).model_dump(),
        "sentiment_customer": _score_from(data.get("sentiment_customer")).model_dump(),
        "sentiment_agent": _score_from(data.get("sentiment_agent")).model_dump(),
        "sentiment_trend": str(data.get("sentiment_trend", "steady") or "steady"),
        "sentiment_drivers": data.get("sentiment_drivers", []) or [],
        "escalation_risk": str(data.get("escalation_risk", "low") or "low"),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "chunk_chars": CHUNK_CHARS,
    }


@app.post("/summarize-call", response_model=CallSummaryResponse)
def summarize_call(req: CallSummaryRequest):
    if len(req.transcript) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail=f"Transcript too large (limit: {MAX_INPUT_CHARS} chars)")

    transcript = clean_text(strip_ivr(req.transcript))
    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcript")

    chunks = chunk_text(transcript, CHUNK_CHARS)

    # Single-chunk = one timeline point
    if len(chunks) == 1:
        out = summarize_one_pass(req, chunks[0])

        timeline = [
            SentimentTimelinePoint(
                index=0,
                start_char=0,
                end_char=len(chunks[0]),
                overall=_score_from(out.get("sentiment_overall")),
                customer=_score_from(out.get("sentiment_customer")),
                agent=_score_from(out.get("sentiment_agent")),
                drivers=out.get("sentiment_drivers", []) or [],
                escalation_risk=str(out.get("escalation_risk", "low") or "low"),
            )
        ]

        return CallSummaryResponse(
            **out,
            model=LLM_MODEL,
            sentiment_timeline=timeline,
        )

    # Map: summarise each chunk with a smaller budget, capture sentiment timeline
    per_chunk_req = req.model_copy(update={"max_tokens": min(req.max_tokens, 500)})

    per_chunk: List[Dict[str, Any]] = []
    timeline: List[SentimentTimelinePoint] = []

    cursor = 0
    for i, c in enumerate(chunks):
        start = cursor
        end = cursor + len(c)
        cursor = end + 1  # approximate separator

        out_i = summarize_one_pass(per_chunk_req, c)
        per_chunk.append(out_i)

        timeline.append(
            SentimentTimelinePoint(
                index=i,
                start_char=start,
                end_char=end,
                overall=_score_from(out_i.get("sentiment_overall")),
                customer=_score_from(out_i.get("sentiment_customer")),
                agent=_score_from(out_i.get("sentiment_agent")),
                drivers=out_i.get("sentiment_drivers", []) or [],
                escalation_risk=str(out_i.get("escalation_risk", "low") or "low"),
            )
        )

    # Reduce: combine chunk summaries into one final summary
    combined = "\n\n".join(
        (
            f"CHUNK {i+1}:\n{pc.get('summary','')}\n"
            f"Key points: {pc.get('key_points',[])}\n"
            f"Decisions: {pc.get('decisions',[])}\n"
            f"Actions: {pc.get('action_items',[])}\n"
            f"Risks: {pc.get('risks',[])}\n"
            f"Follow-ups: {pc.get('follow_ups',[])}\n"
            f"Sentiment overall: {pc.get('sentiment_overall',{})}\n"
            f"Sentiment customer: {pc.get('sentiment_customer',{})}\n"
            f"Sentiment agent: {pc.get('sentiment_agent',{})}\n"
            f"Sentiment drivers: {pc.get('sentiment_drivers',[])}\n"
            f"Escalation risk: {pc.get('escalation_risk','low')}"
        )
        for i, pc in enumerate(per_chunk)
    )

    final_req = req.model_copy(
        update={"call_reason": req.call_reason or "Combine chunk summaries into one coherent call summary."}
    )
    out = summarize_one_pass(final_req, combined)

    # Compute a simple trend from customer score first -> last
    try:
        first = timeline[0].customer.score
        last = timeline[-1].customer.score
        if last - first > 0.15:
            out["sentiment_trend"] = "improving"
        elif first - last > 0.15:
            out["sentiment_trend"] = "worsening"
        else:
            out["sentiment_trend"] = out.get("sentiment_trend", "steady") or "steady"
    except Exception:
        pass

    return CallSummaryResponse(
        **out,
        model=LLM_MODEL,
        sentiment_timeline=timeline,
    )
