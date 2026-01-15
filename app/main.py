import os
import re
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm_client import (
    LLMConfig,
    OpenAICompatLLM,
    clean_text,
    extract_json_object,
    estimate_tokens_rough,
    chunk_turns_by_token_budget,
)

app = FastAPI(title="Telephone Call Summary API (Llama via vLLM)", version="1.5.0")


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
MODEL_MAX_LEN = env_int("MODEL_MAX_LEN", 4096)   # MUST match vLLM --max-model-len
CACHE_SIZE = env_int("CACHE_SIZE", 512)

llm = OpenAICompatLLM(LLMConfig(base_url=LLM_BASE_URL, model=LLM_MODEL, timeout_secs=TIMEOUT))


# -------------------------
# Thread-safe LRU cache
# -------------------------
class LRUCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._lock = Lock()
        self._data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            v = self._data.get(key)
            if v is None:
                return None
            self._data.move_to_end(key)
            return v

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)


cache = LRUCache(CACHE_SIZE)


# -------------------------
# API Models
# -------------------------
class CallSummaryRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Full call transcript (plain text)")
    agent: Optional[str] = Field(None, description="Agent name")
    customer: Optional[str] = Field(None, description="Customer name")
    call_reason: Optional[str] = Field(None, description="What the call is about (if known)")
    style: str = Field("bullets", pattern="^(bullets|short|detailed)$")
    max_tokens: int = Field(600, ge=200, le=2000)
    temperature: float = Field(0.0, ge=0.0, le=1.2)


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
    escalation_risk: str = "low"


class CallSummaryResponse(BaseModel):
    summary: str
    key_points: List[str] = []

    sentiment_overall: SentimentScore = SentimentScore()
    sentiment_customer: SentimentScore = SentimentScore()
    sentiment_agent: SentimentScore = SentimentScore()

    # Trend is computed in code from timeline (no contradictions)
    sentiment_trend: str = "steady"

    sentiment_drivers: List[str] = []
    escalation_risk: str = "low"

    # Confidence + grounding
    confidence_summary: float = 0.6
    confidence_sentiment: float = 0.6
    evidence_quotes: List[str] = []

    sentiment_timeline: List[SentimentTimelinePoint] = []

    model: str


# -------------------------
# Preprocess transcript -> turns
# -------------------------
_TS_RE = re.compile(r"^\[\d{2}:\d{2}\.\d{2}\]\s*")  # [00:02.41]
_SPK_PAREN_RE = re.compile(r"^\((SPEAKER_\d+|UNKNOWN)\)\s*")
_SPK_BRACKET_RE = re.compile(r"^\[(SPEAKER_\d+)\]\s*")


def preprocess_transcript(raw: str) -> List[str]:
    """
    Returns a list of turns (lines), preserving boundaries for chunking.
    - Removes timestamps
    - Drops IVR SPEAKER_00 and UNKNOWN
    - Compresses speaker tags: SPEAKER_01->A:, SPEAKER_02->C:, else S##
    - Normalizes common ASR slips: CLS/PLS -> TLS, fofoam -> softphone, year links -> Yealink
    """
    turns: List[str] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue

        s = _TS_RE.sub("", s)

        speaker = None
        m = _SPK_PAREN_RE.match(s)
        if m:
            speaker = m.group(1)
            s = _SPK_PAREN_RE.sub("", s)
        else:
            m2 = _SPK_BRACKET_RE.match(s)
            if m2:
                speaker = m2.group(1)
                s = _SPK_BRACKET_RE.sub("", s)

        if speaker == "SPEAKER_00" or speaker == "UNKNOWN":
            continue

        # Normalize common ASR slips
        s = re.sub(r"\bCLS\b|\bPLS\b", "TLS", s)
        s = re.sub(r"\bfofoam\b", "softphone", s, flags=re.IGNORECASE)
        s = re.sub(r"\byear links\b", "Yealink", s, flags=re.IGNORECASE)

        s = clean_text(s)
        if not s:
            continue

        prefix = ""
        if speaker == "SPEAKER_01":
            prefix = "A: "
        elif speaker == "SPEAKER_02":
            prefix = "C: "
        elif speaker and speaker.startswith("SPEAKER_"):
            try:
                n = speaker.split("_", 1)[1].zfill(2)
                prefix = f"S{n}: "
            except Exception:
                prefix = ""

        turns.append(prefix + s)

    return turns


# -------------------------
# Sentiment enforcement
# -------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def label_from_score(score: float) -> str:
    score = float(score)
    if score <= -0.2:
        return "negative"
    if score >= 0.2:
        return "positive"
    return "neutral"


def normalize_sentiment(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"label": "neutral", "score": 0.0}
    score = clamp(obj.get("score", 0.0), -1.0, 1.0)
    return {"label": label_from_score(score), "score": score}


def normalize_escalation_risk(v: Any) -> str:
    v = str(v or "low").lower()
    return v if v in ("low", "medium", "high") else "low"


def compute_trend_from_timeline(timeline: List[SentimentTimelinePoint]) -> str:
    if not timeline:
        return "steady"
    first = timeline[0].customer.score
    last = timeline[-1].customer.score
    if last - first > 0.15:
        return "improving"
    if first - last > 0.15:
        return "worsening"
    return "steady"


# -------------------------
# Token budget adaptation
# -------------------------
def choose_limits(req: CallSummaryRequest, est_input_tokens: int) -> Dict[str, int]:
    if req.max_tokens <= 450 or est_input_tokens >= 2600:
        return {"max_kp": 5, "max_drv": 3, "max_quotes": 1}
    return {"max_kp": 8, "max_drv": 5, "max_quotes": 2}


# -------------------------
# Prompt
# -------------------------
def build_messages(req: CallSummaryRequest, text: str, limits: Dict[str, int]) -> List[Dict[str, str]]:
    style_hint = {
        "short": "Summarize in 3–5 short sentences.",
        "detailed": "Write a structured summary with very short headings and concise bullets.",
        "bullets": "Summarize as concise bullet points.",
    }[req.style]

    if req.max_tokens <= 450 and req.style != "bullets":
        style_hint = "Summarize as concise bullet points."

    meta_bits: List[str] = []
    if req.agent:
        meta_bits.append(f"Agent: {req.agent}")
    if req.customer:
        meta_bits.append(f"Customer: {req.customer}")
    if req.call_reason:
        meta_bits.append(f"Call reason: {req.call_reason}")
    meta = "\n".join(meta_bits).strip()
    meta_block = f"Metadata:\n{meta}\n" if meta else ""

    schema = f"""Return ONLY valid JSON with these keys:
- summary (string)
- key_points (array of strings, max {limits["max_kp"]} items)

Sentiment (score-based; label derived from score thresholds):
- sentiment_overall: {{ "score": -1.0..1.0 }}
- sentiment_customer: {{ "score": -1.0..1.0 }}
- sentiment_agent: {{ "score": -1.0..1.0 }}
- sentiment_drivers (array of strings, max {limits["max_drv"]} items)
- escalation_risk: "low|medium|high"

Quality / grounding:
- confidence_summary (number 0..1)
- confidence_sentiment (number 0..1)
- evidence_quotes (array of short phrases from the transcript, max {limits["max_quotes"]} items, <= 12 words each)
"""

    system = (
        "You are a call summarisation assistant for a telecoms service desk. "
        "Be accurate and concise. Output JSON only (no markdown, no code fences)."
    )

    user = f"""
{style_hint}

{schema}

Rubric (use this scoring):
- negative if clear frustration/anger/complaints/threats to leave
- positive if praise/thanks/resolution/relief
- neutral if calm technical discussion
Score thresholds (must follow):
- score <= -0.2 => negative
- -0.2 < score < 0.2 => neutral
- score >= 0.2 => positive

Rules:
- Do not invent facts. If unclear, use "unclear".
- Keep lists short and compact.
- If this is internal staff-to-staff and no customer is directly speaking, set sentiment_customer.score = 0.0.
- Use evidence_quotes that actually appear in the transcript.

{meta_block}TRANSCRIPT:
{text}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -------------------------
# Summarize + JSON repair
# -------------------------
def summarize_one_pass(req: CallSummaryRequest, text: str, limits: Dict[str, int]) -> Dict[str, Any]:
    content = llm.chat(
        build_messages(req, text, limits),
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    data = extract_json_object(content)

    if data is None:
        repair_messages = [
            {"role": "system", "content": "You are a strict JSON formatter. Output JSON only."},
            {"role": "user", "content": (
                "Return ONLY valid JSON with keys: summary, key_points, "
                "sentiment_overall, sentiment_customer, sentiment_agent, sentiment_drivers, "
                "escalation_risk, confidence_summary, confidence_sentiment, evidence_quotes.\n"
                "No markdown. No code fences. No extra text.\n"
                "If unknown, use empty lists and 0.0 scores.\n\n"
                f"TEXT:\n{content}"
            )},
        ]
        repaired = llm.chat(repair_messages, max_tokens=min(350, req.max_tokens), temperature=0.0)
        data = extract_json_object(repaired)

    if data is None:
        return {
            "summary": clean_text(content)[:1800],
            "key_points": [],
            "sentiment_overall": {"label": "neutral", "score": 0.0},
            "sentiment_customer": {"label": "neutral", "score": 0.0},
            "sentiment_agent": {"label": "neutral", "score": 0.0},
            "sentiment_drivers": [],
            "escalation_risk": "low",
            "confidence_summary": 0.4,
            "confidence_sentiment": 0.4,
            "evidence_quotes": [],
        }

    def _get_list(k: str) -> List[str]:
        v = data.get(k, [])
        return v if isinstance(v, list) else []

    out = {
        "summary": clean_text(str(data.get("summary", ""))),
        "key_points": _get_list("key_points")[: limits["max_kp"]],
        "sentiment_overall": normalize_sentiment(data.get("sentiment_overall", {})),
        "sentiment_customer": normalize_sentiment(data.get("sentiment_customer", {})),
        "sentiment_agent": normalize_sentiment(data.get("sentiment_agent", {})),
        "sentiment_drivers": _get_list("sentiment_drivers")[: limits["max_drv"]],
        "escalation_risk": normalize_escalation_risk(data.get("escalation_risk", "low")),
        "confidence_summary": clamp(data.get("confidence_summary", 0.6), 0.0, 1.0),
        "confidence_sentiment": clamp(data.get("confidence_sentiment", 0.6), 0.0, 1.0),
        "evidence_quotes": [clean_text(str(x)) for x in _get_list("evidence_quotes")][: limits["max_quotes"]],
    }

    out["evidence_quotes"] = [
        q if len(q.split()) <= 12 else " ".join(q.split()[:12])
        for q in out["evidence_quotes"]
    ]
    return out


@app.get("/health")
def health():
    return {
        "ok": True,
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "model_max_len": MODEL_MAX_LEN,
        "cache_size": CACHE_SIZE,
    }


def make_cache_key(req: CallSummaryRequest, turns: List[str]) -> str:
    h = hashlib.sha256()
    joined = "\n".join(turns)
    h.update(joined.encode("utf-8", errors="ignore"))
    h.update(
        f"|style={req.style}|max_tokens={req.max_tokens}|temp={req.temperature}"
        f"|model={LLM_MODEL}|maxlen={MODEL_MAX_LEN}".encode("utf-8")
    )
    return h.hexdigest()


@app.post("/summarize-call", response_model=CallSummaryResponse)
def summarize_call(req: CallSummaryRequest):
    if len(req.transcript) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail=f"Transcript too large (limit: {MAX_INPUT_CHARS} chars)")

    turns = preprocess_transcript(req.transcript)
    if not turns:
        raise HTTPException(status_code=400, detail="Empty transcript (after preprocessing)")

    key = make_cache_key(req, turns)
    cached = cache.get(key)
    if cached is not None:
        return CallSummaryResponse(**cached)

    processed_text = "\n".join(turns)
    est_input_tokens = estimate_tokens_rough(processed_text)
    limits = choose_limits(req, est_input_tokens)

    prompt_overhead = 650
    reserved = prompt_overhead + req.max_tokens

    # Single-pass if it fits
    if est_input_tokens <= max(256, MODEL_MAX_LEN - reserved):
        out = summarize_one_pass(req, processed_text, limits)

        timeline = [
            SentimentTimelinePoint(
                index=0,
                start_char=0,
                end_char=len(processed_text),
                overall=SentimentScore(**out["sentiment_overall"]),
                customer=SentimentScore(**out["sentiment_customer"]),
                agent=SentimentScore(**out["sentiment_agent"]),
                drivers=out.get("sentiment_drivers", []) or [],
                escalation_risk=out.get("escalation_risk", "low") or "low",
            )
        ]
        trend = compute_trend_from_timeline(timeline)

        resp = CallSummaryResponse(
            **out,
            sentiment_trend=trend,
            sentiment_timeline=timeline,
            model=LLM_MODEL,
        ).model_dump()

        cache.set(key, resp)
        return CallSummaryResponse(**resp)

    # Chunk by turns (avoid splitting mid-turn)
    chunks = chunk_turns_by_token_budget(turns, max_input_tokens=MODEL_MAX_LEN, reserved_tokens=reserved)

    per_chunk_req = req.model_copy(update={"max_tokens": min(req.max_tokens, 500)})
    per_chunk_limits = choose_limits(per_chunk_req, est_input_tokens)

    per_chunk: List[Dict[str, Any]] = []
    timeline: List[SentimentTimelinePoint] = []

    cursor = 0
    for i, c in enumerate(chunks):
        start = cursor
        end = cursor + len(c)
        cursor = end + 1

        out_i = summarize_one_pass(per_chunk_req, c, per_chunk_limits)
        per_chunk.append(out_i)

        timeline.append(
            SentimentTimelinePoint(
                index=i,
                start_char=start,
                end_char=end,
                overall=SentimentScore(**out_i["sentiment_overall"]),
                customer=SentimentScore(**out_i["sentiment_customer"]),
                agent=SentimentScore(**out_i["sentiment_agent"]),
                drivers=out_i.get("sentiment_drivers", []) or [],
                escalation_risk=out_i.get("escalation_risk", "low") or "low",
            )
        )

    combined = "\n\n".join(
        (
            f"CHUNK {i+1}:\n{pc.get('summary','')}\n"
            f"Key points: {pc.get('key_points',[])}\n"
            f"Customer sentiment: {pc.get('sentiment_customer',{})}\n"
            f"Drivers: {pc.get('sentiment_drivers',[])}\n"
            f"Risk: {pc.get('escalation_risk','low')}"
        )
        for i, pc in enumerate(per_chunk)
    )

    final_req = req.model_copy(
        update={"call_reason": req.call_reason or "Combine chunk summaries into one coherent call summary."}
    )
    out = summarize_one_pass(final_req, combined, limits)

    trend = compute_trend_from_timeline(timeline)

    resp = CallSummaryResponse(
        **out,
        sentiment_trend=trend,
        sentiment_timeline=timeline,
        model=LLM_MODEL,
    ).model_dump()

    cache.set(key, resp)
    return CallSummaryResponse(**resp)
