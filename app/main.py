import os
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

app = FastAPI(title="Call Summary API (Llama via vLLM)", version="1.0.0")


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
    # optional metadata for better summaries
    agent: Optional[str] = Field(None, description="Agent name")
    customer: Optional[str] = Field(None, description="Customer name")
    call_reason: Optional[str] = Field(None, description="What the call is about (if known)")
    style: str = Field("bullets", pattern="^(bullets|short|detailed)$")
    max_tokens: int = Field(700, ge=200, le=2000)
    temperature: float = Field(0.2, ge=0.0, le=1.2)


class CallSummaryResponse(BaseModel):
    summary: str
    key_points: List[str] = []
    decisions: List[str] = []
    action_items: List[str] = []
    risks: List[str] = []
    follow_ups: List[str] = []
    model: str


def build_messages(req: CallSummaryRequest, text: str) -> List[Dict[str, str]]:
    style_hint = {
        "short": "Return a short paragraph summary (3–5 sentences) + a short bullet list of action items.",
        "detailed": "Return a structured summary with headings and bullets, include decisions, actions, and any risks.",
        "bullets": "Return a concise bullet-point summary. Capture decisions and actions.",
    }[req.style]

    meta_bits = []
    if req.agent:
        meta_bits.append(f"Agent: {req.agent}")
    if req.customer:
        meta_bits.append(f"Customer: {req.customer}")
    if req.call_reason:
        meta_bits.append(f"Call reason: {req.call_reason}")
    meta = "\n".join(meta_bits).strip()

    system = (
        "You are a call summarisation assistant for a telecoms service desk. "
        "Be accurate. Do not invent facts. If something is unclear, say 'unclear'. "
        "Keep names, numbers, dates exactly as stated."
    )

    # Ask for JSON so we can return structured fields, but tolerate if the model doesn't comply.
    user = f"""
{style_hint}

Return output as JSON with these keys:
- summary (string)
- key_points (array of strings)
- decisions (array of strings)
- action_items (array of strings)
- risks (array of strings)
- follow_ups (array of strings)

{("Metadata:\n" + meta + "\n") if meta else ""}

TRANSCRIPT:
{text}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def summarize_one_pass(req: CallSummaryRequest, text: str) -> Dict[str, Any]:
    content = llm.chat(build_messages(req, text), max_tokens=req.max_tokens, temperature=req.temperature)
    data = extract_json_object(content)
    if data is None:
        # fallback: keep raw text in summary
        return {
            "summary": content,
            "key_points": [],
            "decisions": [],
            "action_items": [],
            "risks": [],
            "follow_ups": [],
        }
    # normalise keys
    return {
        "summary": clean_text(str(data.get("summary", ""))),
        "key_points": data.get("key_points", []) or [],
        "decisions": data.get("decisions", []) or [],
        "action_items": data.get("action_items", []) or [],
        "risks": data.get("risks", []) or [],
        "follow_ups": data.get("follow_ups", []) or [],
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

    transcript = clean_text(req.transcript)
    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcript")

    chunks = chunk_text(transcript, CHUNK_CHARS)

    # If short enough, do one pass.
    if len(chunks) == 1:
        out = summarize_one_pass(req, chunks[0])
        return CallSummaryResponse(**out, model=LLM_MODEL)

    # Map: summarise each chunk with smaller budget
    per_chunk = []
    per_chunk_req = req.model_copy(update={"max_tokens": min(req.max_tokens, 500), "temperature": req.temperature})
    for c in chunks:
        per_chunk.append(summarize_one_pass(per_chunk_req, c))

    # Reduce: combine chunk summaries into one final summary
    combined = "\n\n".join(
        f"CHUNK {i+1}:\n{pc.get('summary','')}\n"
        f"Key points: {pc.get('key_points',[])}\n"
        f"Decisions: {pc.get('decisions',[])}\n"
        f"Actions: {pc.get('action_items',[])}\n"
        f"Risks: {pc.get('risks',[])}\n"
        f"Follow-ups: {pc.get('follow_ups',[])}"
        for i, pc in enumerate(per_chunk)
    )

    final_req = req.model_copy(update={"call_reason": req.call_reason or "Combine chunk summaries into one coherent call summary."})
    out = summarize_one_pass(final_req, combined)
    return CallSummaryResponse(**out, model=LLM_MODEL)
