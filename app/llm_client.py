import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    More forgiving JSON extraction:
    - Strips code fences
    - Tries parsing entire text
    - Falls back to the first {...} block
    """
    if not text:
        return None

    text = text.strip()

    # Strip code fences
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```", "", text)

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def estimate_tokens_rough(text: str) -> int:
    """
    Rough token estimator (fast, no tokenizer dependency).
    Rule of thumb: ~4 chars/token for English-ish text.
    Add a small padding.
    """
    t = clean_text(text)
    if not t:
        return 0
    return max(1, (len(t) // 4) + 8)


def chunk_turns_by_token_budget(
    turns: List[str],
    max_input_tokens: int,
    reserved_tokens: int,
) -> List[str]:
    """
    Chunk by 'turns' (lines) to avoid splitting mid-utterance.
    Keeps each chunk within: max_input_tokens - reserved_tokens (rough estimate).
    """
    budget = max(256, max_input_tokens - max(0, reserved_tokens))

    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0

    for t in turns:
        t = t.strip()
        if not t:
            continue
        tt = estimate_tokens_rough(t)

        # If a single turn exceeds budget, hard split it.
        if tt > budget:
            if cur:
                chunks.append("\n".join(cur).strip())
                cur, cur_tok = [], 0

            # Hard split by chars ~ tokens*4
            hard_chars = max(256, budget * 4)
            for i in range(0, len(t), hard_chars):
                piece = t[i : i + hard_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        if cur_tok + tt <= budget:
            cur.append(t)
            cur_tok += tt
        else:
            chunks.append("\n".join(cur).strip())
            cur = [t]
            cur_tok = tt

    if cur:
        chunks.append("\n".join(cur).strip())

    return chunks


@dataclass
class LLMConfig:
    base_url: str
    model: str
    timeout_secs: float


class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = httpx.Client(timeout=cfg.timeout_secs)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 600, temperature: float = 0.2) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        r = self.client.post(url, json=payload)

        if r.status_code >= 400:
            raise RuntimeError(f"vLLM {r.status_code} error: {r.text}")

        data = r.json()
        return clean_text(data["choices"][0]["message"]["content"])
