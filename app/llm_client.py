import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s


def chunk_text(text: str, max_chars: int) -> List[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]

    # Try to chunk on sentence boundaries / newlines.
    parts = re.split(r"(?<=[\.\!\?])\s+|\n{2,}", text)
    chunks: List[str] = []
    cur = ""

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p

    if cur:
        chunks.append(cur)

    # Fallback: if any chunk is still too big, hard-split
    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i : i + max_chars])
    return final


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to recover a JSON object even if the model wrapped it in prose or code fences.
    """
    if not text:
        return None

    # Remove code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```", "", text)

    # Find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


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
        r.raise_for_status()
        data = r.json()
        return clean_text(data["choices"][0]["message"]["content"])
