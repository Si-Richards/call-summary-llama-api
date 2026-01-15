# Telephone Call Summary and Sentiment API (Llama via vLLM)

A Docker-first project that runs a local **Llama** model using **vLLM** and exposes a small **FastAPI** service to generate **call summaries** from transcripts.

Designed for telephone call transcripts: produces concise summaries plus **key points** and **sentiment** (overall/agent/customer) with a **sentiment timeline**. Long transcripts are handled using **chunk + reduce**.

Tested on **RTX 5090 (Blackwell) 32GB**.

---

## What’s included

- **vLLM container** serving Llama via an **OpenAI-compatible API** (`/v1/chat/completions`)
- **FastAPI container** exposing:
  - `GET /health`
  - `POST /summarize-call`
- **API Docs**
  - `GET /docs`
- **Sentiment + Sentiment Timeline**
  - Overall, agent, and customer sentiment (score + label)
  - Per-chunk sentiment timeline for trend analysis
  - **Trend is computed in code** from the first vs last timeline customer score (no LLM inconsistency)
- **Efficiency improvements**
  - Transcript preprocessing (removes timestamps/IVR/UNKNOWN, compresses speaker tags)
  - Token-aware chunking (avoids vLLM 400 errors from context overflow)
  - Compact JSON schema + list limits to improve reliability
  - Optional in-memory caching (LRU) to avoid re-processing identical transcripts

The FastAPI service talks to vLLM over the internal Docker network (no external LLM dependency).

---

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (for vLLM)
- A Hugging Face token (`HF_TOKEN`) for the Llama model (read-only is fine)

---

## Quick start

1) Copy env file and set your Hugging Face API token:

```bash
cp .env.example .env
nano .env
```

2) Start everything:

```bash
docker compose up -d --build
```

3) Watch vLLM load the model:

```bash
docker compose logs -f llm
```

4) Check API health:

```bash
curl -s http://localhost:8000/health
```

---

## Generate a call summary (with sentiment)

### Endpoint
`POST /summarize-call`

### Example request

```bash
curl -s http://localhost:8000/summarize-call \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "Simon",
    "customer": "John",
    "call_reason": "Account login not working",
    "style": "bullets",
    "max_tokens": 600,
    "temperature": 0.0,
    "transcript": "[00:02.41] (SPEAKER_01) Morning.\n[00:05.10] (SPEAKER_02) Hi, I can\u2019t log in..."
  }'
```

### Request fields

- `transcript` (string, **required**): full transcript text
- `agent` (string, optional)
- `customer` (string, optional)
- `call_reason` (string, optional)
- `style` (string): `bullets` | `short` | `detailed` (default: `bullets`)
- `max_tokens` (int): output budget for the model (default: `600`)
- `temperature` (float): creativity / determinism (default: `0.0`)

---

## Response fields (summary + sentiment)

### Example response

```json
{
  "summary": "Inbound calls ring briefly then go to voicemail; likely router/firewall blocking inbound SIP/RTP from some gateways. Recommended softphone tests to isolate network vs platform.",
  "key_points": [
    "Seat shows as registered but inbound calls do not ring and fall to voicemail",
    "Test using the mobile softphone on the same network, then on mobile data",
    "If softphone works on mobile data, issue likely local router/firewall restrictions",
    "Ask the customer/IT to provide router logs showing inbound/outbound traffic for failed calls",
    "If needed, switch to monitored SIP trunking domain to gather evidence of dropped traffic"
  ],
  "sentiment_overall": { "label": "neutral", "score": 0.10 },
  "sentiment_customer": { "label": "neutral", "score": 0.00 },
  "sentiment_agent": { "label": "positive", "score": 0.20 },
  "sentiment_trend": "steady",
  "sentiment_drivers": [
    "Collaborative troubleshooting",
    "Mild frustration about customer IT capability",
    "Clear next steps agreed"
  ],
  "escalation_risk": "low",
  "confidence_summary": 0.72,
  "confidence_sentiment": 0.62,
  "evidence_quotes": [
    "test with the mobile softphone",
    "traffic from particular servers is being blocked"
  ],
  "sentiment_timeline": [
    {
      "index": 0,
      "start_char": 0,
      "end_char": 8200,
      "overall": { "label": "neutral", "score": 0.10 },
      "customer": { "label": "neutral", "score": 0.00 },
      "agent": { "label": "positive", "score": 0.20 },
      "drivers": ["Collaborative troubleshooting"],
      "escalation_risk": "low"
    }
  ],
  "model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

### Sentiment interpretation

- `label`: `positive | neutral | negative`
- `score`: float range **-1.0 .. +1.0**
- `sentiment_trend`: `improving | steady | worsening` (**computed from the timeline**)
- `escalation_risk`: `low | medium | high`
- `confidence_summary` / `confidence_sentiment`: 0..1 (higher = more confident)
- `evidence_quotes`: short phrases pulled from the transcript as grounding

---

## Efficiency / Reliability Notes

### Transcript preprocessing
The API removes common token bloat before sending text to the LLM:
- Drops IVR/menu prompts and `(UNKNOWN)` lines
- Removes leading timestamps like `[00:02.41]`
- Compresses speaker labels (e.g. `SPEAKER_01` -> `A:` and `SPEAKER_02` -> `C:`)
- Normalizes common ASR slips (e.g. `CLS`/`PLS` -> `TLS`)

### Token-aware chunking
Chunking is based on an estimated token budget so the request stays within `VLLM_MAX_MODEL_LEN`, avoiding `400 Bad Request` from vLLM due to context overflow.

### Compact JSON schema
The prompt enforces:
- JSON-only responses
- Short list limits (key_points max 8; sentiment_drivers max 5; evidence_quotes max 2)
- Sentiment labels derived from scores using a rubric (reduces inconsistent outputs)

### Caching
If enabled, identical requests (same transcript + parameters) return from an in-memory LRU cache.

---

## GPU / VRAM tuning (important)

vLLM reserves GPU memory based on `VLLM_GPU_MEMORY_UTILIZATION`. If you’re running other GPU services (ASR/TTS), you must lower it.

Good starting point when sharing a ~32GB GPU:
- `VLLM_GPU_MEMORY_UTILIZATION=0.45–0.55`
- `VLLM_MAX_MODEL_LEN=4096` (recommended for summarisation)

Apply changes in `.env` and restart:

```bash
docker compose down
docker compose up -d
```

---

## Configuration

Edit `.env`:

- `HF_TOKEN` – Hugging Face token (needed for gated models)
- `LLM_MODEL` – model name (e.g. `meta-llama/Llama-3.1-8B-Instruct`)
- `VLLM_GPU_MEMORY_UTILIZATION` – fraction of total VRAM vLLM can reserve
- `VLLM_MAX_MODEL_LEN` – context length (lower = less KV cache VRAM use)
- `VLLM_MAX_NUM_SEQS` – concurrency (lower = more stable on shared GPU)
- `VLLM_MAX_BATCHED_TOKENS` – batching limit (controls throughput vs memory)
- `MAX_INPUT_CHARS` – max accepted transcript size
- `MODEL_MAX_LEN` – app-side max context (must match `VLLM_MAX_MODEL_LEN`)
- `REQUEST_TIMEOUT_SECS` – timeout for LLM calls
- `CACHE_SIZE` – in-memory LRU cache size (0 disables caching)

---

## Troubleshooting

### vLLM container keeps restarting
Usually insufficient free VRAM. Lower:
- `VLLM_GPU_MEMORY_UTILIZATION`
and/or
- `VLLM_MAX_MODEL_LEN`

Then restart.

### API returns `400 Bad Request` from vLLM
This usually indicates context overflow. Fix by:
- Ensuring `MODEL_MAX_LEN` matches `VLLM_MAX_MODEL_LEN`
- Keeping `VLLM_MAX_MODEL_LEN=4096` and letting token-aware chunking do the work
- Reducing `max_tokens` if you request very long outputs

### Not seeing sentiment fields or inconsistent trends
- Trend is computed from timeline in code (should not contradict itself).
- If sentiment feels wrong, check `evidence_quotes` to see what the model grounded on.

---

## Notes

- The API enforces JSON output. If the model returns invalid JSON, the API performs a strict JSON repair pass and still returns a structured response.
- For best results, provide transcripts with speaker labels and timestamps (both are supported and will be cleaned/compressed automatically).
