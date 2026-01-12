# Telephone Call Summary and Sentiment API (Llama via vLLM)

A Docker-first project that runs a local **Llama** model using **vLLM** and exposes a small **FastAPI** service to generate **call summaries** from transcripts.

Designed for telephone call transcripts: produces concise summaries plus **key points, decisions, action items, risks, and follow-ups**. Handles long transcripts using **chunk + reduce** summarisation.

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
  - Overall, agent, and customer sentiment
  - Per-chunk sentiment timeline for trend analysis

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
    "call_reason": "i need help with my account, my login wont work!",
    "style": "detailed",
    "max_tokens": 900,
    "temperature": 0.2,
    "transcript": "Agent: Hello, you are through to support...\nCustomer: We are seeing calls drop after 20 seconds...\n..."
  }'
```

### Request fields

- `transcript` (string, **required**): full transcript text
- `agent` (string, optional)
- `customer` (string, optional)
- `call_reason` (string, optional)
- `style` (string): `bullets` | `short` | `detailed` (default: `bullets`)
- `max_tokens` (int): output budget for the model (default: `700`)
- `temperature` (float): creativity / determinism (default: `0.2`)

---

## Response fields (summary + sentiment)

The API returns your call summary plus sentiment and a per-chunk sentiment timeline.

### Example response

```json
{
  "summary": "…",
  "key_points": ["…"],
  "decisions": ["…"],
  "action_items": ["…"],
  "risks": ["…"],
  "follow_ups": ["…"],

  "sentiment_overall": { "label": "neutral", "score": 0.05 },
  "sentiment_customer": { "label": "negative", "score": -0.35 },
  "sentiment_agent": { "label": "neutral", "score": 0.10 },
  "sentiment_trend": "improving",
  "sentiment_drivers": ["Frustration about purchase date selection", "Resolution agreed and reassurance given"],
  "escalation_risk": "low",

  "sentiment_timeline": [
    {
      "index": 0,
      "start_char": 0,
      "end_char": 11800,
      "overall": { "label": "neutral", "score": 0.10 },
      "customer": { "label": "positive", "score": 0.25 },
      "agent": { "label": "positive", "score": 0.20 },
      "drivers": ["Friendly greeting and small talk"],
      "escalation_risk": "low"
    },
    {
      "index": 1,
      "start_char": 11801,
      "end_char": 24050,
      "overall": { "label": "neutral", "score": -0.05 },
      "customer": { "label": "negative", "score": -0.40 },
      "agent": { "label": "neutral", "score": 0.00 },
      "drivers": ["Annoyance about lead times / bank holidays causing rejected requests"],
      "escalation_risk": "low"
    }
  ],

  "model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

### Sentiment interpretation

- `label`: `positive | neutral | negative`
- `score`: float range **-1.0 .. +1.0** (more negative = more negative sentiment)
- `sentiment_trend`: `improving | steady | worsening` (derived from the first vs last timeline point)
- `escalation_risk`: `low | medium | high`

---

## GPU / VRAM tuning (important)

vLLM reserves GPU memory based on `VLLM_GPU_MEMORY_UTILIZATION`. If you’re running other GPU services (ASR/TTS), you must lower it.

Good starting point when sharing a ~32GB GPU:
- `VLLM_GPU_MEMORY_UTILIZATION=0.45–0.60`
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
- `MAX_INPUT_CHARS` – max accepted transcript size
- `CHUNK_CHARS` – chunk size for long transcripts (also drives timeline granularity)
- `REQUEST_TIMEOUT_SECS` – timeout for LLM calls

---

## Troubleshooting

### vLLM container keeps restarting
Usually insufficient free VRAM. Lower:
- `VLLM_GPU_MEMORY_UTILIZATION`
and/or
- `VLLM_MAX_MODEL_LEN`

Then restart.

## Notes

- The API requests JSON output. If the model returns invalid JSON, the API falls back to returning the raw `summary` string.
- For best results, provide clean transcripts (speaker labels and timestamps help).
- If your transcript includes IVR/menu prompts as `[SPEAKER_00]`, the API can be configured to ignore them to improve sentiment accuracy.
