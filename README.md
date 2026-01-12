# Call Summary API (Llama via vLLM)

A Docker-first project that runs a local **Llama** model using **vLLM** and exposes a small **FastAPI** service to generate **call summaries** from transcripts.

Designed for support/telecom call transcripts: produces concise summaries plus **key points, decisions, action items, risks, and follow-ups**. Handles long transcripts using **chunk + reduce** summarisation.

---

## What’s included

- **vLLM container** serving Llama via an **OpenAI-compatible API** (`/v1/chat/completions`)
- **FastAPI container** exposing:
  - `GET /health`
  - `POST /summarize-call`

The FastAPI service talks to vLLM over the internal Docker network (no external LLM dependency).

---

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (for vLLM)
- A Hugging Face token (`HF_TOKEN`) if using gated Meta Llama models (e.g. `meta-llama/*`)

---

## Quick start

1) Copy env file and set your Hugging Face token:

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

## Generate a call summary

### Endpoint
`POST /summarize-call`

### Example request

```bash
curl -s http://localhost:8000/summarize-call   -H "Content-Type: application/json"   -d '{
    "agent": "Simon",
    "customer": "John",
    "call_reason": "Calls dropping on SIP trunk",
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

### Example response

```json
{
  "summary": "…",
  "key_points": ["…"],
  "decisions": ["…"],
  "action_items": ["…"],
  "risks": ["…"],
  "follow_ups": ["…"],
  "model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

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
- `CHUNK_CHARS` – chunk size for long transcripts
- `REQUEST_TIMEOUT_SECS` – timeout for LLM calls

---

## Troubleshooting

### vLLM container keeps restarting
Usually insufficient free VRAM. Lower:
- `VLLM_GPU_MEMORY_UTILIZATION`
and/or
- `VLLM_MAX_MODEL_LEN`

Then restart.

### API returns “Upstream LLM error”
Check vLLM from inside the API container:

```bash
docker compose exec api sh -lc 'curl -sS http://llm:8000/v1/models | head'
```

---

## Notes

- The API requests JSON output. If the model returns invalid JSON, the API falls back to returning the raw summary string.
- For best results, provide clean transcripts (speaker labels and timestamps help).
