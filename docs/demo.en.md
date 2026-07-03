# Demo App

TUI conversation demo showcasing hopnot's complete workflow.

## Quick Start

```bash
cd demo
pip install -r requirements.txt
cp .env.example .env     # fill in your API info
python demo.py
```

## Configuration

Edit `.env`:

```ini
BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-key-here
MODEL_NAME=gpt-4o-mini

# Optional: custom prompt template file
# PROMPT_TEMPLATE_FILE=./my_template.txt
```

Any OpenAI-compatible API works (e.g. DeepSeek):

```ini
BASE_URL=https://api.deepseek.com/v1
API_KEY=sk-xxx
MODEL_NAME=deepseek-chat
```

## Features

| Feature | Description |
|:---|:---|
| **Short-term memory** | Last 3 conversation turns (sliding window) |
| **Long-term memory** | `<main_point>` triples auto-extracted from LLM responses |
| **LLM calling** | OpenAI-compatible API (BASE_URL/API_KEY/MODEL_NAME) |
| **Local mode** | Runs without API, demonstrates memory only |
| **Cold start** | New knowledge auto-creates nodes (activation = 1.0) |

## Commands

| Command | Description |
|:---|:---|
| `/memory` | View memory graph and association strengths |
| `/stats` | System statistics |
| `/clear` | Clear short-term context |
| `/dump` | Export memory snapshot |
| `/help` | Help |
| `/quit` | Exit |

## Workflow

```
User input → Retrieve long-term memory (hopnot retrieval phase)
→ Build prompt (system instruction + short-term 3 turns + long-term context)
→ Call LLM API
→ Parse <main_point> triples from response
→ Store into memory (hopnot consolidation phase)
```

## Custom Prompt Template

Set `PROMPT_TEMPLATE_FILE` env var to a custom template file:

```bash
export PROMPT_TEMPLATE_FILE=./my_prompt.txt
python demo.py
```

Use `{recent_context}` and `{memory_context}` as placeholders in the template.
