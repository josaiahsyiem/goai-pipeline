"""
tools/llm_client.py
-------------------
LLM client for the GoAI pipeline.
Tier 1: Groq llama-3.3-70b-versatile — rotates across 3 API keys.
Tier 2: OpenAI GPT-4o-mini — reliable fallback.
Tier 3: Ollama (local) — emergency fallback.
Embeddings: Ollama nomic-embed-text (768-dim).
"""

import hashlib
import os

import requests
from groq import Groq
from openai import OpenAI

try:
    from langfuse_client import langfuse as _langfuse
except Exception:
    _langfuse = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST",  "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"

# ── Fix 1: Model pricing (per million tokens) ─────────────────────────────────
# Source: Groq and OpenAI pricing pages
PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gpt-4o-mini":              {"input": 0.15, "output": 0.60},
    "ollama":                   {"input": 0.0,  "output": 0.0},
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return round((input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000, 6)


# ── Groq key rotation ─────────────────────────────────────────────────────────
GROQ_KEYS = [
    k for k in [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY_4"),
        os.getenv("GROQ_API_KEY_5"),
        os.getenv("GROQ_API_KEY_6"),
        os.getenv("GROQ_API_KEY_7"),
    ] if k
]
_groq_key_index = 0


def _next_groq_key() -> str:
    global _groq_key_index
    key = GROQ_KEYS[_groq_key_index % len(GROQ_KEYS)]
    _groq_key_index += 1
    return key


# ── LLM backends ──────────────────────────────────────────────────────────────

def groq_chat(system: str, user: str, call_name: str = "groq_chat") -> str:
    """Tries all Groq keys in rotation before giving up."""
    last_error = None
    for _ in range(len(GROQ_KEYS)):
        key = _next_groq_key()
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
            if not content or len(content) < 2:
                raise Exception("Empty or truncated response from Groq")

            # Fix 1+2: Langfuse generation with cost tracking
            if _langfuse:
                try:
                    usage = response.usage
                    inp_tok = usage.prompt_tokens
                    out_tok = usage.completion_tokens
                    cost = _compute_cost(GROQ_MODEL, inp_tok, out_tok)
                    _langfuse.create_event(
                        name=call_name,
                        input={"system": system[:500], "user": user[:500]},
                        output={
                            "response": content[:500],
                            "model": GROQ_MODEL,
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                            "cost_usd": cost,
                        },
                    )
                except Exception:
                    pass

            return content
        except Exception as e:
            print(f"[LLM] Groq key ...{key[-6:]} failed: {e}")
            last_error = e
    raise last_error


def openai_chat(system: str, user: str, call_name: str = "openai_chat") -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    content = response.choices[0].message.content.strip()

    # Fix 1+2: Track OpenAI calls too
    if _langfuse:
        try:
            usage = response.usage
            inp_tok = usage.prompt_tokens
            out_tok = usage.completion_tokens
            cost = _compute_cost(OPENAI_MODEL, inp_tok, out_tok)
            _langfuse.generation(
                name=call_name,
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": system[:800]},
                    {"role": "user",   "content": user[:800]},
                ],
                output=content[:800],
                usage={
                    "input":  inp_tok,
                    "output": out_tok,
                    "total":  usage.total_tokens,
                    "unit":   "TOKENS",
                },
                metadata={
                    "cost_usd":  cost,
                    "model":     OPENAI_MODEL,
                    "call_name": call_name,
                },
            )
        except Exception:
            pass

    return content


def ollama_chat(system: str, user: str, timeout: int = 120) -> str:
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    OLLAMA_MODEL,
            "stream":   False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    """768-dimensional embedding via Ollama nomic-embed-text."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["embedding"]
    except Exception:
        pass

    # Fallback: deterministic hash-based pseudo-embedding (768-dim)
    hash_val = hashlib.md5(text.encode()).hexdigest()
    vec = [int(hash_val[i:i + 2], 16) / 255.0 for i in range(0, 32, 2)]
    return (vec * 49)[:768]


# ── Router ────────────────────────────────────────────────────────────────────

def smart_chat(system: str, user: str, use_groq: bool = True, call_name: str = "llm_call") -> str:
    if use_groq and GROQ_KEYS:
        try:
            return groq_chat(system, user, call_name=call_name)
        except Exception as e:
            print(f"[LLM] All Groq keys exhausted — escalating to GPT-4o-mini")

    if OPENAI_API_KEY:
        try:
            return openai_chat(system, user, call_name=call_name)
        except Exception as e:
            print(f"[LLM] GPT-4o-mini failed: {e} — falling back to Ollama")

    print("[LLM] Using Ollama emergency fallback")
    return ollama_chat(system, user)
