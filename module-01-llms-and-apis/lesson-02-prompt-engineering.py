"""
Module 01 — Lesson 2: Prompt Engineering
==========================================
We'll send the SAME question to 3 different system prompts and compare
how drastically the output changes. This proves that the system prompt
is the most powerful lever you have when working with LLMs.

Running locally via Ollama — no API credits needed.
Model: qwen2.5-coder:7b

Techniques demonstrated:
  1. Zero-shot      — just ask, no examples
  2. Few-shot       — show examples before asking
  3. Chain-of-thought — force step-by-step reasoning
  4. Role + constraint — persona + hard output rules
"""

import requests
import json

# ── Ollama config ─────────────────────────────────────────────────────────────
# Ollama runs a local HTTP server at port 11434.
# We call it like any API — just using localhost instead of the internet.
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5-coder:7b"

# ── Core function: send a prompt, get a reply ─────────────────────────────────
def ask(system_prompt: str, user_message: str) -> str:
    payload = {
        "model": MODEL,
        "stream": False,           # wait for full response, don't stream tokens
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message}
        ]
    }
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    return data["message"]["content"]


# ── The 4 system prompts we'll compare ───────────────────────────────────────

PROMPTS = {

    # 1. ZERO-SHOT — no guidance at all, just a bare assistant
    "Zero-shot (no system prompt)":
        "You are a helpful assistant.",

    # 2. ROLE + CONSTRAINT — strong persona with strict output rules
    "Role + Constraint":
        """You are a senior software engineer with 20 years of experience.
You are brutally concise. You answer in bullet points only. No prose.
Max 5 bullets. No greetings or filler.""",

    # 3. FEW-SHOT — teach by example inside the system prompt
    "Few-shot (examples)":
        """You explain technical concepts using simple analogies.
Always follow this format:

Concept: [restate the concept in plain English]
Analogy: [a real-world analogy a 10-year-old would understand]
Key insight: [the one thing they must remember]

Example:
Concept: HTTP is a protocol for transferring data on the web
Analogy: Like ordering food — you (browser) ask, restaurant (server) sends
Key insight: It's stateless — the server forgets you after each request

Now do the same for the user's question.""",

    # 4. CHAIN-OF-THOUGHT — force the model to reason before answering
    "Chain-of-thought":
        """Before answering any question, you must:
1. Restate what is being asked
2. List what you know that's relevant
3. Identify any common misconceptions
4. Then give your final answer

Always show all 4 steps.""",
}


# ── Main: run the comparison ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Prompt Engineering Comparison Tool")
    print("  Model: qwen2.5-coder:7b (local via Ollama)")
    print("=" * 60)

    question = input("\nEnter your question: ").strip()
    if not question:
        question = "What is a neural network?"

    print(f"\nQuestion: {question}")

    for name, system_prompt in PROMPTS.items():
        print(f"\n{'-' * 60}")
        print(f"  STYLE: {name}")
        print(f"{'-' * 60}")
        reply = ask(system_prompt, question)
        print(reply)

    print(f"\n{'=' * 60}")
    print("Notice how the same question produces completely different")
    print("outputs based purely on the system prompt. No code changed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
