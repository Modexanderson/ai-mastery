"""
Module 01 — Lesson 1: CLI Chat Assistant
=========================================
We're building a multi-turn terminal chatbot using the Claude API.

Key concepts:
  - How the messages array works (roles: user / assistant)
  - What a system prompt is and why it matters
  - How context accumulates across turns
  - Token usage — what you're being "charged" per message
"""

import os
import anthropic

# ── Client setup ──────────────────────────────────────────────────────────────
# The client reads ANTHROPIC_API_KEY from your environment automatically.
client = anthropic.Anthropic()

# ── System prompt ─────────────────────────────────────────────────────────────
# This is a special message that sets the behavior/persona of the AI.
# It is NOT part of the conversation history — it's sent separately on every call.
# Think of it as permanent instructions the model always reads first.
SYSTEM_PROMPT = """You are a concise, direct AI assistant helping a developer learn AI.
When explaining concepts, use short examples. Avoid filler phrases like 'Certainly!' or 'Great question!'.
Keep responses focused and practical."""

# ── Conversation history ───────────────────────────────────────────────────────
# This is the core of multi-turn chat. We keep a list of all messages so far.
# Each message is a dict with two keys:
#   "role"    → either "user" or "assistant"
#   "content" → the text of that message
#
# Every API call sends the FULL history. The model has no memory on its own —
# you are responsible for maintaining and sending the conversation context.
messages = []

# ── Helper: send a message and get a reply ────────────────────────────────────
def chat(user_input: str) -> str:
    # 1. Append the new user message to history
    messages.append({
        "role": "user",
        "content": user_input
    })

    # 2. Call the API with the full conversation history
    response = client.messages.create(
        model="claude-sonnet-4-6",        # which Claude model to use
        max_tokens=1024,                   # max tokens in the reply
        system=SYSTEM_PROMPT,              # system prompt (sent separately)
        messages=messages                  # full conversation history
    )

    # 3. Extract the reply text
    reply = response.content[0].text

    # 4. Append the assistant's reply to history (so next turn remembers it)
    messages.append({
        "role": "assistant",
        "content": reply
    })

    # 5. Show token usage so you can see the cost accumulating
    usage = response.usage
    print(f"\n[tokens — input: {usage.input_tokens} | output: {usage.output_tokens}]")

    return reply


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Claude CLI Assistant — Module 01 Lesson 1")
    print("  Type 'quit' to exit | 'reset' to clear history")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Bye.")
            break

        if user_input.lower() == "reset":
            messages.clear()
            print("[History cleared]")
            continue

        reply = chat(user_input)
        print(f"\nClaude: {reply}")


if __name__ == "__main__":
    main()
