"""
Module 01 — Lesson 4: Multi-Turn Conversations & Memory
=========================================================
KEY INSIGHT: LLMs have ZERO memory. They don't remember your last message.
Every API call is completely independent. To create a "conversation," YOU
must resend the entire chat history with every single request.

This means:
  - You control what the model "remembers"
  - The history grows with every exchange → tokens increase → costs increase
  - Eventually you hit the model's context window limit
  - Smart apps summarize or trim history to stay within limits

We'll build a terminal chatbot that demonstrates all of this.

Commands:
  /tokens  — show current conversation size
  /clear   — wipe history and start fresh
  /summary — compress history into a summary (real-world technique)
  /quit    — exit

Model: qwen2.5-coder:7b (local via Ollama)
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5-coder:7b"


# ── The conversation history ─────────────────────────────────────────────────
# This is the KEY data structure. It's just a list of message dicts.
# Every time we call the API, we send this ENTIRE list.
# The model sees all of it and responds based on the full context.

system_message = {
    "role": "system",
    "content": (
        "You are a friendly AI tutor helping someone learn about AI and programming. "
        "Keep answers concise but thorough. Reference earlier parts of the conversation "
        "when relevant — the user is testing whether you remember."
    )
}

# We always start with just the system message.
# User and assistant messages get appended as the conversation progresses.
history = [system_message]


# ── Helper: estimate token count ─────────────────────────────────────────────
# Real tokenizers (like tiktoken) split text into subword pieces.
# A rough estimate: 1 token ≈ 4 characters for English text.
# This is imprecise but useful for understanding the concept.

def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: total characters / 4."""
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 4


# ── Core: send messages and get a response ───────────────────────────────────

def chat(messages: list[dict]) -> str:
    """Send the full conversation history and return the assistant's reply."""
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": messages       # ← THE WHOLE HISTORY goes here, every time
    })
    return response.json()["message"]["content"]


# ── Command: /summary ────────────────────────────────────────────────────────
# This is a real technique used in production chatbots.
# When the conversation gets too long, you ask the model to summarize it,
# then replace the full history with just the summary.
# The model "forgets" the details but retains the key context.

def summarize_and_compress():
    """Ask the model to summarize the conversation, then replace history."""
    global history

    if len(history) <= 1:
        print("  Nothing to summarize yet.\n")
        return

    # Count before
    tokens_before = estimate_tokens(history)
    turns_before  = len(history) - 1   # minus the system message

    # Ask the model to summarize everything so far
    summary_request = history + [{
        "role": "user",
        "content": (
            "Summarize our entire conversation so far in 2-3 sentences. "
            "Include any key facts, names, or topics we discussed. "
            "This summary will replace the full history to save space."
        )
    }]
    summary = chat(summary_request)

    # Replace history: system message + one assistant message with the summary
    history = [
        system_message,
        {
            "role": "assistant",
            "content": f"[Conversation summary: {summary}]"
        }
    ]

    tokens_after = estimate_tokens(history)

    print(f"\n  Compressed {turns_before} messages (~{tokens_before} tokens)")
    print(f"  → into summary (~{tokens_after} tokens)")
    print(f"  Saved ~{tokens_before - tokens_after} tokens")
    print(f"\n  Summary: {summary}\n")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    global history

    print("=" * 60)
    print("  Conversational Chatbot — Memory Demo")
    print("  Model: qwen2.5-coder:7b (local via Ollama)")
    print("=" * 60)
    print("  Commands: /tokens  /clear  /summary  /quit")
    print("  Try telling the bot your name, then ask if it remembers!\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # ── Handle slash commands ─────────────────────────────────────
        if user_input == "/quit":
            print("Goodbye!")
            break

        if user_input == "/tokens":
            tokens = estimate_tokens(history)
            turns  = len(history) - 1    # exclude system message
            print(f"\n  Messages in history: {turns}")
            print(f"  Estimated tokens:    ~{tokens}")
            print(f"  (Context window for most models: 4,096 — 128,000 tokens)\n")
            continue

        if user_input == "/clear":
            history = [system_message]
            print("\n  History cleared. The bot has forgotten everything.\n")
            continue

        if user_input == "/summary":
            summarize_and_compress()
            continue

        # ── Normal message: append to history, send, get reply ────────
        # Step 1: Add the user's message to history
        history.append({"role": "user", "content": user_input})

        # Step 2: Send the ENTIRE history to the model
        reply = chat(history)

        # Step 3: Add the assistant's reply to history too
        # (so next time, the model sees this exchange as well)
        history.append({"role": "assistant", "content": reply})

        # Step 4: Display
        print(f"\nBot: {reply}\n")


if __name__ == "__main__":
    main()
