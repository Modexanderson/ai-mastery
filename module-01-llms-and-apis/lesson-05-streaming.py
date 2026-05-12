"""
Module 01 — Lesson 5: Streaming Responses
==========================================
Until now, every API call waited for the FULL response before showing anything.
That's fine for scripts, but terrible UX for chatbots — users stare at a blank
screen for 10+ seconds.

Streaming sends tokens AS THEY'RE GENERATED. Same model, same API — just
delivered in real-time chunks instead of one big blob.

This lesson builds a chat app that shows the response appearing word-by-word,
exactly like ChatGPT does.

Technical detail:
  stream: False  →  one JSON object with the complete response
  stream: True   →  a series of JSON lines (NDJSON), each with a small chunk

Model: qwen2.5-coder:7b (local via Ollama)
"""

import requests
import json
import sys
import time

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5-coder:7b"

history = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. Keep responses clear and concise. "
            "When explaining technical topics, use simple language."
        )
    }
]


# ── Non-streaming (what we've been doing) ────────────────────────────────────
# Included here so you can compare side-by-side.

def chat_blocking(messages: list[dict]) -> str:
    """Old way: wait for the entire response, return it all at once."""
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": messages
    })
    return response.json()["message"]["content"]


# ── Streaming (the new technique) ────────────────────────────────────────────
# Key differences:
#   1. "stream": True in the payload
#   2. stream=True in requests.post() — tells requests to not wait for full body
#   3. We iterate over lines as they arrive instead of reading one response
#   4. Each line is a JSON object with a "message.content" chunk

def chat_streaming(messages: list[dict]) -> str:
    """New way: print tokens as they arrive, return the full response at the end."""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "stream": True,          # ← Tell Ollama to stream
            "messages": messages
        },
        stream=True                  # ← Tell requests to stream the HTTP body
    )

    full_reply = ""

    # The response arrives as newline-delimited JSON (NDJSON).
    # Each line looks like: {"message": {"content": "Hello"}, "done": false}
    # The last line has "done": true.
    for line in response.iter_lines():
        if not line:
            continue

        chunk = json.loads(line)

        # Check if the stream is done
        if chunk.get("done", False):
            break

        # Extract the text fragment from this chunk
        token = chunk["message"]["content"]
        full_reply += token

        # Print it immediately — no newline, flush the buffer
        # sys.stdout.write + flush = text appears instantly
        # Regular print() buffers and waits, which defeats the purpose
        sys.stdout.write(token)
        sys.stdout.flush()

    return full_reply


# ── Demo: compare both modes ─────────────────────────────────────────────────

def demo_comparison():
    """Show the difference between blocking and streaming."""
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what an API is in 3 sentences."}
    ]

    # --- Blocking ---
    print("=" * 60)
    print("  MODE 1: Blocking (non-streaming)")
    print("  You see nothing until the full response arrives...")
    print("=" * 60)

    start = time.time()
    reply = chat_blocking(test_messages)
    elapsed = time.time() - start

    print(f"\n{reply}")
    print(f"\n  [Appeared all at once after {elapsed:.1f}s]\n")

    # --- Streaming ---
    print("=" * 60)
    print("  MODE 2: Streaming")
    print("  Watch the text appear token by token...")
    print("=" * 60)
    print()

    start = time.time()
    reply = chat_streaming(test_messages)
    elapsed = time.time() - start

    print(f"\n\n  [Streamed over {elapsed:.1f}s — felt faster even if same total time]\n")


# ── Interactive streaming chat ───────────────────────────────────────────────

def interactive_chat():
    """Full chat loop with streaming responses and conversation memory."""
    global history

    print("=" * 60)
    print("  Streaming Chat")
    print("  Model: qwen2.5-coder:7b (local via Ollama)")
    print("=" * 60)
    print("  Type /quit to exit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input == "/quit":
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        print("\nBot: ", end="")
        sys.stdout.flush()

        # Stream the response — tokens print in real-time inside this function
        reply = chat_streaming(history)
        print("\n")   # newline after the streamed response

        history.append({"role": "assistant", "content": reply})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n  LESSON 5: STREAMING RESPONSES\n")
    print("  First, we'll compare blocking vs streaming on the same question.")
    print("  Then you'll enter an interactive streaming chat.\n")

    # Part 1: side-by-side comparison
    demo_comparison()

    input("  Press Enter to start the interactive streaming chat...\n")

    # Part 2: interactive chat with streaming
    interactive_chat()


if __name__ == "__main__":
    main()
