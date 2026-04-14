"""
Module 01 — Lesson 3: Structured Output
=========================================
The problem: LLMs return freeform text by default.
That's great for chatting, but useless if you need to parse
results into a UI, database, or another system.

The solution: instruct the model to return valid JSON — always,
in a specific shape you define. No special API needed, just prompt design.

We'll build a code reviewer that returns structured feedback.
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5-coder:7b"

# ── The system prompt is doing all the heavy lifting here ─────────────────────
# Notice: we define the exact JSON shape, explain each field,
# and use "ONLY return JSON" as a hard constraint.
SYSTEM_PROMPT = """You are an expert Python code reviewer.

When given code, you MUST respond with ONLY a valid JSON object. No explanation,
no markdown, no code fences. Just raw JSON.

The JSON must follow this exact structure:
{
  "score": <integer 1-10, overall code quality>,
  "issues": [<string>, ...],
  "suggestions": [<string>, ...],
  "summary": "<one sentence overall assessment>"
}

Rules:
- "score" must be a number between 1 and 10
- "issues" lists actual problems (bugs, security risks, bad practices)
- "suggestions" lists improvements that would make the code better
- "summary" is a single sentence — direct, no filler
- If the code is empty or invalid, return score 0 with a relevant issue
- NEVER wrap the JSON in markdown or code blocks"""


def review_code(code: str) -> dict:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": f"Review this code:\n\n{code}"}
        ]
    })

    raw = response.json()["message"]["content"].strip()

    # Parse the JSON the model returned
    # If the model disobeyed and wrapped it in ```json ... ```, strip that too
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def display_review(result: dict):
    score = result.get("score", "?")
    issues = result.get("issues", [])
    suggestions = result.get("suggestions", [])
    summary = result.get("summary", "")

    # Score bar visual
    filled = int(score) if isinstance(score, int) else 0
    bar = "[" + "#" * filled + "-" * (10 - filled) + "]"

    print(f"\n{'=' * 55}")
    print(f"  CODE REVIEW RESULT")
    print(f"{'=' * 55}")
    print(f"  Score: {score}/10  {bar}")
    print(f"  Summary: {summary}")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

    if suggestions:
        print(f"\n  SUGGESTIONS ({len(suggestions)}):")
        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s}")

    print(f"{'=' * 55}\n")


# ── Two code samples to review ────────────────────────────────────────────────
# We'll review both so you can compare a bad vs decent piece of code.

BAD_CODE = """
def divide(a, b):
    return a / b

x = divide(10, 0)
print(x)
"""

DECENT_CODE = """
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

scores = [85, 92, 78, 95, 88]
avg = calculate_average(scores)
print(f"Average score: {avg}")
"""

def main():
    print("=" * 55)
    print("  Code Reviewer — Structured Output Demo")
    print("  Model: qwen2.5-coder:7b (local via Ollama)")
    print("=" * 55)

    # --- Review 1: bad code ---
    print("\n>> Reviewing BAD code (division by zero, no error handling)...")
    result1 = review_code(BAD_CODE)
    display_review(result1)

    # --- Review 2: decent code ---
    print(">> Reviewing DECENT code (clean, handles edge case)...")
    result2 = review_code(DECENT_CODE)
    display_review(result2)

    # --- Show raw JSON so you can see exactly what the model returned ---
    print("Raw JSON from model (bad code example):")
    print(json.dumps(result1, indent=2))


if __name__ == "__main__":
    main()
