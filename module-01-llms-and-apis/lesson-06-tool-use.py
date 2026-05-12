"""
Module 01 — Lesson 6: Tool Use / Function Calling
===================================================
Until now, the model could only generate text. Ask it the time? It guesses.
Ask it to calculate something? It might get it wrong.

Tool use changes everything. You define functions the model CAN call.
When it needs real data, it returns a structured tool call instead of text.
Your code executes the function, sends the result back, and the model
writes a final answer using real data.

THE FLOW:
  1. User asks a question
  2. Model decides it needs a tool and outputs: [TOOL: func_name(args)]
  3. Your code parses that, executes the real function
  4. You send the result back to the model
  5. Model writes the final answer using the real data

REAL-WORLD NOTE:
  Big models (Claude, GPT-4) use a proper structured tool_calls format.
  Smaller local models often can't follow that spec reliably.
  So we use a simple text pattern the model CAN follow — the concept is
  identical, just a different format. Once you understand this pattern,
  switching to Claude/GPT-4's native tool_calls is trivial.

Model: qwen2.5-coder:7b (local via Ollama)
"""

import requests
import json
import re
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5-coder:7b"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Define the REAL functions the model can use
# ══════════════════════════════════════════════════════════════════════════════
# These are normal Python functions. The model doesn't run them — YOU do.

def get_current_time() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Error: invalid characters in expression"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Simulate a weather lookup. In production, this calls a real API."""
    fake_weather = {
        "tokyo":     {"temp": 22, "condition": "sunny",         "humidity": 45},
        "london":    {"temp": 14, "condition": "cloudy",        "humidity": 78},
        "new york":  {"temp": 28, "condition": "partly cloudy", "humidity": 55},
        "paris":     {"temp": 19, "condition": "rainy",         "humidity": 82},
        "sydney":    {"temp": 17, "condition": "windy",         "humidity": 60},
    }
    data = fake_weather.get(city.lower())
    if data:
        return f"{city}: {data['temp']}C, {data['condition']}, humidity {data['humidity']}%"
    return f"Weather data not available for {city}"


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units."""
    conversions = {
        ("km", "miles"):  lambda v: v * 0.621371,
        ("miles", "km"):  lambda v: v / 0.621371,
        ("kg", "lbs"):    lambda v: v * 2.20462,
        ("lbs", "kg"):    lambda v: v / 2.20462,
        ("c", "f"):       lambda v: (v * 9/5) + 32,
        ("f", "c"):       lambda v: (v - 32) * 5/9,
        ("m", "ft"):      lambda v: v * 3.28084,
        ("ft", "m"):      lambda v: v / 3.28084,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](float(value))
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    return f"Cannot convert from {from_unit} to {to_unit}"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: The system prompt teaches the model HOW to call tools
# ══════════════════════════════════════════════════════════════════════════════
# Instead of using a formal tool_calls API (which small models struggle with),
# we teach the model a simple text format: [TOOL: name(args)]
# This is the same concept — just expressed in a way the model can handle.

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

AVAILABLE TOOLS:
1. get_current_time() — returns current date and time. No arguments needed.
2. calculate(expression) — evaluates a math expression. Example: calculate(1547 * 38 + 92)
3. get_weather(city) — gets weather for a city. Example: get_weather(Tokyo)
4. convert_units(value, from_unit, to_unit) — converts units. Example: convert_units(100, km, miles)

HOW TO USE TOOLS:
- When you need real data, respond with ONLY a tool call in this exact format:
  [TOOL: function_name(arguments)]
- Examples:
  [TOOL: get_current_time()]
  [TOOL: calculate(1547 * 38 + 92)]
  [TOOL: get_weather(London)]
  [TOOL: convert_units(30, c, f)]
- Output ONLY the [TOOL: ...] tag, nothing else. No explanation before or after.
- For math questions, ALWAYS use calculate instead of doing math yourself.
- If you don't need a tool, just respond normally with text."""


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Parse tool calls from the model's text output
# ══════════════════════════════════════════════════════════════════════════════

def parse_tool_call(text: str):
    """
    Parse [TOOL: func_name(args)] from the model's response.
    Returns (func_name, args_list) or (None, None) if no tool call found.
    """
    match = re.search(r'\[TOOL:\s*(\w+)\(([^)]*)\)\]', text)
    if not match:
        return None, None

    func_name = match.group(1)
    raw_args = match.group(2).strip()

    # Split arguments by comma, strip whitespace
    if raw_args:
        args = [a.strip().strip("'\"") for a in raw_args.split(",")]
    else:
        args = []

    return func_name, args


def execute_tool(func_name: str, args: list) -> str:
    """Execute a tool by name with the parsed arguments."""
    if func_name == "get_current_time":
        return get_current_time()
    elif func_name == "calculate":
        return calculate(args[0] if args else "0")
    elif func_name == "get_weather":
        return get_weather(args[0] if args else "unknown")
    elif func_name == "convert_units":
        if len(args) >= 3:
            return convert_units(args[0], args[1], args[2])
        return "Error: convert_units needs value, from_unit, to_unit"
    else:
        return f"Unknown tool: {func_name}"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: The tool-use loop
# ══════════════════════════════════════════════════════════════════════════════
# Same concept as production tool use:
#   Send message → model returns tool call → execute → send result back → final answer

def chat(messages: list[dict]) -> str:
    """Send messages to the model, return the response text."""
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": messages,
    })
    return response.json()["message"]["content"]


def chat_with_tools(user_message: str, history: list[dict]) -> str:
    """Handle a user message, executing any tool calls the model makes."""
    history.append({"role": "user", "content": user_message})

    for _ in range(3):  # max 3 rounds of tool calls
        reply = chat(history)

        # Check if the model wants to call a tool
        func_name, args = parse_tool_call(reply)

        if func_name:
            # ── Model requested a tool call ──────────────────────────
            print(f"  [Tool call]  {func_name}({', '.join(args)})")

            result = execute_tool(func_name, args)
            print(f"  [Result]     {result}")

            # Add the exchange to history so the model sees what happened
            history.append({"role": "assistant", "content": reply})
            history.append({"role": "user", "content": f"Tool result: {result}\n\nNow use this result to answer the original question in natural language."})

            continue  # let model generate final answer with the result

        else:
            # ── Normal text response, no tool call ───────────────────
            history.append({"role": "assistant", "content": reply})
            return reply

    return "Could not complete the request after multiple tool attempts."


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Interactive chat
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Tool Use / Function Calling Demo")
    print(f"  Model: {MODEL} (local via Ollama)")
    print("=" * 60)
    print("  Available tools: time, calculator, weather, unit converter")
    print()
    print("  Try asking:")
    print('    "What time is it?"')
    print('    "What is 1547 * 38 + 92?"')
    print('    "What\'s the weather in Tokyo?"')
    print('    "Convert 100 km to miles"')
    print('    "What\'s 30 celsius in fahrenheit?"')
    print()
    print("  Type /quit to exit\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input == "/quit":
            print("Goodbye!")
            break

        print()
        reply = chat_with_tools(user_input, history)
        print(f"\nBot: {reply}\n")


if __name__ == "__main__":
    main()
