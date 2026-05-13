"""
Module 05 -- Lesson 1: Your First AI Agent
============================================
What makes an agent different from a chatbot?

A chatbot:  You ask -> it answers -> done.
An agent:   You give a GOAL -> it PLANS steps -> EXECUTES them -> CHECKS results
            -> ADJUSTS if needed -> keeps going until the goal is met.

The key ingredients of an agent:
  1. A GOAL (what to accomplish)
  2. TOOLS (functions it can call)
  3. A LOOP (observe -> think -> act -> repeat)
  4. MEMORY (what it has done so far)

This lesson builds a Research Agent that:
  - Takes a topic from you
  - Plans what questions to investigate
  - Uses tools to gather information
  - Synthesizes a final report
  - All running locally with Ollama

This is the ReAct pattern (Reasoning + Acting) -- the foundation
of every AI agent framework (LangChain, CrewAI, AutoGPT, etc.)
"""

import requests
import json
import re
import os
import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"

# -- TOOLS: these are the functions the agent can call --------------------------
# In a real agent, these would call APIs, databases, web scrapers, etc.
# For learning, we simulate them so you can see the pattern without dependencies.

KNOWLEDGE_BASE = {
    "python": {
        "creator": "Guido van Rossum",
        "year": 1991,
        "type": "interpreted, high-level",
        "popular_for": "AI/ML, web dev, scripting, data science",
        "key_feature": "readability and simplicity",
    },
    "javascript": {
        "creator": "Brendan Eich",
        "year": 1995,
        "type": "interpreted, multi-paradigm",
        "popular_for": "web development, frontend, backend (Node.js)",
        "key_feature": "runs in every browser",
    },
    "rust": {
        "creator": "Graydon Hoare (Mozilla)",
        "year": 2010,
        "type": "compiled, systems language",
        "popular_for": "systems programming, WebAssembly, performance-critical apps",
        "key_feature": "memory safety without garbage collection",
    },
    "neural networks": {
        "creator": "concept from McCulloch & Pitts (1943)",
        "year": 1943,
        "type": "computational model inspired by biological neurons",
        "popular_for": "image recognition, NLP, game AI, self-driving cars",
        "key_feature": "learns patterns from data via backpropagation",
    },
    "transformers": {
        "creator": "Vaswani et al. (Google, 2017)",
        "year": 2017,
        "type": "neural network architecture",
        "popular_for": "LLMs (GPT, Claude, Llama), translation, text generation",
        "key_feature": "self-attention mechanism processes all tokens in parallel",
    },
    "machine learning": {
        "creator": "Arthur Samuel coined the term (1959)",
        "year": 1959,
        "type": "subset of AI",
        "popular_for": "predictions, recommendations, fraud detection, medical diagnosis",
        "key_feature": "systems improve from experience without explicit programming",
    },
}


def tool_lookup(topic):
    """Look up a topic in our knowledge base."""
    topic_lower = topic.lower().strip()
    for key, info in KNOWLEDGE_BASE.items():
        if key in topic_lower or topic_lower in key:
            lines = [f"Topic: {key.title()}"]
            for field, value in info.items():
                lines.append(f"  {field}: {value}")
            return "\n".join(lines)
    return f"No information found for '{topic}'. Available topics: {', '.join(KNOWLEDGE_BASE.keys())}"


def tool_calculate(expression):
    """Safely evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Result: {expression} = {result}"
        return "Error: invalid characters in expression"
    except Exception as e:
        return f"Calculation error: {e}"


def tool_note(content):
    """Save a note to the agent's scratchpad."""
    return f"Note saved: {content}"


def tool_get_date():
    """Get current date and time."""
    now = datetime.datetime.now()
    return f"Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"


# -- TOOL REGISTRY: maps tool names to functions + descriptions -----------------
TOOLS = {
    "lookup": {
        "func": tool_lookup,
        "desc": "Look up a topic. Usage: [TOOL: lookup(topic name)]",
        "takes_arg": True,
    },
    "calculate": {
        "func": tool_calculate,
        "desc": "Calculate a math expression. Usage: [TOOL: calculate(2 + 2)]",
        "takes_arg": True,
    },
    "note": {
        "func": tool_note,
        "desc": "Save a note for later reference. Usage: [TOOL: note(your note here)]",
        "takes_arg": True,
    },
    "get_date": {
        "func": tool_get_date,
        "desc": "Get current date and time. Usage: [TOOL: get_date()]",
        "takes_arg": False,
    },
}


def build_system_prompt():
    """Build the system prompt that teaches the agent how to use tools."""
    tool_list = "\n".join(f"  - {name}: {info['desc']}" for name, info in TOOLS.items())

    return f"""You are a Research Agent. Your job is to investigate topics thoroughly.

You have access to these tools:
{tool_list}

HOW TO USE TOOLS:
When you need information, write [TOOL: function_name(argument)] in your response.
The system will execute the tool and give you the result.
You can use multiple tools in one response.

YOUR PROCESS:
1. PLAN: When given a topic, first plan what you need to investigate
2. ACT: Use tools to gather information
3. OBSERVE: Read the tool results
4. THINK: Analyze what you learned and what else you need
5. REPORT: When you have enough info, write [DONE] followed by your final report

RULES:
- Always start by using [TOOL: lookup(topic)] to gather information
- Use [TOOL: note(insight)] to save important findings as you go
- When you have gathered enough information, write [DONE] and then your final report
- Keep your thinking SHORT between tool calls
- Your final report should be 3-5 sentences summarizing what you found
- Be conversational, not robotic"""


def parse_tool_calls(text):
    """Extract all [TOOL: func(args)] patterns from text."""
    pattern = r'\[TOOL:\s*(\w+)\(([^)]*)\)\]'
    matches = re.findall(pattern, text)
    return matches


def execute_tool(name, arg):
    """Execute a tool by name with the given argument."""
    if name not in TOOLS:
        return f"Unknown tool: {name}"
    tool = TOOLS[name]
    if tool["takes_arg"]:
        return tool["func"](arg)
    else:
        return tool["func"]()


def run_agent(goal, max_steps=8):
    """
    The Agent Loop -- this is the heart of every AI agent.

    1. Send the goal + history to the LLM
    2. Parse any tool calls from the response
    3. Execute the tools
    4. Add results to history
    5. Repeat until [DONE] or max steps reached
    """
    print(f"\n  GOAL: {goal}")
    print(f"  Max steps: {max_steps}")
    print("  " + "-" * 50)

    history = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Research this topic thoroughly: {goal}"},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n  --- Step {step}/{max_steps} ---")

        # 1. Ask the LLM
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "stream": False,
                "messages": history,
            }, timeout=120)
            reply = response.json()["message"]["content"]
        except Exception as e:
            print(f"  ERROR: Could not reach Ollama: {e}")
            break

        # 2. Check if agent says it's done
        if "[DONE]" in reply:
            print(f"\n  Agent finished in {step} steps!")
            # Extract the final report (everything after [DONE])
            report = reply.split("[DONE]", 1)[1].strip()
            if not report:
                report = reply  # fallback: show full response
            history.append({"role": "assistant", "content": reply})
            return report, history

        # 3. Parse tool calls
        tool_calls = parse_tool_calls(reply)

        if tool_calls:
            print(f"  Agent is thinking and using {len(tool_calls)} tool(s):")

            # Show the agent's reasoning (text before/between tool calls)
            reasoning = re.sub(r'\[TOOL:.*?\]', '', reply).strip()
            if reasoning:
                # Show just first 200 chars of reasoning
                short = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                print(f"  Reasoning: {short}")

            # 4. Execute each tool and collect results
            tool_results = []
            for func_name, arg in tool_calls:
                result = execute_tool(func_name, arg.strip())
                tool_results.append(f"[Result of {func_name}({arg})]: {result}")
                print(f"    -> {func_name}({arg}): {result[:100]}...")

            # 5. Add to history and continue the loop
            history.append({"role": "assistant", "content": reply})
            history.append({"role": "user", "content": "\n".join(tool_results) + "\n\nContinue your research. Use more tools or write [DONE] followed by your final report."})

        else:
            # No tool calls and no [DONE] -- agent is just thinking
            print(f"  Agent thinking: {reply[:200]}...")
            history.append({"role": "assistant", "content": reply})
            history.append({"role": "user", "content": "Use your tools to gather information, or write [DONE] followed by your final report."})

    print(f"\n  Agent hit max steps ({max_steps}). Forcing completion.")
    return "Agent reached maximum steps without completing.", history


def main():
    print("\n" + "=" * 60)
    print("  MODULE 05 -- LESSON 1: YOUR FIRST AI AGENT")
    print("  The ReAct Pattern: Reasoning + Acting")
    print("  Model: qwen2.5-coder:7b (local via Ollama)")
    print("=" * 60)

    # Check Ollama
    print("\n  Checking Ollama...")
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        print("  Ollama ready.")
    except Exception:
        print("  ERROR: Ollama not running! Start it with 'ollama serve'")
        return

    # -- Demo 1: Automatic research --
    print("\n" + "=" * 60)
    print("  DEMO: Watch the agent research a topic autonomously")
    print("=" * 60)

    report, history = run_agent("Tell me about Python and how it relates to machine learning")

    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"\n  {report}")

    # -- Demo 2: Interactive mode --
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Give the agent any topic to research")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        topic = input("\n  Research topic: ").strip()
        if not topic or topic.lower() in ["quit", "exit", "q"]:
            break

        report, _ = run_agent(topic)
        print("\n" + "=" * 60)
        print("  FINAL REPORT")
        print("=" * 60)
        print(f"\n  {report}")

    print("\n" + "=" * 60)
    print("  KEY TAKEAWAYS:")
    print("  1. An agent = LLM + Tools + Loop")
    print("  2. The ReAct pattern: Reason about what to do, Act with tools")
    print("  3. The system prompt teaches the LLM HOW to use tools")
    print("  4. The loop keeps going until the goal is met")
    print("  5. This is exactly how LangChain, CrewAI, AutoGPT work")
    print("=" * 60)


if __name__ == "__main__":
    main()
