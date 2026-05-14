"""
Module 05 -- Lesson 3: The Planner Agent (Plan-and-Execute)
============================================================
Lesson 1: ReAct agent -- thinks and acts one step at a time
Lesson 2: Memory agent -- remembers across sessions

This lesson: PLANNER agent -- given a big goal, it:
  1. Breaks the goal into a numbered plan of subtasks
  2. Executes each subtask using tools
  3. Checks results and adjusts the plan if needed
  4. Produces a final deliverable

This is how the most powerful agents work:
  - Devin (AI software engineer) -- plans code changes then executes
  - AutoGPT -- creates task lists and works through them
  - Claude Code -- plans implementation then writes code

The pattern: PLAN -> EXECUTE -> VERIFY -> DELIVER

We'll build a Code Project Generator that:
  - Takes a project idea from you
  - Plans the file structure
  - Generates each file
  - Creates a README
  - All saved to disk as a real project
"""

import requests
import json
import re
import os
import sys
import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_projects")


# -- TOOLS ---------------------------------------------------------------------

def tool_create_folder(path):
    """Create a folder in the output directory."""
    safe_path = os.path.join(OUTPUT_DIR, path.strip())
    os.makedirs(safe_path, exist_ok=True)
    return f"Created folder: {path}"


def tool_create_file(args):
    """Create a file. Format: filepath|content"""
    if "|" not in args:
        return "Error: use format filepath|content"
    filepath, content = args.split("|", 1)
    filepath = filepath.strip()
    safe_path = os.path.join(OUTPUT_DIR, filepath)
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    with open(safe_path, "w") as f:
        f.write(content)
    lines = content.count("\n") + 1
    return f"Created {filepath} ({lines} lines)"


def tool_list_files(path):
    """List files in a directory."""
    safe_path = os.path.join(OUTPUT_DIR, path.strip()) if path.strip() else OUTPUT_DIR
    if not os.path.exists(safe_path):
        return f"Directory not found: {path}"
    items = []
    for item in os.listdir(safe_path):
        full = os.path.join(safe_path, item)
        if os.path.isdir(full):
            items.append(f"  [DIR] {item}/")
        else:
            size = os.path.getsize(full)
            items.append(f"  [FILE] {item} ({size} bytes)")
    return f"Contents of {path or 'root'}:\n" + "\n".join(items) if items else "Empty directory"


def tool_read_file(filepath):
    """Read a file from the output directory."""
    safe_path = os.path.join(OUTPUT_DIR, filepath.strip())
    try:
        with open(safe_path, "r") as f:
            content = f.read()
        return f"Contents of {filepath}:\n{content[:800]}"
    except FileNotFoundError:
        return f"File not found: {filepath}"


def tool_plan_complete(summary):
    """Signal that the plan is complete with a summary."""
    return f"PROJECT COMPLETE: {summary}"


TOOLS = {
    "create_folder": {"func": tool_create_folder, "desc": "Create folder. [TOOL: create_folder(path)]",          "takes_arg": True},
    "create_file":   {"func": tool_create_file,   "desc": "Create file. [TOOL: create_file(path|content)]",      "takes_arg": True},
    "list_files":    {"func": tool_list_files,     "desc": "List files. [TOOL: list_files(path)]",                "takes_arg": True},
    "read_file":     {"func": tool_read_file,      "desc": "Read file. [TOOL: read_file(path)]",                  "takes_arg": True},
    "plan_complete":  {"func": tool_plan_complete,  "desc": "Mark done. [TOOL: plan_complete(summary)]",           "takes_arg": True},
}


def build_planner_prompt():
    tool_list = "\n".join(f"  - {name}: {info['desc']}" for name, info in TOOLS.items())
    return f"""You are a project generator agent. You create coding projects.

Tools:
{tool_list}

PROCESS:
1. First, output your PLAN as a numbered list of steps
2. Then execute each step using tools
3. Create real, working code files
4. When done use [TOOL: plan_complete(summary)]

RULES:
- Create SIMPLE but WORKING code (no placeholders)
- Use [TOOL: create_file(path|content)] for each file
- Keep files short and practical
- End with plan_complete"""


def parse_tool_calls(text):
    # Match [TOOL: name(args)] -- args can contain escaped parens and quotes
    pattern = r'\[TOOL:\s*(\w+)\(((?:[^)(]|\\.)*)\)\]'
    return re.findall(pattern, text)


def execute_tool(name, arg):
    if name not in TOOLS:
        return f"Unknown tool: {name}"
    tool = TOOLS[name]
    if tool["takes_arg"]:
        # Clean up: models often wrap args in quotes or escape them
        clean_arg = arg.strip().strip('"').strip("'").replace('\\"', '"').replace('\\n', '\n')
        return tool["func"](clean_arg)
    return tool["func"]()


def stream_response(messages):
    """Get a streaming response from the LLM."""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "stream": True,
            "messages": messages,
            "options": {"num_ctx": 2048, "num_predict": 500, "num_gpu": 99},
        }, stream=True, timeout=300)

        reply = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                reply += token
                sys.stdout.write(token)
                sys.stdout.flush()
        print()
        return reply
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return None


def run_planner(goal, max_steps=6):
    """Run the planner agent loop."""
    print(f"\n  GOAL: {goal}")
    print(f"  Output: {OUTPUT_DIR}")
    print("  " + "-" * 50)

    system_msg = {"role": "system", "content": build_planner_prompt()}
    history = [
        {"role": "user", "content": f"Create this project: {goal}"},
    ]
    files_created = []  # track what was created

    for step in range(1, max_steps + 1):
        print(f"\n  --- Step {step}/{max_steps} ---")
        sys.stdout.write("  Agent: ")
        sys.stdout.flush()

        # Keep context small: system + first message + last 4 messages
        trimmed = [system_msg, history[0]] + history[-4:] if len(history) > 5 else [system_msg] + history

        reply = stream_response(trimmed)
        if reply is None:
            break

        history.append({"role": "assistant", "content": reply})

        # Check for completion
        if "plan_complete" in reply.lower() or "PROJECT COMPLETE" in reply:
            tool_calls = parse_tool_calls(reply)
            for func_name, arg in tool_calls:
                result = execute_tool(func_name, arg)
                if "create_file" in func_name:
                    files_created.append(arg.split("|")[0].strip())
                print(f"    -> {func_name}: {result[:150]}")
            print(f"\n  Project finished in {step} steps!")
            return True

        # Execute tool calls
        tool_calls = parse_tool_calls(reply)
        if tool_calls:
            results = []
            for func_name, arg in tool_calls:
                result = execute_tool(func_name, arg)
                if "create_file" in func_name:
                    files_created.append(arg.split("|")[0].strip())
                results.append(f"[{func_name} OK]: {result[:100]}")
                print(f"    -> {func_name}: {result[:150]}")

            # Short follow-up prompt to keep context small
            summary = f"Done. Files so far: {files_created}. Create the next file or [TOOL: plan_complete(done)]."
            history.append({"role": "user", "content": summary})
        else:
            history.append({
                "role": "user",
                "content": "Use tools now. Create files with [TOOL: create_file(path|code)]."
            })

    print(f"\n  Hit max steps ({max_steps}).")
    return False


def show_project_tree(path, prefix=""):
    """Display the generated project structure."""
    if not os.path.exists(path):
        print("  No files generated.")
        return
    items = sorted(os.listdir(path))
    for i, item in enumerate(items):
        full = os.path.join(path, item)
        is_last = i == len(items) - 1
        connector = "`-- " if is_last else "|-- "
        print(f"  {prefix}{connector}{item}")
        if os.path.isdir(full):
            extension = "    " if is_last else "|   "
            show_project_tree(full, prefix + extension)


def main():
    print("\n" + "=" * 60)
    print("  MODULE 05 -- LESSON 3: THE PLANNER AGENT")
    print("  Plan-and-Execute: Break Goals into Steps")
    print("  Model: qwen2.5-coder:7b (local via Ollama)")
    print("=" * 60)

    # Check Ollama
    print("\n  Checking Ollama...")
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("  ERROR: Ollama not running!")
        return

    # Warm up
    print("  Warming up model...")
    try:
        requests.post(OLLAMA_URL, json={
            "model": MODEL, "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_ctx": 512, "num_predict": 5, "num_gpu": 99},
        }, timeout=300)
        print("  Ready!")
    except Exception:
        print("  Warning: warm-up slow, continuing...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Demo --
    print("\n" + "=" * 60)
    print("  DEMO: Watch the agent plan and build a project")
    print("=" * 60)

    demo_goal = "A simple Python calculator with add, subtract, multiply, divide functions and a main menu"
    print(f"\n  Demo goal: {demo_goal}")
    run_planner(demo_goal)

    # Show what was generated
    print("\n" + "=" * 60)
    print("  GENERATED PROJECT STRUCTURE:")
    print("=" * 60)
    show_project_tree(OUTPUT_DIR)

    # -- Interactive --
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Give the agent a project to build")
    print("  Examples:")
    print("    - 'A todo list app in Python'")
    print("    - 'A number guessing game'")
    print("    - 'A contact book with save/load'")
    print("  Type 'quit' to exit, 'show' to see files")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n  Project idea: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if user_input.lower() == "show":
            show_project_tree(OUTPUT_DIR)
            continue

        run_planner(user_input)
        print("\n  Generated files:")
        show_project_tree(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  KEY TAKEAWAYS:")
    print("  1. Planner agents DECOMPOSE big goals into small steps")
    print("  2. Pattern: PLAN -> EXECUTE -> VERIFY -> DELIVER")
    print("  3. Each step uses tools to make real changes")
    print("  4. This is how Devin, AutoGPT, and Claude Code work")
    print("  5. The LLM is the 'brain', tools are the 'hands'")
    print("=" * 60)


if __name__ == "__main__":
    main()
