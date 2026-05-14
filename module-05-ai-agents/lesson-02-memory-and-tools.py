"""
Module 05 -- Lesson 2: Agent Memory & Multi-Tool Chaining
==========================================================
In Lesson 1, our agent had SHORT-TERM memory (conversation history)
but forgot everything when the script ended.

Real agents need:
  1. SHORT-TERM MEMORY -- what happened in THIS conversation (chat history)
  2. LONG-TERM MEMORY  -- facts that persist across conversations (saved to disk)
  3. TOOL CHAINING     -- using output of one tool as input to another

This lesson builds a Personal Assistant Agent that:
  - Remembers facts you tell it (saved to a JSON file)
  - Can search, add, and delete memories
  - Chains tools together (e.g., save a note -> confirm it -> recall it later)
  - Has file tools (read/write files on disk)
  - Maintains a task list

This is how real AI assistants (ChatGPT memory, Claude projects) work.
"""

import requests
import json
import re
import os
import sys
import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "agent_memory.json")
TASKS_FILE = os.path.join(os.path.dirname(__file__), "agent_tasks.json")


# -- LONG-TERM MEMORY SYSTEM ---------------------------------------------------

def load_memory():
    """Load memories from disk."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_memory(memories):
    """Save memories to disk."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def load_tasks():
    """Load tasks from disk."""
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r") as f:
            return json.load(f)
    return []


def save_tasks(tasks):
    """Save tasks to disk."""
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)


# -- TOOLS ---------------------------------------------------------------------

def tool_remember(fact):
    """Save a fact to long-term memory."""
    memories = load_memory()
    entry = {
        "fact": fact,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    memories.append(entry)
    save_memory(memories)
    return f"Saved to memory: '{fact}' (total memories: {len(memories)})"


def tool_recall(query):
    """Search long-term memory for matching facts."""
    memories = load_memory()
    if not memories:
        return "No memories stored yet."
    query_lower = query.lower()
    matches = [m for m in memories if query_lower in m["fact"].lower()]
    if not matches:
        return f"No memories matching '{query}'. Total memories: {len(memories)}"
    lines = [f"Found {len(matches)} memory/memories:"]
    for m in matches:
        lines.append(f"  - {m['fact']} (saved: {m['timestamp']})")
    return "\n".join(lines)


def tool_forget(query):
    """Delete memories matching a query."""
    memories = load_memory()
    query_lower = query.lower()
    before = len(memories)
    memories = [m for m in memories if query_lower not in m["fact"].lower()]
    after = len(memories)
    save_memory(memories)
    removed = before - after
    return f"Removed {removed} memory/memories matching '{query}'. Remaining: {after}"


def tool_list_memories():
    """List all stored memories."""
    memories = load_memory()
    if not memories:
        return "No memories stored."
    lines = [f"All memories ({len(memories)}):"]
    for i, m in enumerate(memories, 1):
        lines.append(f"  {i}. {m['fact']} (saved: {m['timestamp']})")
    return "\n".join(lines)


def tool_add_task(task):
    """Add a task to the to-do list."""
    tasks = load_tasks()
    entry = {
        "task": task,
        "done": False,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    tasks.append(entry)
    save_tasks(tasks)
    return f"Task added: '{task}' (total tasks: {len(tasks)})"


def tool_list_tasks():
    """List all tasks."""
    tasks = load_tasks()
    if not tasks:
        return "No tasks."
    lines = [f"Tasks ({len(tasks)}):"]
    for i, t in enumerate(tasks, 1):
        status = "DONE" if t["done"] else "TODO"
        lines.append(f"  {i}. [{status}] {t['task']}")
    return "\n".join(lines)


def tool_complete_task(number):
    """Mark a task as complete by its number."""
    tasks = load_tasks()
    try:
        idx = int(number) - 1
        if 0 <= idx < len(tasks):
            tasks[idx]["done"] = True
            save_tasks(tasks)
            return f"Completed: '{tasks[idx]['task']}'"
        return f"Invalid task number: {number}. You have {len(tasks)} tasks."
    except ValueError:
        return f"Error: '{number}' is not a valid number."


def tool_get_date():
    """Get current date and time."""
    now = datetime.datetime.now()
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"


def tool_write_file(args):
    """Write content to a file. Format: filename|content"""
    if "|" not in args:
        return "Error: use format filename|content"
    filename, content = args.split("|", 1)
    filename = filename.strip()
    # Safety: only allow writing in the module directory
    safe_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(safe_path, "w") as f:
            f.write(content)
        return f"Written to {filename} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_read_file(filename):
    """Read a file from the module directory."""
    safe_path = os.path.join(os.path.dirname(__file__), filename.strip())
    try:
        with open(safe_path, "r") as f:
            content = f.read()
        return f"Contents of {filename}:\n{content[:500]}"
    except FileNotFoundError:
        return f"File not found: {filename}"
    except Exception as e:
        return f"Error reading file: {e}"


# -- TOOL REGISTRY --------------------------------------------------------------
TOOLS = {
    "remember":       {"func": tool_remember,       "desc": "Save a fact. [TOOL: remember(the fact)]",             "takes_arg": True},
    "recall":         {"func": tool_recall,          "desc": "Search memory. [TOOL: recall(search term)]",         "takes_arg": True},
    "forget":         {"func": tool_forget,          "desc": "Delete memories. [TOOL: forget(search term)]",       "takes_arg": True},
    "list_memories":  {"func": tool_list_memories,   "desc": "Show all memories. [TOOL: list_memories()]",         "takes_arg": False},
    "add_task":       {"func": tool_add_task,        "desc": "Add a task. [TOOL: add_task(description)]",          "takes_arg": True},
    "list_tasks":     {"func": tool_list_tasks,      "desc": "Show all tasks. [TOOL: list_tasks()]",               "takes_arg": False},
    "complete_task":  {"func": tool_complete_task,    "desc": "Complete task by number. [TOOL: complete_task(1)]",  "takes_arg": True},
    "get_date":       {"func": tool_get_date,        "desc": "Get current date/time. [TOOL: get_date()]",          "takes_arg": False},
    "write_file":     {"func": tool_write_file,      "desc": "Write file. [TOOL: write_file(name|content)]",      "takes_arg": True},
    "read_file":      {"func": tool_read_file,       "desc": "Read file. [TOOL: read_file(name)]",                "takes_arg": True},
}


def build_system_prompt():
    """Build the system prompt with memory context."""
    tool_list = "\n".join(f"  - {name}: {info['desc']}" for name, info in TOOLS.items())

    # Load existing memories as context
    memories = load_memory()
    memory_context = ""
    if memories:
        facts = [m["fact"] for m in memories[-10:]]  # last 10 memories
        memory_context = f"\n\nYou remember these facts about the user:\n" + "\n".join(f"  - {f}" for f in facts)

    return f"""You are a personal assistant with long-term memory and tools.

Tools:
{tool_list}

To use a tool write [TOOL: name(arg)] in your reply.
You can use multiple tools in one reply.

IMPORTANT RULES:
- When the user tells you a fact about themselves, use [TOOL: remember(fact)] to save it
- When asked about something you might know, use [TOOL: recall(topic)] to check memory
- Keep responses SHORT (1-3 sentences plus any tool calls)
- Be helpful and conversational{memory_context}"""


def parse_tool_calls(text):
    """Extract all [TOOL: func(args)] patterns from text."""
    pattern = r'\[TOOL:\s*(\w+)\(([^)]*)\)\]'
    return re.findall(pattern, text)


def execute_tool(name, arg):
    """Execute a tool by name."""
    if name not in TOOLS:
        return f"Unknown tool: {name}"
    tool = TOOLS[name]
    if tool["takes_arg"]:
        return tool["func"](arg.strip())
    else:
        return tool["func"]()


def chat_with_agent(user_input, history):
    """One turn of the agent loop with tool execution."""
    history.append({"role": "user", "content": user_input})

    # Keep history manageable (system + last 20 messages)
    system_msg = {"role": "system", "content": build_system_prompt()}
    trimmed = [system_msg] + history[-20:]

    # Get LLM response (streaming)
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "stream": True,
            "messages": trimmed,
            "options": {"num_ctx": 2048, "num_predict": 300, "num_gpu": 99},
        }, stream=True, timeout=300)

        reply = ""
        sys.stdout.write("\n  Assistant: ")
        sys.stdout.flush()
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                reply += token
                sys.stdout.write(token)
                sys.stdout.flush()
        print()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return history

    history.append({"role": "assistant", "content": reply})

    # Execute any tool calls
    tool_calls = parse_tool_calls(reply)
    if tool_calls:
        results = []
        for func_name, arg in tool_calls:
            result = execute_tool(func_name, arg)
            results.append(f"[Result of {func_name}]: {result}")
            print(f"    -> {func_name}: {result[:150]}")

        # Send results back and get follow-up response
        tool_msg = "\n".join(results)
        history.append({"role": "user", "content": tool_msg})

        trimmed2 = [system_msg] + history[-20:]
        try:
            response2 = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "stream": True,
                "messages": trimmed2,
                "options": {"num_ctx": 2048, "num_predict": 200, "num_gpu": 99},
            }, stream=True, timeout=300)

            reply2 = ""
            sys.stdout.write("  Assistant: ")
            sys.stdout.flush()
            for line in response2.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    reply2 += token
                    sys.stdout.write(token)
                    sys.stdout.flush()
            print()
            history.append({"role": "assistant", "content": reply2})
        except Exception as e:
            print(f"\n  ERROR on follow-up: {e}")

    return history


def main():
    print("\n" + "=" * 60)
    print("  MODULE 05 -- LESSON 2: MEMORY & MULTI-TOOL CHAINING")
    print("  Personal Assistant with Long-Term Memory")
    print("  Model: qwen2.5-coder:7b (local via Ollama)")
    print("=" * 60)

    # Check Ollama
    print("\n  Checking Ollama...")
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("  ERROR: Ollama not running! Start it with 'ollama serve'")
        return

    # Warm up model
    print("  Warming up model...")
    try:
        requests.post(OLLAMA_URL, json={
            "model": MODEL, "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_ctx": 512, "num_predict": 5, "num_gpu": 99},
        }, timeout=300)
        print("  Model ready!")
    except Exception as e:
        print(f"  Warning: warm-up slow ({e}), continuing anyway...")

    # Show existing memories
    memories = load_memory()
    if memories:
        print(f"\n  Loaded {len(memories)} memories from previous sessions:")
        for m in memories[-5:]:
            print(f"    - {m['fact']}")
    else:
        print("\n  No previous memories. Start telling me things!")

    print("\n" + "=" * 60)
    print("  THINGS TO TRY:")
    print("  - 'My name is Cyborg'         (agent saves to memory)")
    print("  - 'What is my name?'           (agent recalls from memory)")
    print("  - 'Add task: learn PyTorch'    (agent manages tasks)")
    print("  - 'What are my tasks?'         (agent lists tasks)")
    print("  - 'Write a shopping list to a file'  (tool chaining)")
    print("  Type 'quit' to exit, 'clear' to reset memories")
    print("=" * 60)

    history = []

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if user_input.lower() == "clear":
            save_memory([])
            save_tasks([])
            history = []
            print("  All memories and tasks cleared!")
            continue
        if user_input.lower() == "memories":
            memories = load_memory()
            if memories:
                for m in memories:
                    print(f"    - {m['fact']} ({m['timestamp']})")
            else:
                print("  No memories.")
            continue

        history = chat_with_agent(user_input, history)

    # Summary
    memories = load_memory()
    tasks = load_tasks()
    print("\n" + "=" * 60)
    print(f"  Session ended.")
    print(f"  Memories saved: {len(memories)} (stored in agent_memory.json)")
    print(f"  Tasks saved: {len(tasks)} (stored in agent_tasks.json)")
    print()
    print("  KEY TAKEAWAYS:")
    print("  1. SHORT-TERM memory = conversation history (lost on exit)")
    print("  2. LONG-TERM memory = saved to disk (persists forever)")
    print("  3. Tool chaining = output of one tool feeds into the next")
    print("  4. This is how ChatGPT 'memory' and Claude 'projects' work")
    print("  5. Real apps use databases (PostgreSQL, Redis) instead of JSON")
    print("=" * 60)


if __name__ == "__main__":
    main()
