"""
Module 02 — AI Writing Toolkit — Backend API
==============================================
Lesson 1: Basic API (FastAPI + Ollama)
Lesson 2: Frontend (React + Tailwind)
Lesson 3: Auth + Usage Limits
Lesson 4: Payments (simulated Stripe) ← NEW

CONCEPTS:
  - Usage tracking with daily limits
  - Payment flow: checkout → payment → webhook → unlock pro
  - In production, replace /api/checkout and /api/webhook with real Stripe
  - The pattern is identical — only the payment processor changes

In production you'd use a real database (Postgres, SQLite) and real auth
(Google OAuth, email/password). We use in-memory dicts to keep it simple —
the PATTERN is what matters, not the storage.

Run:
  cd module-02-ai-saas/backend
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
import requests
import json
from datetime import date
from collections import defaultdict

# ── Create the FastAPI app ───────────────────────────────────────────────────
app = FastAPI(title="AI Writing Toolkit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"

# ══════════════════════════════════════════════════════════════════════════════
# USAGE TRACKING (Lesson 3)
# ══════════════════════════════════════════════════════════════════════════════
# In-memory storage. In production, this would be a database.
# Structure: { "user_id": { "2026-05-12": 3 } }
#   → user "abc123" has used 3 requests on May 12th
#
# Why per-day? Because the free tier resets daily. This is common in SaaS:
#   - ChatGPT free: limited messages per day
#   - Many APIs: X requests per day/month

FREE_TIER_LIMIT = 2  # requests per day (lowered for testing, set to 10 in production)

usage_store: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

# Track which users are "pro" (paid). In production = database + Stripe webhook.
pro_users: set[str] = set()


def check_usage(user_id: str) -> dict:
    """Check how many uses a user has left today."""
    today = str(date.today())
    used = usage_store[user_id][today]
    is_pro = user_id in pro_users

    if is_pro:
        return {"allowed": True, "used": used, "limit": None, "remaining": None, "is_pro": True}

    remaining = max(0, FREE_TIER_LIMIT - used)
    return {
        "allowed": remaining > 0,
        "used": used,
        "limit": FREE_TIER_LIMIT,
        "remaining": remaining,
        "is_pro": False,
    }


def increment_usage(user_id: str):
    """Record one API use for today."""
    today = str(date.today())
    usage_store[user_id][today] += 1


# ══════════════════════════════════════════════════════════════════════════════
# AI Tools (same as before)
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = {
    "rewrite": {
        "name": "Rewrite",
        "system_prompt": (
            "You are an expert editor. Rewrite the user's text to be clearer, "
            "more concise, and more professional. Preserve the original meaning. "
            "Return ONLY the rewritten text, no explanations or labels."
        ),
    },
    "summarize": {
        "name": "Summarize",
        "system_prompt": (
            "You are an expert summarizer. Condense the user's text into 2-4 "
            "bullet points capturing the key ideas. Use '•' for bullets. "
            "Return ONLY the bullet points, no intro or outro."
        ),
    },
    "translate": {
        "name": "Translate",
        "system_prompt": (
            "You are a professional translator. The user will provide text and "
            "a target language in the format: 'Translate to [language]: [text]'. "
            "Return ONLY the translated text. No explanations."
        ),
    },
    "tone": {
        "name": "Change Tone",
        "system_prompt": (
            "You are a writing style expert. The user will provide text and a "
            "desired tone (formal, casual, persuasive, friendly, etc.) in the "
            "format: 'Make this [tone]: [text]'. "
            "Rewrite the text in that tone. Return ONLY the rewritten text."
        ),
    },
    "fix_grammar": {
        "name": "Fix Grammar",
        "system_prompt": (
            "You are a grammar expert. Fix all grammar, spelling, and punctuation "
            "errors in the user's text. Preserve the original style and meaning. "
            "Return ONLY the corrected text."
        ),
    },
}


class ToolRequest(BaseModel):
    tool: str
    text: str
    option: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/tools")
def list_tools():
    return {
        key: {"name": val["name"]}
        for key, val in TOOLS.items()
    }


@app.get("/api/usage")
def get_usage(user_id: str = ""):
    """
    Check usage for the current user.
    User ID comes as a query param: /api/usage?user_id=abc123
    Simpler than custom headers — no CORS preflight issues.
    """
    if not user_id:
        return {"error": "Missing user_id parameter"}
    return check_usage(user_id)


@app.post("/api/stream")
def stream_text(req: ToolRequest, user_id: str = ""):
    """Process text with streaming — now with usage checking."""

    if not user_id:
        return JSONResponse(status_code=401, content={"error": "Missing user ID"})

    status = check_usage(user_id)
    if not status["allowed"]:
        return JSONResponse(status_code=429, content={
            "error": "Daily limit reached",
            "used": status["used"],
            "limit": status["limit"],
        })

    if req.tool not in TOOLS:
        return JSONResponse(status_code=400, content={"error": f"Unknown tool: {req.tool}"})

    tool = TOOLS[req.tool]

    user_message = req.text
    if req.option:
        if req.tool == "translate":
            user_message = f"Translate to {req.option}: {req.text}"
        elif req.tool == "tone":
            user_message = f"Make this {req.option}: {req.text}"

    # ── Increment usage BEFORE processing ────────────────────────────
    # Why before? So users can't spam requests during the streaming delay.
    increment_usage(user_id)

    def generate():
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "stream": True,
                "messages": [
                    {"role": "system", "content": tool["system_prompt"]},
                    {"role": "user", "content": user_message},
                ]
            },
            stream=True
        )

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                yield f"data: [DONE]\n\n"
                break
            token = chunk["message"]["content"]
            yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ══════════════════════════════════════════════════════════════════════════════
# PAYMENTS (Lesson 4)
# ══════════════════════════════════════════════════════════════════════════════
# In production, this flow uses Stripe:
#   1. Frontend calls POST /api/checkout → backend creates Stripe Checkout Session
#   2. User is redirected to Stripe's hosted payment page
#   3. User pays → Stripe calls your webhook → you mark user as pro
#
# We simulate this with a simple HTML page that has a "Pay" button.
# The PATTERN is identical to real Stripe — swap in the SDK and it works.

@app.get("/api/checkout")
def create_checkout(user_id: str = ""):
    """
    Simulate Stripe Checkout.
    In production: stripe.checkout.Session.create(...)
    Returns a URL the frontend redirects to.
    """
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id"})

    # In production, this would be a Stripe checkout URL
    # Stripe hosts the payment page — you never touch credit card data
    checkout_url = f"http://localhost:8001/checkout-page?user_id={user_id}"
    return {"checkout_url": checkout_url}


@app.get("/checkout-page")
def checkout_page(user_id: str = ""):
    """
    Simulated payment page.
    In production, Stripe hosts this page — you never build it yourself.
    This exists only so you can see the full flow working.
    """
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Checkout — AI Writing Toolkit</title>
        <style>
            body {{
                font-family: system-ui, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }}
            .card {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 40px;
                max-width: 400px;
                text-align: center;
            }}
            h1 {{ color: white; font-size: 24px; margin-bottom: 8px; }}
            .price {{ font-size: 36px; color: #818cf8; font-weight: bold; margin: 20px 0; }}
            .price span {{ font-size: 16px; color: #94a3b8; }}
            ul {{ text-align: left; padding-left: 20px; color: #94a3b8; line-height: 2; }}
            button {{
                background: #4f46e5;
                color: white;
                border: none;
                padding: 14px 40px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                margin-top: 20px;
            }}
            button:hover {{ background: #4338ca; }}
            .test {{ color: #f59e0b; font-size: 12px; margin-top: 16px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>AI Writing Toolkit Pro</h1>
            <p style="color: #94a3b8;">Unlimited access to all tools</p>
            <div class="price">$9<span>/month</span></div>
            <ul>
                <li>Unlimited daily uses</li>
                <li>All 5 writing tools</li>
                <li>Priority processing</li>
                <li>Cancel anytime</li>
            </ul>
            <button onclick="handlePayment()">Pay $9.00</button>
            <p class="test">TEST MODE — no real payment</p>
        </div>
        <script>
            function handlePayment() {{
                // Simulate payment processing
                document.querySelector('button').textContent = 'Processing...';
                document.querySelector('button').disabled = true;

                // Call the webhook (simulates what Stripe does after payment)
                fetch('http://localhost:8001/api/webhook', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        event: 'payment_success',
                        user_id: '{user_id}'
                    }})
                }})
                .then(r => r.json())
                .then(data => {{
                    if (data.status === 'ok') {{
                        document.querySelector('button').textContent = 'Payment Successful!';
                        document.querySelector('button').style.background = '#059669';
                        // Redirect back to app after 1.5s
                        setTimeout(() => {{
                            window.location.href = 'http://localhost:5173?upgraded=true';
                        }}, 1500);
                    }}
                }});
            }}
        </script>
    </body>
    </html>
    """)


@app.post("/api/webhook")
def payment_webhook(data: dict):
    """
    Simulated Stripe webhook.

    In production, Stripe sends a POST to this endpoint after payment:
      1. Verify the webhook signature (prevent fake requests)
      2. Check the event type (payment_success, subscription_cancelled, etc.)
      3. Update the user's status in your database

    This is the critical endpoint — it's how your app KNOWS someone paid.
    """
    if data.get("event") == "payment_success":
        user_id = data.get("user_id", "")
        if user_id:
            pro_users.add(user_id)    # ← Mark user as pro
            return {"status": "ok", "message": f"User {user_id} upgraded to Pro"}

    return {"status": "ignored"}


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL}
