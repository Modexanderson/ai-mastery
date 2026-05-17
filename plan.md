# Plan: GitHub READMEs + Live RAG Chatbot Demo

## Task 1: GitHub READMEs (6 modules)

Write a professional README.md at the root of each module folder. Each README will have:
- Project title and one-line description
- Screenshot (where available) or demo description
- What it does (features)
- Tech stack
- How to run it
- What I learned

Files to create:
1. `module-02-ai-saas/README.md` -- AI Writing Toolkit (FastAPI + React + Stripe)
2. `module-03-computer-vision/README.md` -- YOLO Security Camera (includes real screenshots from screenshots/ folder)
3. `module-04-voice-audio/README.md` -- Voice Assistant (Whisper + Ollama + TTS)
4. `module-05-ai-agents/README.md` -- AI Agent System (ReAct + Memory + Planner)
5. `module-06-training-models/README.md` -- Fine-Tuning + RAG System (PyTorch + ChromaDB)
6. `module-07-hardware-edge-ai/README.md` -- Edge AI Pipeline (PyTorch to ONNX quantization)

Also delete the boilerplate `module-02-ai-saas/frontend/README.md`.

## Task 2: Live RAG Chatbot Demo on Portfolio Site

### Architecture
- **Frontend:** New `ChatWidget.tsx` component -- floating chat bubble on portfolio site (bottom-right corner)
- **Backend:** New `POST /api/chat` endpoint on the existing Express backend (Cloud Run)
- **AI:** Claude API (Anthropic SDK) with a system prompt containing Mordecai's resume/skills/services as context -- context-grounded responses
- **No vector DB needed** -- the context is small enough to fit in the system prompt. Keeps it simple, no extra infra, and still demonstrates the concept.

### Frontend changes (E:\Programs\my-portfolio)
1. Create `src/components/shared/ChatWidget.tsx`:
   - Floating chat bubble button (bottom-right)
   - Click opens a chat panel (350px wide, 500px tall)
   - Message input + send button
   - Message history with user/assistant bubbles
   - Typing indicator
   - "Ask me anything about Mordecai's work" placeholder
   - Close/minimize button
   - Max 20 messages per session (frontend limit)

2. Add `ChatWidget` to `App.tsx` (rendered on all public pages)

3. Add styles in `src/styles/components/_chat-widget.css`

### Backend changes (E:\Programs\my-portfolio\backend)
1. Install `@anthropic-ai/sdk` dependency
2. Add `POST /api/chat` endpoint to `server.js`:
   - Accept `{ message, history }` body
   - System prompt: Mordecai's bio, skills, services, projects, experience (hardcoded)
   - Call Claude API (claude-haiku-4-5 for speed + cost)
   - Rate limit: 30 requests per IP per hour
   - Return `{ reply }` response
3. Add `ANTHROPIC_API_KEY` to Cloud Run env vars

### Why this approach
- **Simple:** No vector DB, no embeddings, no extra infrastructure
- **Cheap:** Haiku is fast and cheap (~$0.001 per conversation)
- **Impressive:** Visitors can ask "What AI projects has Mordecai built?" and get grounded answers
- **Demonstrates RAG concept:** Context-grounded generation
- **Deployed on existing infra:** Same Cloud Run backend, same Firebase frontend

### Cost controls
- Haiku model (cheapest)
- Rate limit per IP (30/hr)
- Max 20 messages per chat session (frontend)
- Max conversation history of 10 messages sent to API
- System prompt is ~800 tokens (small)
