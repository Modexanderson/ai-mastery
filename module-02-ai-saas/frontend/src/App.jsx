/**
 * Module 02 — Lesson 3: Auth + Usage Limits
 * ==========================================
 * NEW CONCEPTS:
 *   1. User ID — generated once, stored in localStorage, sent with every request
 *   2. Usage tracking — the backend counts uses, frontend displays remaining
 *   3. Usage gate — when limit hit, UI shows upgrade prompt instead of processing
 *   4. useEffect — fetch usage on load and after each request
 *
 * This is the standard SaaS pattern:
 *   Free tier (attracts users) → limit reached → upgrade prompt → paid tier
 */

import { useState, useEffect } from 'react'

const API_URL = 'http://localhost:8001/api'

// ── Generate a unique user ID (stored in browser) ───────────────────────────
// In production this would be a real auth system (Google login, email, etc.)
// For now, we generate a random ID and store it in localStorage.
// localStorage persists across browser refreshes — like a simple cookie.
function getUserId() {
  let id = localStorage.getItem('ai_toolkit_user_id')
  if (!id) {
    id = 'user_' + Math.random().toString(36).substring(2, 15)
    localStorage.setItem('ai_toolkit_user_id', id)
  }
  return id
}

const USER_ID = getUserId()

const TOOLS = [
  { id: 'rewrite',     name: 'Rewrite',      icon: '✏️',  description: 'Improve clarity and flow',       needsOption: false },
  { id: 'summarize',   name: 'Summarize',     icon: '📋',  description: 'Condense into key points',       needsOption: false },
  { id: 'fix_grammar', name: 'Fix Grammar',   icon: '🔤',  description: 'Fix spelling and grammar',       needsOption: false },
  { id: 'translate',   name: 'Translate',      icon: '🌍',  description: 'Translate to another language',   needsOption: true, optionLabel: 'Target language', optionPlaceholder: 'e.g. Spanish, French, Japanese' },
  { id: 'tone',        name: 'Change Tone',    icon: '🎭',  description: 'Shift the writing style',        needsOption: true, optionLabel: 'Desired tone', optionPlaceholder: 'e.g. formal, casual, persuasive' },
]

function App() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [selectedTool, setSelectedTool] = useState('rewrite')
  const [option, setOption] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // ── NEW: Usage state ────────────────────────────────────────────────────
  const [usage, setUsage] = useState(null)  // { used, limit, remaining, is_pro }

  const currentTool = TOOLS.find(t => t.id === selectedTool)

  // ── Fetch usage on load (and after returning from checkout) ─────────────
  useEffect(() => {
    fetchUsage()

    // Check if user just returned from checkout page
    const params = new URLSearchParams(window.location.search)
    if (params.get('upgraded') === 'true') {
      // Clean the URL so it doesn't trigger again on refresh
      window.history.replaceState({}, '', window.location.pathname)
      fetchUsage()  // refresh to get pro status
    }
  }, [])

  async function fetchUsage() {
    try {
      const res = await fetch(`${API_URL}/usage?user_id=${USER_ID}`)
      const data = await res.json()
      setUsage(data)
    } catch {
      // Backend might not be running yet
    }
  }

  // ── Handle: call the backend with streaming ─────────────────────────────
  async function handleProcess() {
    if (!inputText.trim()) return

    // Check if user is out of free uses
    if (usage && !usage.is_pro && usage.remaining <= 0) {
      setError('Daily limit reached! Upgrade to Pro for unlimited access.')
      return
    }

    setLoading(true)
    setOutputText('')
    setError('')

    try {
      const response = await fetch(`${API_URL}/stream?user_id=${USER_ID}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool: selectedTool,
          text: inputText,
          option: option,
        }),
      })

      // ── Check for rate limit response ─────────────────────────────
      if (response.status === 429) {
        const data = await response.json()
        setError(`Daily limit reached (${data.used}/${data.limit}). Upgrade to Pro!`)
        setLoading(false)
        fetchUsage()  // refresh usage display
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value)
        const lines = text.split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') break
            try {
              const parsed = JSON.parse(data)
              accumulated += parsed.token
              setOutputText(accumulated)
            } catch { /* skip */ }
          }
        }
      }
    } catch (err) {
      setError('Failed to connect to backend. Is the server running on port 8000?')
    } finally {
      setLoading(false)
      fetchUsage()  // ← Refresh usage count after each request
    }
  }

  function handleCopy() {
    navigator.clipboard.writeText(outputText)
  }

  // ── Handle: upgrade to Pro ──────────────────────────────────────────────
  // Calls the backend to get a checkout URL, then redirects the user.
  // In production, this would redirect to Stripe's hosted checkout page.
  async function handleUpgrade() {
    try {
      const res = await fetch(`${API_URL}/checkout?user_id=${USER_ID}`)
      const data = await res.json()
      if (data.checkout_url) {
        window.location.href = data.checkout_url
      }
    } catch {
      setError('Failed to start checkout')
    }
  }

  // ── Derived values for the usage bar ──────────────────────────────────
  const usagePercent = usage && !usage.is_pro
    ? Math.min(100, (usage.used / usage.limit) * 100)
    : 0
  const isLimitReached = usage && !usage.is_pro && usage.remaining <= 0

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">

      {/* Header */}
      <header className="border-b border-slate-700 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold text-white">
            AI Writing Toolkit
          </h1>
          {/* Usage indicator in header */}
          {usage && !usage.is_pro && (
            <div className="flex items-center gap-3">
              <div className="text-sm text-slate-400">
                {usage.remaining}/{usage.limit} uses left today
              </div>
              <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    usagePercent > 80 ? 'bg-red-500' : usagePercent > 50 ? 'bg-yellow-500' : 'bg-indigo-500'
                  }`}
                  style={{ width: `${usagePercent}%` }}
                />
              </div>
            </div>
          )}
          {usage?.is_pro && (
            <span className="text-sm text-emerald-400 font-medium">PRO</span>
          )}
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">

        {/* Upgrade banner — shows when limit reached */}
        {isLimitReached && (
          <div className="mb-6 p-4 bg-indigo-900/50 border border-indigo-500 rounded-lg text-center">
            <p className="text-indigo-200 font-medium mb-1">
              You've used all {usage.limit} free uses for today
            </p>
            <p className="text-indigo-300 text-sm mb-3">
              Upgrade to Pro for unlimited access — no daily limits
            </p>
            <button
              onClick={handleUpgrade}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-all shadow-lg shadow-indigo-500/25"
            >
              Upgrade to Pro — $9/month
            </button>
          </div>
        )}

        {/* Tool selector */}
        <div className="mb-6">
          <label className="block text-sm text-slate-400 mb-3">Select a tool</label>
          <div className="flex flex-wrap gap-2">
            {TOOLS.map(tool => (
              <button
                key={tool.id}
                onClick={() => { setSelectedTool(tool.id); setOption(''); setOutputText(''); }}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedTool === tool.id
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/25'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {tool.icon} {tool.name}
              </button>
            ))}
          </div>
          <p className="text-sm text-slate-500 mt-2">{currentTool?.description}</p>
        </div>

        {/* Option input */}
        {currentTool?.needsOption && (
          <div className="mb-4">
            <label className="block text-sm text-slate-400 mb-1">
              {currentTool.optionLabel}
            </label>
            <input
              type="text"
              value={option}
              onChange={e => setOption(e.target.value)}
              placeholder={currentTool.optionPlaceholder}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>
        )}

        {/* Input + Output */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Input</label>
            <textarea
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              placeholder="Paste your text here..."
              rows={10}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 resize-none"
            />
            <div className="text-xs text-slate-500 mt-1">
              {inputText.length} characters
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-sm text-slate-400">Output</label>
              {outputText && (
                <button onClick={handleCopy} className="text-xs text-indigo-400 hover:text-indigo-300">
                  Copy to clipboard
                </button>
              )}
            </div>
            <div className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-3 min-h-[260px] whitespace-pre-wrap text-sm">
              {loading && !outputText && (
                <span className="text-slate-500 animate-pulse">Thinking...</span>
              )}
              {outputText || (!loading && (
                <span className="text-slate-500">Result will appear here...</span>
              ))}
            </div>
            {outputText && (
              <div className="text-xs text-slate-500 mt-1">{outputText.length} characters</div>
            )}
          </div>
        </div>

        {/* Process button */}
        <button
          onClick={handleProcess}
          disabled={loading || !inputText.trim() || isLimitReached}
          className={`w-full py-3 rounded-lg font-medium text-white transition-all ${
            loading || !inputText.trim() || isLimitReached
              ? 'bg-slate-700 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-500/25'
          }`}
        >
          {isLimitReached
            ? 'Daily limit reached — Upgrade to Pro'
            : loading
              ? 'Processing...'
              : `${currentTool?.icon} ${currentTool?.name}`
          }
        </button>

        {error && (
          <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-300 text-sm">
            {error}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
