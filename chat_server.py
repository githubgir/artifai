"""
chat_server.py
--------------
Test chat UI for the Artefact Registry.
Single file: FastAPI backend + React SPA served inline.

Usage:
    pip install fastapi uvicorn anthropic pandas numpy openpyxl pyarrow python-multipart
    export ANTHROPIC_API_KEY=sk-...
    python chat_server.py

Then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import anthropic
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from artifact_registry import ArtefactRegistry, ExecutionResult, Provenance, run_analysis, ingest_file
from fixtures import make_all_fixtures

# ─────────────────────────────────────────────────────────────────────────────
# Session store (in-memory, single-user test server)
# ─────────────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.registry = ArtefactRegistry()
        self.history: list[dict] = []
        self.pending_results: dict[str, Any] = {}   # execution_id → output

SESSIONS: dict[str, Session] = {}

def get_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session()
    return SESSIONS[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are an analytical assistant with access to a registry of tabular datasets.

## Available Artefacts
{manifest}

## Your Capabilities
You can analyse these artefacts by generating Python code. When you want to run analysis:
1. Briefly describe what you are going to do in plain English (the "intent")
2. Then write the code in a ```python block

## Code Rules
- Only use: pd (pandas), np (numpy), and the named artefact variables
- No imports of any kind
- Assign your final output to a variable called `result`
- You may only reference artefacts that appear in the manifest above

## Response Format for Analysis
When proposing code, structure your response as:

INTENT: <one sentence plain English description>
```python
<your code here>
```

For questions that don't need code, just answer directly.
After analysis runs, interpret the result for a non-technical user.
"""

def llm_describe_file(filename: str, df) -> str:
    """Call the LLM to describe an uploaded file."""
    import pandas as pd
    sample = df.head(3).to_string()
    schema = df.dtypes.to_string()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                f"File: '{filename}'\nSchema:\n{schema}\nFirst 3 rows:\n{sample}\n\n"
                "Describe in one plain-English sentence what this file contains "
                "and what it could be used for in data analysis. No preamble."
            )
        }]
    )
    return msg.content[0].text.strip()


def parse_intent_and_code(text: str) -> tuple[str | None, str | None]:
    """Extract INTENT and code block from LLM response."""
    import re
    intent = None
    code = None

    intent_match = re.search(r"INTENT:\s*(.+?)(?:\n|$)", text)
    if intent_match:
        intent = intent_match.group(1).strip()

    code_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()

    return intent, code


def extract_requested_artefacts(code: str, available: list[str]) -> list[str]:
    """Find which artefact names appear in the code."""
    return [name for name in available if name in code]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Artefact Registry Chat")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(REACT_APP)


@app.get("/api/manifest")
def get_manifest(session_id: str = "default"):
    session = get_session(session_id)
    artefacts = []
    for name, art in session.registry._store.items():
        artefacts.append({
            "name": name,
            "provenance": art.provenance.value,
            "dtype_summary": art.dtype_summary,
            "description": art.description,
            "parents": art.parent_artefacts,
        })
    return {"artefacts": artefacts, "manifest": session.registry.manifest()}


@app.post("/api/load-test-data")
def load_test_data(session_id: str = "default"):
    session = get_session(session_id)
    fixtures = make_all_fixtures()
    descriptions = {
        "universe":         "Static universe: 1 000 stocks with stock_name, region, industry, ISIN, freefloat, SEDOL, CUSIP",
        "stock_returns":    "Daily total return matrix: 5 years of business days × 1 000 instruments, GARCH-like vol clustering",
        "position_history": "Monthly rebalancing weights (benchmark_weight, index_weight, mcap_usd) per instrument; weights sum to 1.0 per date",
        "factor_exposures": "Factor loading matrix at monthly dates: Value, Quality, Momentum, Low Vol, Size",
        "user_signals":     "Weekly proprietary signals per ISIN (Proprietary 1, Proprietary 2); dates misaligned with rebalancing dates",
    }
    for name, data in fixtures.items():
        session.registry.register(
            name=name,
            data=data,
            description=descriptions[name],
            provenance=Provenance.ENGINE,
        )
    return {"message": f"Loaded {len(fixtures)} test artefacts", "manifest": session.registry.manifest()}


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form(default="default"),
    file: UploadFile | None = File(default=None),
):
    session = get_session(session_id)
    extra_context = ""

    # ── Handle file upload ────────────────────────────────────────────────────
    if file and file.filename:
        raw = await file.read()
        try:
            ingested = ingest_file(
                file.filename, raw,
                llm_describe=llm_describe_file,
            )
            session.registry.register(
                name=ingested.suggested_artefact_name,
                data=ingested.df,
                description=ingested.description,
                provenance=Provenance.USER_UPLOAD,
            )
            extra_context = (
                f"\n[File uploaded: `{ingested.suggested_artefact_name}` — "
                f"{ingested.description}. "
                f"Classification: {ingested.classification}. "
                f"Shape: {ingested.df.shape}]"
            )
        except Exception as e:
            extra_context = f"\n[File upload failed: {e}]"

    # ── Build messages ─────────────────────────────────────────────────────────
    system = SYSTEM_PROMPT.format(manifest=session.registry.manifest())
    session.history.append({"role": "user", "content": message + extra_context})

    # ── Call LLM ──────────────────────────────────────────────────────────────
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system,
        messages=session.history,
    )
    assistant_text = response.content[0].text
    session.history.append({"role": "assistant", "content": assistant_text})

    # ── Parse for code ────────────────────────────────────────────────────────
    intent, code = parse_intent_and_code(assistant_text)

    pending_id = None
    ast_error = None

    if code:
        from artifact_registry import _ast_check
        violations = _ast_check(code)
        if violations:
            ast_error = "Code blocked: " + "; ".join(violations)
        else:
            pending_id = str(uuid.uuid4())
            session.pending_results[pending_id] = {
                "code": code,
                "artefacts": extract_requested_artefacts(code, session.registry.names()),
            }

    return {
        "text": assistant_text,
        "intent": intent,
        "has_code": code is not None,
        "pending_execution_id": pending_id,
        "ast_error": ast_error,
        "manifest": session.registry.manifest(),
    }


@app.post("/api/execute")
def execute(body: dict):
    session_id = body.get("session_id", "default")
    execution_id = body["execution_id"]
    session = get_session(session_id)

    pending = session.pending_results.get(execution_id)
    if not pending:
        return {"success": False, "error": "Unknown execution ID"}

    result: ExecutionResult = run_analysis(
        code=pending["code"],
        registry=session.registry,
        requested_artefacts=pending["artefacts"],
        timeout=30,
    )

    output_preview = None
    output_type = None
    import pandas as pd
    import numpy as np

    if result.success and result.output is not None:
        if isinstance(result.output, pd.DataFrame):
            output_type = "dataframe"
            output_preview = {
                "columns": list(result.output.columns),
                "index": [str(i) for i in result.output.index[:10]],
                "data": result.output.head(10).values.tolist(),
                "shape": list(result.output.shape),
            }
        elif isinstance(result.output, pd.Series):
            output_type = "series"
            output_preview = {
                "name": result.output.name,
                "index": [str(i) for i in result.output.index[:10]],
                "values": result.output.head(10).tolist(),
                "len": len(result.output),
            }
        elif isinstance(result.output, np.ndarray):
            output_type = "ndarray"
            output_preview = {"shape": list(result.output.shape), "data": result.output.tolist()}
        else:
            output_type = "scalar"
            output_preview = str(result.output)

        # Store output for potential registration
        session.pending_results[execution_id]["output"] = result.output

    return {
        "success": result.success,
        "output_type": output_type,
        "output_preview": output_preview,
        "stdout": result.stdout,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.post("/api/register-derived")
def register_derived(body: dict):
    session_id = body.get("session_id", "default")
    name = body["name"]
    description = body["description"]
    execution_id = body["execution_id"]

    session = get_session(session_id)
    pending = session.pending_results.get(execution_id)
    if not pending or "output" not in pending:
        return {"success": False, "error": "No output to register"}

    session.registry.register(
        name=name,
        data=pending["output"],
        description=description,
        provenance=Provenance.DERIVED,
        parents=pending.get("artefacts", []),
    )
    return {"success": True, "manifest": session.registry.manifest()}


@app.delete("/api/registry")
def clear_registry(session_id: str = "default"):
    session = get_session(session_id)
    session.registry.clear()
    session.history.clear()
    session.pending_results.clear()
    return {"message": "Registry and history cleared"}


# ─────────────────────────────────────────────────────────────────────────────
# Inline React SPA
# ─────────────────────────────────────────────────────────────────────────────

REACT_APP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Artefact Registry Chat</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.development.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.development.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; height: 100vh; overflow: hidden; }
.app { display: flex; height: 100vh; }

/* Manifest panel */
.manifest-panel { width: 300px; min-width: 300px; background: #161b22; border-right: 1px solid #30363d;
                  display: flex; flex-direction: column; }
.panel-header { padding: 14px 16px; background: #1c2128; border-bottom: 1px solid #30363d;
                font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
                color: #7d8590; }
.panel-actions { padding: 10px 12px; display: flex; gap: 8px; border-bottom: 1px solid #30363d; }
.btn-sm { padding: 5px 10px; border: 1px solid #30363d; border-radius: 6px; cursor: pointer;
          font-size: 12px; background: #21262d; color: #c9d1d9; transition: all 0.15s; }
.btn-sm:hover { background: #30363d; }
.btn-sm.danger { border-color: #f8514966; color: #f85149; }
.btn-sm.danger:hover { background: #f8514920; }
.manifest-list { flex: 1; overflow-y: auto; padding: 8px; }
.artefact-row { padding: 8px 10px; border-radius: 6px; margin-bottom: 4px; cursor: pointer;
                border: 1px solid transparent; transition: all 0.15s; }
.artefact-row:hover { background: #21262d; border-color: #30363d; }
.artefact-row.engine { border-left: 3px solid #388bfd; }
.artefact-row.user_upload { border-left: 3px solid #d29922; }
.artefact-row.derived { border-left: 3px solid #3fb950; }
.artefact-name { font-size: 13px; font-weight: 600; font-family: monospace; color: #e2e8f0; }
.artefact-meta { font-size: 11px; color: #7d8590; margin-top: 2px; }
.artefact-desc { font-size: 11px; color: #8b949e; margin-top: 3px; line-height: 1.4; }
.artefact-parents { font-size: 10px; color: #3fb950; margin-top: 2px; }

/* Chat area */
.chat-area { flex: 1; display: flex; flex-direction: column; }
.chat-header { padding: 14px 20px; background: #1c2128; border-bottom: 1px solid #30363d;
               font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
               color: #7d8590; }
.messages { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
.message { max-width: 80%; }
.message.user { align-self: flex-end; }
.message.assistant { align-self: flex-start; }
.message-bubble { padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.6; }
.user .message-bubble { background: #1f6feb; color: #fff; border-bottom-right-radius: 3px; }
.assistant .message-bubble { background: #21262d; border: 1px solid #30363d; border-bottom-left-radius: 3px; }
.message-bubble pre { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
                      padding: 10px; margin: 8px 0; overflow-x: auto; font-size: 12px; }

/* Intent card */
.intent-card { background: #1c2128; border: 1px solid #388bfd44; border-radius: 8px;
               padding: 12px 14px; margin-top: 8px; }
.intent-label { font-size: 11px; color: #388bfd; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.06em; margin-bottom: 6px; }
.intent-text { font-size: 13px; color: #c9d1d9; line-height: 1.5; }
.intent-actions { display: flex; gap: 8px; margin-top: 10px; }
.btn-run { padding: 6px 16px; background: #238636; border: 1px solid #2ea043; border-radius: 6px;
           color: #fff; font-size: 13px; cursor: pointer; transition: all 0.15s; }
.btn-run:hover { background: #2ea043; }
.btn-skip { padding: 6px 14px; background: transparent; border: 1px solid #30363d; border-radius: 6px;
            color: #8b949e; font-size: 13px; cursor: pointer; transition: all 0.15s; }
.btn-skip:hover { background: #21262d; }
.code-toggle { font-size: 11px; color: #388bfd; cursor: pointer; margin-left: auto;
               background: none; border: none; text-decoration: underline; }
.code-block { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
              padding: 10px; margin-top: 8px; font-size: 12px; font-family: monospace;
              color: #c9d1d9; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }

/* Result */
.result-block { margin-top: 8px; }
.result-label { font-size: 11px; color: #3fb950; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.06em; margin-bottom: 6px; }
.result-table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: monospace; }
.result-table th { background: #161b22; padding: 5px 8px; text-align: left; color: #7d8590;
                   border-bottom: 1px solid #30363d; }
.result-table td { padding: 4px 8px; border-bottom: 1px solid #21262d; color: #c9d1d9; }
.result-scalar { font-size: 20px; font-weight: 700; color: #3fb950; font-family: monospace; }
.error-block { background: #f8514915; border: 1px solid #f8514944; border-radius: 6px;
               padding: 10px 12px; margin-top: 8px; }
.error-label { font-size: 11px; color: #f85149; font-weight: 600; margin-bottom: 4px; }
.error-text { font-size: 12px; color: #ffa198; font-family: monospace; }

/* Register derived */
.register-form { background: #1c2128; border: 1px solid #3fb95044; border-radius: 8px;
                 padding: 12px 14px; margin-top: 8px; }
.register-label { font-size: 11px; color: #3fb950; font-weight: 600; margin-bottom: 8px; }
.register-inputs { display: flex; gap: 8px; flex-direction: column; }
.reg-input { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
             padding: 6px 10px; color: #e2e8f0; font-size: 13px; width: 100%; }
.reg-input:focus { outline: none; border-color: #3fb950; }

/* Input area */
.input-area { padding: 12px 20px; background: #161b22; border-top: 1px solid #30363d; }
.attachment-pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
                   background: #21262d; border: 1px solid #30363d; border-radius: 12px;
                   font-size: 12px; color: #8b949e; margin-bottom: 8px; }
.attachment-pill button { background: none; border: none; color: #8b949e; cursor: pointer;
                           font-size: 14px; line-height: 1; padding: 0; }
.input-row { display: flex; gap: 8px; align-items: flex-end; }
.attach-btn { padding: 8px 10px; background: #21262d; border: 1px solid #30363d;
              border-radius: 8px; color: #7d8590; cursor: pointer; font-size: 16px;
              flex-shrink: 0; transition: all 0.15s; }
.attach-btn:hover { background: #30363d; color: #c9d1d9; }
.msg-input { flex: 1; background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
             padding: 10px 14px; color: #e2e8f0; font-size: 14px; resize: none;
             min-height: 44px; max-height: 120px; }
.msg-input:focus { outline: none; border-color: #388bfd; }
.send-btn { padding: 10px 18px; background: #1f6feb; border: none; border-radius: 8px;
            color: #fff; font-size: 14px; cursor: pointer; flex-shrink: 0;
            font-weight: 600; transition: all 0.15s; }
.send-btn:hover { background: #388bfd; }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.loading-dot { display: inline-block; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:0.2} 50%{opacity:1} }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const { useState, useRef, useEffect, useCallback } = React;

const SESSION_ID = "session_" + Math.random().toString(36).slice(2, 9);

function App() {
  const [messages, setMessages] = useState([{
    role: "assistant",
    text: "Hello! I'm ready to analyse your data. Load the test fixtures using the button on the left, or upload your own CSV/XLSX/Parquet file.",
    id: "welcome"
  }]);
  const [artefacts, setArtefacts] = useState([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef();
  const messagesRef = useRef();

  const refreshManifest = async () => {
    const r = await fetch(`/api/manifest?session_id=${SESSION_ID}`);
    const d = await r.json();
    setArtefacts(d.artefacts);
  };

  useEffect(() => { refreshManifest(); }, []);
  useEffect(() => {
    if (messagesRef.current)
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages]);

  const loadTestData = async () => {
    const r = await fetch(`/api/load-test-data?session_id=${SESSION_ID}`, { method: "POST" });
    const d = await r.json();
    refreshManifest();
    setMessages(m => [...m, {
      role: "assistant", id: Date.now().toString(),
      text: d.message + "\\n\\nYou can now ask me to analyse these datasets. Try: \"What does the return distribution look like for the top 10 stocks?\" or \"Join the latest benchmark weights to the universe metadata.\""
    }]);
  };

  const clearAll = async () => {
    await fetch(`/api/registry?session_id=${SESSION_ID}`, { method: "DELETE" });
    setArtefacts([]);
    setMessages([{ role: "assistant", id: "cleared", text: "Registry and history cleared." }]);
  };

  const handleSend = async () => {
    if (!input.trim() && !file) return;
    const userMsg = { role: "user", text: input + (file ? ` [📎 ${file.name}]` : ""), id: Date.now().toString() };
    setMessages(m => [...m, userMsg]);
    setInput("");
    setLoading(true);

    const formData = new FormData();
    formData.append("message", input || "Please analyse this file.");
    formData.append("session_id", SESSION_ID);
    if (file) formData.append("file", new Blob([file.bytes]), file.name);
    setFile(null);

    try {
      const r = await fetch("/api/chat", { method: "POST", body: formData });
      const d = await r.json();
      refreshManifest();
      setMessages(m => [...m, {
        role: "assistant", id: Date.now().toString(),
        text: d.text,
        intent: d.intent,
        pendingId: d.pending_execution_id,
        astError: d.ast_error,
        hasCode: d.has_code,
      }]);
    } catch(e) {
      setMessages(m => [...m, { role: "assistant", id: Date.now().toString(), text: "Error: " + e.message }]);
    }
    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = ev => setFile({ name: f.name, bytes: ev.target.result });
    reader.readAsArrayBuffer(f);
  };

  return (
    <div className="app">
      <div className="manifest-panel">
        <div className="panel-header">Artefact Registry</div>
        <div className="panel-actions">
          <button className="btn-sm" onClick={loadTestData}>Load Test Data</button>
          <button className="btn-sm danger" onClick={clearAll}>Clear</button>
        </div>
        <div className="manifest-list">
          {artefacts.length === 0 && (
            <div style={{padding:"12px",color:"#7d8590",fontSize:"12px",textAlign:"center"}}>
              No artefacts registered.<br/>Click "Load Test Data" to start.
            </div>
          )}
          {artefacts.map(a => (
            <div key={a.name} className={`artefact-row ${a.provenance}`}>
              <div className="artefact-name">
                {a.provenance === "engine" ? "📦" : a.provenance === "user_upload" ? "📤" : "🔬"} {a.name}
              </div>
              <div className="artefact-meta">{a.dtype_summary}</div>
              <div className="artefact-desc">{a.description}</div>
              {a.parents?.length > 0 && (
                <div className="artefact-parents">↳ from: {a.parents.join(", ")}</div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="chat-area">
        <div className="chat-header">Conversation</div>
        <div className="messages" ref={messagesRef}>
          {messages.map(msg => (
            <Message key={msg.id} msg={msg} sessionId={SESSION_ID}
                     onExecuted={refreshManifest} onRegistered={refreshManifest} />
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-bubble">
                <span className="loading-dot">●</span>
                <span className="loading-dot" style={{animationDelay:"0.2s"}}>●</span>
                <span className="loading-dot" style={{animationDelay:"0.4s"}}>●</span>
              </div>
            </div>
          )}
        </div>
        <div className="input-area">
          {file && (
            <div>
              <span className="attachment-pill">
                📎 {file.name}
                <button onClick={() => setFile(null)}>×</button>
              </span>
            </div>
          )}
          <div className="input-row">
            <input type="file" ref={fileRef} onChange={handleFile}
                   accept=".csv,.xlsx,.xls,.parquet,.json" hidden />
            <button className="attach-btn" onClick={() => fileRef.current.click()}>📎</button>
            <textarea
              className="msg-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question or upload a file..."
              rows={1}
            />
            <button className="send-btn" onClick={handleSend} disabled={loading}>
              {loading ? "..." : "Send →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Message({ msg, sessionId, onExecuted, onRegistered }) {
  const [showCode, setShowCode] = useState(false);
  const [execResult, setExecResult] = useState(null);
  const [skipped, setSkipped] = useState(false);
  const [regName, setRegName] = useState("");
  const [regDesc, setRegDesc] = useState("");
  const [registered, setRegistered] = useState(false);

  const extractCleanText = (text) => {
    return text
      .replace(/INTENT:.*?\\n/s, "")
      .replace(/```python[\\s\\S]*?```/g, "")
      .trim();
  };

  const handleRun = async () => {
    const r = await fetch("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, execution_id: msg.pendingId }),
    });
    const d = await r.json();
    setExecResult(d);
    if (d.success) {
      onExecuted();
      if (d.output_type === "dataframe" || d.output_type === "series") {
        setRegName("derived_" + Date.now().toString().slice(-4));
        setRegDesc(msg.intent || "Derived analysis result");
      }
    }
  };

  const handleRegister = async () => {
    const r = await fetch("/api/register-derived", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        execution_id: msg.pendingId,
        name: regName,
        description: regDesc,
      }),
    });
    const d = await r.json();
    if (d.success) { setRegistered(true); onRegistered(); }
  };

  const cleanText = extractCleanText(msg.text);

  return (
    <div className={`message ${msg.role}`}>
      <div className="message-bubble">
        <div style={{whiteSpace:"pre-wrap"}}>{cleanText}</div>

        {msg.astError && (
          <div className="error-block">
            <div className="error-label">⛔ Code Blocked</div>
            <div className="error-text">{msg.astError}</div>
          </div>
        )}

        {msg.pendingId && !skipped && !execResult && (
          <div className="intent-card">
            <div className="intent-label">📊 Proposed Analysis</div>
            <div className="intent-text">{msg.intent || "Run analysis code"}</div>
            <div className="intent-actions">
              <button className="btn-run" onClick={handleRun}>▶ Run</button>
              <button className="btn-skip" onClick={() => setSkipped(true)}>Skip</button>
              <button className="code-toggle" onClick={() => setShowCode(s => !s)}>
                {showCode ? "Hide code" : "Show code"}
              </button>
            </div>
            {showCode && (
              <div className="code-block">
                {msg.text.match(/```python\\s*([\\s\\S]*?)```/)?.[1] || ""}
              </div>
            )}
          </div>
        )}

        {execResult && (
          <div className="result-block">
            {execResult.success ? (
              <>
                <div className="result-label">
                  ✓ Result ({execResult.output_type}) — {execResult.duration_ms?.toFixed(0)}ms
                </div>
                <ResultDisplay result={execResult} />
                {execResult.stdout && (
                  <div style={{marginTop:"6px",fontSize:"12px",color:"#7d8590",fontFamily:"monospace"}}>
                    stdout: {execResult.stdout}
                  </div>
                )}
                {(execResult.output_type === "dataframe" || execResult.output_type === "series") && !registered && (
                  <div className="register-form">
                    <div className="register-label">💾 Save as artefact?</div>
                    <div className="register-inputs">
                      <input className="reg-input" value={regName}
                             onChange={e => setRegName(e.target.value)} placeholder="Artefact name" />
                      <input className="reg-input" value={regDesc}
                             onChange={e => setRegDesc(e.target.value)} placeholder="Description" />
                      <button className="btn-run" style={{alignSelf:"flex-start"}} onClick={handleRegister}>
                        Register
                      </button>
                    </div>
                  </div>
                )}
                {registered && <div style={{color:"#3fb950",fontSize:"12px",marginTop:"6px"}}>✓ Registered</div>}
              </>
            ) : (
              <div className="error-block">
                <div className="error-label">⚠ Execution Error</div>
                <div className="error-text">{execResult.error.split("\\n").slice(-3).join("\\n")}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ResultDisplay({ result }) {
  if (result.output_type === "scalar") {
    return <div className="result-scalar">{result.output_preview}</div>;
  }
  if (result.output_type === "ndarray") {
    return (
      <div style={{fontFamily:"monospace",fontSize:"12px",color:"#c9d1d9"}}>
        shape: {result.output_preview.shape.join(" × ")}
        <pre>{JSON.stringify(result.output_preview.data, null, 2).slice(0, 300)}</pre>
      </div>
    );
  }
  const preview = result.output_preview;
  const columns = result.output_type === "dataframe"
    ? ["index", ...preview.columns]
    : ["index", preview.name || "value"];
  const rows = result.output_type === "dataframe"
    ? preview.index.map((idx, i) => [idx, ...preview.data[i]])
    : preview.index.map((idx, i) => [idx, preview.values[i]]);

  return (
    <div style={{overflowX:"auto",maxHeight:"200px",overflowY:"auto"}}>
      <table className="result-table">
        <thead>
          <tr>{columns.map(c => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{row.map((cell, j) => (
              <td key={j}>{typeof cell === "number" ? cell.toFixed(4) : String(cell)}</td>
            ))}</tr>
          ))}
        </tbody>
      </table>
      {(preview.shape?.[0] > 10 || preview.len > 10) && (
        <div style={{fontSize:"11px",color:"#7d8590",padding:"4px 8px"}}>
          Showing first 10 of {preview.shape?.[0] || preview.len} rows
        </div>
      )}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Artefact Registry Chat UI")
    print("Open: http://localhost:8000")
    print("API key:", "✓ set" if os.environ.get("ANTHROPIC_API_KEY") else "✗ missing (set ANTHROPIC_API_KEY)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
