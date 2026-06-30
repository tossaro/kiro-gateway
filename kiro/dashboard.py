# -*- coding: utf-8 -*-
"""
Usage Dashboard - Token usage logging and visualization for Kiro Gateway.
"""

import sqlite3
import time
import json
from pathlib import Path
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from loguru import logger

DB_PATH = Path(__file__).parent / "usage.db"

router = APIRouter()


def _init_db():
    """Create usage table if not exists."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            model TEXT,
            endpoint TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_creation_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            duration_ms REAL DEFAULT 0,
            status_code INTEGER DEFAULT 200
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp)
    """)
    # Migrate existing tables
    try:
        conn.execute("ALTER TABLE usage_log ADD COLUMN cache_read_tokens INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE usage_log ADD COLUMN cache_creation_tokens INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE usage_log ADD COLUMN cost_usd REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE usage_log ADD COLUMN session_id TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_log(session_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_names (
            session_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


_init_db()

# Pricing per 1M tokens (Anthropic rates with cache discounts)
PRICING = {
    "claude-opus-4": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_creation": 18.75},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_creation": 3.75},
    "claude-haiku-4": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_creation": 1.0},
}
DEFAULT_PRICE = {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_creation": 3.75}


def _resolve_pricing(model: str) -> dict:
    """Match model string to pricing tier."""
    if not model:
        return DEFAULT_PRICE
    m = model.lower()
    if "opus" in m:
        return PRICING["claude-opus-4"]
    if "haiku" in m:
        return PRICING["claude-haiku-4"]
    return DEFAULT_PRICE


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int,
                   cache_read_tokens: int = 0, cache_creation_tokens: int = 0) -> float:
    """Calculate real cost from token counts."""
    p = _resolve_pricing(model)
    # prompt_tokens = input tokens MINUS cache tokens (fresh input only)
    fresh_input = max(0, prompt_tokens - cache_read_tokens - cache_creation_tokens)
    cost = (
        (fresh_input / 1_000_000 * p["input"]) +
        (completion_tokens / 1_000_000 * p["output"]) +
        (cache_read_tokens / 1_000_000 * p["cache_read"]) +
        (cache_creation_tokens / 1_000_000 * p["cache_creation"])
    )
    return cost


def log_usage(model: str, endpoint: str, prompt_tokens: int,
              completion_tokens: int, total_tokens: int,
              duration_ms: float, status_code: int,
              cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
              session_id: str = None):
    """Log a request's token usage to SQLite."""
    cost = calculate_cost(model, prompt_tokens, completion_tokens,
                          cache_read_tokens, cache_creation_tokens)
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO usage_log (timestamp, model, endpoint, prompt_tokens, "
            "completion_tokens, total_tokens, cache_read_tokens, cache_creation_tokens, "
            "cost_usd, duration_ms, status_code, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), model, endpoint, prompt_tokens,
             completion_tokens, total_tokens, cache_read_tokens,
             cache_creation_tokens, cost, duration_ms, status_code, session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[Dashboard] Failed to log usage: {e}")


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware placeholder - actual logging happens in streaming_openai.py."""

    async def dispatch(self, request: Request, call_next):
        return await call_next(request)


# --- API Endpoints ---

@router.get("/api/usage")
async def get_usage(days: int = 7):
    """Get usage data for the dashboard."""
    since = time.time() - (days * 86400)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM usage_log WHERE timestamp > ? ORDER BY timestamp DESC",
        (since,)
    ).fetchall()
    conn.close()

    data = [dict(r) for r in rows]

    # Summary stats — use stored cost_usd when available, fallback to calculation
    total_prompt = sum(r["prompt_tokens"] for r in data)
    total_completion = sum(r["completion_tokens"] for r in data)
    total_cache_read = sum(r.get("cache_read_tokens", 0) for r in data)
    total_cache_creation = sum(r.get("cache_creation_tokens", 0) for r in data)
    total_requests = len(data)
    total_cost = sum(
        r.get("cost_usd", 0) or calculate_cost(r["model"], r["prompt_tokens"], r["completion_tokens"])
        for r in data
    )

    # Per-model breakdown
    models = {}
    for r in data:
        m = r["model"]
        if m not in models:
            models[m] = {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                         "cache_read_tokens": 0, "cache_creation_tokens": 0, "cost_usd": 0.0}
        models[m]["requests"] += 1
        models[m]["input_tokens"] += r["prompt_tokens"]
        models[m]["output_tokens"] += r["completion_tokens"]
        models[m]["cache_read_tokens"] += r.get("cache_read_tokens", 0)
        models[m]["cache_creation_tokens"] += r.get("cache_creation_tokens", 0)
        models[m]["cost_usd"] += r.get("cost_usd", 0) or calculate_cost(
            r["model"], r["prompt_tokens"], r["completion_tokens"])

    for m in models:
        models[m]["cost_usd"] = round(models[m]["cost_usd"], 4)

    # Hourly aggregation for chart
    hourly = {}
    for r in data:
        hour = int(r["timestamp"] // 3600) * 3600
        if hour not in hourly:
            hourly[hour] = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "cost_usd": 0.0}
        hourly[hour]["input_tokens"] += r["prompt_tokens"]
        hourly[hour]["output_tokens"] += r["completion_tokens"]
        hourly[hour]["requests"] += 1
        hourly[hour]["cost_usd"] += r.get("cost_usd", 0) or calculate_cost(
            r["model"], r["prompt_tokens"], r["completion_tokens"])

    for h in hourly:
        hourly[h]["cost_usd"] = round(hourly[h]["cost_usd"], 4)

    return JSONResponse({
        "summary": {
            "total_requests": total_requests,
            "input_tokens": total_prompt,
            "output_tokens": total_completion,
            "cache_read_tokens": total_cache_read,
            "cache_creation_tokens": total_cache_creation,
            "total_cost_usd": round(total_cost, 4),
        },
        "models": models,
        "hourly": [{"timestamp": k, **v} for k, v in sorted(hourly.items())],
        "recent": data[:50]
    })


@router.get("/api/usage/sessions")
async def get_sessions(days: int = 7):
    """Get usage grouped by session_id."""
    since = time.time() - (days * 86400)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT u.session_id,
               COUNT(*) as requests,
               SUM(u.prompt_tokens) as input_tokens,
               SUM(u.completion_tokens) as output_tokens,
               SUM(u.cost_usd) as cost_usd,
               MIN(u.timestamp) as first_seen,
               MAX(u.timestamp) as last_seen,
               GROUP_CONCAT(DISTINCT u.model) as models,
               sn.name as session_name
        FROM usage_log u
        LEFT JOIN session_names sn ON u.session_id = sn.session_id
        WHERE u.timestamp > ? AND u.session_id IS NOT NULL
        GROUP BY u.session_id
        ORDER BY last_seen DESC
    """, (since,)).fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@router.get("/api/usage/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get per-prompt detail for a session."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM usage_log
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,)).fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@router.post("/api/usage/sessions/{session_id}/name")
async def set_session_name(session_id: str, request: Request):
    """Set a custom name for a session. Empty name clears it."""
    body = await request.json()
    name = body.get("name", "").strip()
    conn = sqlite3.connect(str(DB_PATH))
    if not name:
        conn.execute("DELETE FROM session_names WHERE session_id = ?", (session_id,))
    else:
        conn.execute(
            "INSERT OR REPLACE INTO session_names (session_id, name, created_at) VALUES (?, ?, ?)",
            (session_id, name, time.time())
        )
    conn.commit()
    conn.close()
    return JSONResponse({"session_id": session_id, "name": name})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML page."""
    return DASHBOARD_HTML


# --- Dashboard HTML ---

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kiro Gateway - Usage Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0b; color: #e4e4e7; padding: 24px; }
h1 { font-size: 1.5rem; margin-bottom: 24px; color: #fff; }
h1 a { color: #7c3aed; text-decoration: none; }
h1 a:hover { text-decoration: underline; }
.session-badge { font-size: 0.85rem; background: #27272a; border: 1px solid #3f3f46; border-radius: 6px; padding: 4px 10px; margin-left: 12px; font-family: monospace; color: #a78bfa; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }
.stat-card { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 20px; }
.stat-card .label { font-size: 0.8rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-card .value { font-size: 1.8rem; font-weight: 600; color: #fff; margin-top: 4px; }
.stat-card .sub { font-size: 0.75rem; color: #52525b; margin-top: 2px; }
.stat-card.cost .value { color: #34d399; }
.stat-card.input .value { color: #7c3aed; }
.stat-card.output .value { color: #06b6d4; }
.chart-container { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 20px; margin-bottom: 24px; }
.chart-container h2 { font-size: 1rem; margin-bottom: 16px; color: #a1a1aa; }
.models { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 24px; }
.model-card { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 16px; }
.model-card .name { font-weight: 600; color: #a78bfa; margin-bottom: 8px; }
.model-card .row { display: flex; justify-content: space-between; font-size: 0.8rem; color: #71717a; margin-top: 4px; }
.model-card .row span:last-child { color: #e4e4e7; }
.model-card .row.cost span:last-child { color: #34d399; }
table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #27272a; }
th { color: #71717a; font-weight: 500; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px; }
td { color: #d4d4d8; }
td code { font-size: 0.75rem; color: #a78bfa; }
tr.clickable { cursor: pointer; }
tr.clickable:hover td { background: #27272a; }
td[style*="cursor:text"]:hover { background: #3f3f46 !important; border-radius: 4px; }
.controls { margin-bottom: 20px; display: flex; gap: 8px; }
.controls button { background: #27272a; border: 1px solid #3f3f46; color: #e4e4e7; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; }
.controls button.active { background: #7c3aed; border-color: #7c3aed; }
.controls button:hover { border-color: #7c3aed; }
</style>
</head>
<body>
<h1 id="title">⚡ Kiro Gateway Usage</h1>
<div class="controls">
  <button class="active" onclick="load(1)">24h</button>
  <button onclick="load(7)">7d</button>
  <button onclick="load(30)">30d</button>
</div>
<div class="stats" id="stats"></div>
<div class="chart-container"><h2>Tokens Over Time</h2><canvas id="chart" height="80"></canvas></div>
<div class="models" id="models"></div>
<div class="chart-container" id="sessions-section"><h2>Sessions</h2><div style="overflow-x:auto"><table id="sessions"><thead><tr><th>Name</th><th>Session ID</th><th>Requests</th><th>In</th><th>Out</th><th>Cost</th><th>Last Active</th></tr></thead><tbody></tbody></table></div></div>
<div class="chart-container"><h2>Recent Requests</h2><div style="overflow-x:auto"><table id="table"><thead><tr id="table-head"><th>Time</th><th>Model</th><th>Session</th><th>In</th><th>Out</th><th>Cost</th></tr></thead><tbody></tbody></table></div></div>
<script>
let chart;
const params=new URLSearchParams(location.search);
const filterSession=params.get('session_id');

if(filterSession){
  document.getElementById('title').innerHTML='<a href="/dashboard">⚡ Kiro Gateway</a><span class="session-badge">'+filterSession+'</span>';
  document.getElementById('sessions-section').style.display='none';
  document.getElementById('table-head').innerHTML='<th>Time</th><th>Model</th><th>In</th><th>Out</th><th>Cost</th>';
  // Fetch session name
  fetch('/api/usage/sessions?days=30').then(r=>r.json()).then(sessions=>{
    const s=sessions.find(x=>x.session_id===filterSession);
    if(s&&s.session_name){
      document.getElementById('title').innerHTML='<a href="/dashboard">⚡ Kiro Gateway</a><span class="session-badge">'+s.session_name+' ('+filterSession.substring(0,8)+'…)</span>';
    }
  });
}

function fmt(n){return n>=1000000?(n/1000000).toFixed(1)+'M':n>=1000?(n/1000).toFixed(1)+'K':n.toString()}

function load(days){
  document.querySelectorAll('.controls button').forEach(b=>b.classList.remove('active'));
  event?.target?.classList.add('active');

  if(filterSession){
    // Session-specific view
    fetch('/api/usage/sessions/'+filterSession).then(r=>r.json()).then(data=>{
      const total_in=data.reduce((a,r)=>a+r.prompt_tokens,0);
      const total_out=data.reduce((a,r)=>a+r.completion_tokens,0);
      const total_cost=data.reduce((a,r)=>a+(r.cost_usd||0),0);
      document.getElementById('stats').innerHTML=`
        <div class="stat-card"><div class="label">Requests</div><div class="value">${data.length}</div></div>
        <div class="stat-card input"><div class="label">Input Tokens</div><div class="value">${fmt(total_in)}</div></div>
        <div class="stat-card output"><div class="label">Output Tokens</div><div class="value">${fmt(total_out)}</div></div>
        <div class="stat-card cost"><div class="label">Total Cost</div><div class="value">$${total_cost.toFixed(4)}</div></div>
      `;
      // Chart per prompt
      const labels=data.map((_,i)=>'#'+(i+1));
      const inp=data.map(r=>r.prompt_tokens);
      const out=data.map(r=>r.completion_tokens);
      if(chart)chart.destroy();
      chart=new Chart(document.getElementById('chart'),{type:'bar',data:{labels,datasets:[
        {label:'Input',data:inp,backgroundColor:'#7c3aed'},
        {label:'Output',data:out,backgroundColor:'#06b6d4'}
      ]},options:{responsive:true,scales:{x:{stacked:true,ticks:{color:'#71717a'}},y:{stacked:true,ticks:{color:'#71717a',callback:v=>fmt(v)}}},plugins:{legend:{labels:{color:'#a1a1aa'}}}}});
      // Models
      const models={};
      for(const r of data){
        if(!models[r.model])models[r.model]={requests:0,input_tokens:0,output_tokens:0,cost_usd:0};
        models[r.model].requests++;
        models[r.model].input_tokens+=r.prompt_tokens;
        models[r.model].output_tokens+=r.completion_tokens;
        models[r.model].cost_usd+=r.cost_usd||0;
      }
      let mhtml='';
      for(const[name,m]of Object.entries(models)){
        mhtml+=`<div class="model-card"><div class="name">${name}</div>
          <div class="row"><span>Requests</span><span>${m.requests}</span></div>
          <div class="row"><span>Input</span><span>${fmt(m.input_tokens)}</span></div>
          <div class="row"><span>Output</span><span>${fmt(m.output_tokens)}</span></div>
          <div class="row cost"><span>Cost</span><span>$${m.cost_usd.toFixed(4)}</span></div></div>`;
      }
      document.getElementById('models').innerHTML=mhtml;
      // Recent (all prompts in session)
      let rows='';
      for(const r of data){
        const t=new Date(r.timestamp*1000).toLocaleString();
        rows+=`<tr><td>${t}</td><td>${r.model}</td><td>${fmt(r.prompt_tokens)}</td><td>${fmt(r.completion_tokens)}</td><td>$${(r.cost_usd||0).toFixed(4)}</td></tr>`;
      }
      document.querySelector('#table tbody').innerHTML=rows;
    });
    return;
  }

  // Global view
  fetch('/api/usage?days='+days).then(r=>r.json()).then(d=>{
    const s=d.summary;
    document.getElementById('stats').innerHTML=`
      <div class="stat-card"><div class="label">Requests</div><div class="value">${s.total_requests}</div></div>
      <div class="stat-card input"><div class="label">Input Tokens</div><div class="value">${fmt(s.input_tokens)}</div><div class="sub">prompt + system + tools</div></div>
      <div class="stat-card output"><div class="label">Output Tokens</div><div class="value">${fmt(s.output_tokens)}</div><div class="sub">completion</div></div>
      <div class="stat-card cost"><div class="label">Total Cost</div><div class="value">$${s.total_cost_usd.toFixed(4)}</div><div class="sub">calculated from token rates</div></div>
    `;
    const labels=d.hourly.map(h=>new Date(h.timestamp*1000).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}));
    const inp=d.hourly.map(h=>h.input_tokens);
    const out=d.hourly.map(h=>h.output_tokens);
    if(chart)chart.destroy();
    chart=new Chart(document.getElementById('chart'),{type:'bar',data:{labels,datasets:[
      {label:'Input',data:inp,backgroundColor:'#7c3aed'},
      {label:'Output',data:out,backgroundColor:'#06b6d4'}
    ]},options:{responsive:true,scales:{x:{stacked:true,ticks:{color:'#71717a'}},y:{stacked:true,ticks:{color:'#71717a',callback:v=>fmt(v)}}},plugins:{legend:{labels:{color:'#a1a1aa'}}}}});
    let mhtml='';
    for(const[name,m]of Object.entries(d.models)){
      mhtml+=`<div class="model-card"><div class="name">${name}</div>
        <div class="row"><span>Requests</span><span>${m.requests}</span></div>
        <div class="row"><span>Input</span><span>${fmt(m.input_tokens)}</span></div>
        <div class="row"><span>Output</span><span>${fmt(m.output_tokens)}</span></div>
        <div class="row cost"><span>Cost</span><span>$${m.cost_usd.toFixed(4)}</span></div></div>`;
    }
    document.getElementById('models').innerHTML=mhtml;
    let rows='';
    for(const r of d.recent){
      const t=new Date(r.timestamp*1000).toLocaleTimeString();
      const sid=r.session_id||'—';
      const cost=r.cost_usd?'$'+r.cost_usd.toFixed(4):'—';
      rows+=`<tr><td>${t}</td><td>${r.model}</td><td><code>${sid}</code></td><td>${fmt(r.prompt_tokens)}</td><td>${fmt(r.completion_tokens)}</td><td>${cost}</td></tr>`;
    }
    document.querySelector('#table tbody').innerHTML=rows;
  });
  // Sessions table
  fetch('/api/usage/sessions?days='+days).then(r=>r.json()).then(sessions=>{
    let shtml='';
    for(const s of sessions){
      const last=new Date(s.last_seen*1000).toLocaleString();
      const sid=s.session_id||'—';
      const name=s.session_name||'';
      shtml+=`<tr class="clickable"><td onclick="event.stopPropagation();renameSession('${sid}',this)" style="cursor:text" title="Click to rename"><strong>${name||'—'}</strong></td><td onclick="location.href='/dashboard?session_id=${sid}'"><code>${sid}</code></td><td onclick="location.href='/dashboard?session_id=${sid}'">${s.requests}</td><td onclick="location.href='/dashboard?session_id=${sid}'">${fmt(s.input_tokens)}</td><td onclick="location.href='/dashboard?session_id=${sid}'">${fmt(s.output_tokens)}</td><td onclick="location.href='/dashboard?session_id=${sid}'">$${(s.cost_usd||0).toFixed(4)}</td><td onclick="location.href='/dashboard?session_id=${sid}'">${last}</td></tr>`;
    }
    document.querySelector('#sessions tbody').innerHTML=shtml||'<tr><td colspan="7" style="color:#52525b">No session data yet</td></tr>';
  });
}
function renameSession(sid,td){
  const current=td.innerText==='—'?'':td.innerText;
  const name=prompt('Session name:',current);
  if(name===null)return;
  fetch('/api/usage/sessions/'+sid+'/name',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name||''})})
    .then(r=>{if(r.ok){td.innerHTML='<strong>'+(name||'—')+'</strong>';}else{alert('Failed to save');}});
}
load(1);
</script>
</body>
</html>"""
