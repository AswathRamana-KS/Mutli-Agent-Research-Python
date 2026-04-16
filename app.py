"""
Multi-Agent Research System — Streamlit Dashboard
Run:  streamlit run app.py
"""

from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EMBEDDING_MODELS,
    RECOMMENDED_MODELS,
    Config,
    DEFAULT_AGENT_MODELS,
    get_agent_color,
    get_agent_icon,
)
from core.models import EventType, ExecutionStats, FinalReport, PipelineEvent
from core.pipeline import ResearchPipeline


# ═══════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Multi-Agent Research System", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════
DARK_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', system-ui, sans-serif; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
input, textarea, select, .stTextInput input, .stTextArea textarea { background-color: #21262d !important; color: #c9d1d9 !important; border: 1px solid #30363d !important; border-radius: 6px !important; }
.stSelectbox div[data-baseweb="select"] { background-color: #21262d !important; border: 1px solid #30363d !important; }
.stButton > button { background: linear-gradient(135deg, #238636, #2ea043); color: white; border: none; border-radius: 6px; padding: 8px 20px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.stButton > button:hover { background: linear-gradient(135deg, #2ea043, #3fb950); transform: translateY(-1px); }
.btn-reset > button { background: linear-gradient(135deg, #6e40c9, #8957e5) !important; }
.metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 18px; text-align: center; min-height: 90px; display: flex; flex-direction: column; justify-content: center; }
.metric-value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.metric-label { font-size: 0.75rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.agent-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px 16px; text-align: center; transition: border-color 0.3s, box-shadow 0.3s; }
.agent-card.active { border-color: var(--agent-color); box-shadow: 0 0 12px color-mix(in srgb, var(--agent-color) 40%, transparent); }
.agent-card.done { border-color: #2ea043; }
.agent-card.failed { border-color: #da3633; }
.agent-icon { font-size: 1.6rem; margin-bottom: 4px; }
.agent-model { font-size: 0.65rem; color: #484f58; margin-top: 2px; }
.agent-status-dot { width: 8px; height: 8px; border-radius: 50%; background: #484f58; display: inline-block; margin-right: 4px; }
.dot-idle { background: #484f58; }
.dot-running { background: #f0e040; animation: pulse 0.8s infinite; }
.dot-done { background: #2ea043; }
.dot-failed { background: #da3633; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.log-container { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px; height: 320px; overflow-y: auto; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 0.78rem; }
.log-line { padding: 2px 0; border-bottom: 1px solid #161b22; line-height: 1.5; }
.log-time { color: #484f58; margin-right: 8px; }
.log-info { color: #58a6ff; }
.log-success { color: #3fb950; }
.log-warning { color: #d29922; }
.log-error { color: #f85149; }
.report-container { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px 28px; max-height: 600px; overflow-y: auto; }
.report-container h1, .report-container h2, .report-container h3 { color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
.report-container p { color: #c9d1d9; line-height: 1.7; }
.report-container li { color: #c9d1d9; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }
.badge-running { background: #161b22; color: #f0e040; border: 1px solid #f0e040; }
.badge-complete { background: #1f2d1f; color: #3fb950; border: 1px solid #3fb950; }
.badge-idle { background: #161b22; color: #8b949e; border: 1px solid #30363d; }
.badge-error { background: #2d1f1f; color: #f85149; border: 1px solid #f85149; }
.section-header { font-size: 0.7rem; color: #4B9EFF; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px; padding-bottom: 4px; border-bottom: 1px solid #21262d; }
hr { border-color: #21262d; }
[data-testid="stTabs"] button { color: #8b949e !important; background: transparent !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }
[data-testid="stFileUploader"] { background: #161b22; border: 1px dashed #30363d; border-radius: 8px; }
</style>
"""

def _init_state() -> None:
    defaults = {
        "running":      False,
        "thread":       None,
        "event_queue":  queue.Queue(),
        "log_lines":    [],          
        "stats":        ExecutionStats(),
        "report":       None,        
        "agent_states": {"Manager": "idle", "FactFinder": "idle", "Analyst": "idle", "Critic": "idle"},
        "active_tasks": {},          
        "run_count":    0,
        "session_id":   _short_id(),
        "error":        None,
        "config":       Config(),
        "last_event_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _short_id() -> str:
    import random, string
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=7))

def _reset() -> None:
    st.session_state.running      = False
    st.session_state.log_lines    = []
    st.session_state.stats        = ExecutionStats()
    st.session_state.report       = None
    st.session_state.agent_states = {"Manager": "idle", "FactFinder": "idle", "Analyst": "idle", "Critic": "idle"}
    st.session_state.error        = None
    st.session_state.event_queue  = queue.Queue()

# ═══════════════════════════════════════════════════════════════════════
# Background thread target
# ═══════════════════════════════════════════════════════════════════════
def _thread_target(query: str, cfg: Config, ev_queue: queue.Queue, local_files: List[str], scrape_urls: List[str], use_rag: bool, use_scraping: bool) -> None:
    try:
        pipeline = ResearchPipeline(config=cfg, event_queue=ev_queue)
        report, stats = pipeline.run(query=query, local_files=local_files, scrape_urls=scrape_urls, use_rag=use_rag, use_scraping=use_scraping)
        ev_queue.put(("__done__", report, stats))
    except Exception as exc:
        import traceback
        ev_queue.put(("__error__", str(exc), traceback.format_exc()))
    finally:
        import os
        if local_files:
            for fp in local_files:
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                except Exception: pass

# ═══════════════════════════════════════════════════════════════════════
# Event processor
# ═══════════════════════════════════════════════════════════════════════
_LOG_LEVEL_MAP: Dict[EventType, str] = {
    EventType.PIPELINE_START: "info", EventType.PIPELINE_COMPLETE: "success", EventType.PIPELINE_ERROR: "error",
    EventType.AGENT_START: "info", EventType.AGENT_THINKING: "info", EventType.AGENT_COMPLETE: "success",
    EventType.AGENT_RETRY: "warning", EventType.AGENT_FAILED: "error", EventType.QUALITY_FAIL: "warning",
    EventType.QUALITY_PASS: "success", EventType.CONTRADICTION: "warning", EventType.GAP_FOUND: "warning",
    EventType.SCRAPE_START: "info", EventType.SCRAPE_DONE: "success", EventType.RAG_QUERY: "info",
    EventType.SYNTHESIS_START: "info", EventType.SYNTHESIS_SECTION: "info", EventType.SYNTHESIS_COMPLETE: "success",
}

def _process_events() -> bool:
    eq = st.session_state.event_queue
    while not eq.empty():
        item = eq.get_nowait()
        if isinstance(item, tuple) and item[0] in ("__done__", "__error__"):
            st.session_state.running = False
            if item[0] == "__done__":
                _, report, stats = item
                st.session_state.report = report
                st.session_state.stats  = stats
                st.session_state.agent_states["Manager"] = "done"
            else:
                _, err_msg, tb = item
                st.session_state.error = err_msg
                st.session_state.log_lines.append((time.strftime("%H:%M:%S"), f"❌ ERROR: {err_msg}", "error"))
            continue

        if not isinstance(item, PipelineEvent): continue
        ev: PipelineEvent = item
        st.session_state.stats.total_events += 1

        if ev.event_type == EventType.AGENT_RETRY: st.session_state.stats.retries += 1
        elif ev.event_type == EventType.CONTRADICTION: st.session_state.stats.contradictions += 1
        elif ev.event_type == EventType.GAP_FOUND: st.session_state.stats.gaps_found += 1
        elif ev.event_type == EventType.TASK_COMPLETE: st.session_state.stats.tasks_done += 1
        elif ev.event_type == EventType.RAG_QUERY: st.session_state.stats.rag_queries += 1

        if ev.agent:
            if ev.event_type in (EventType.AGENT_START, EventType.AGENT_THINKING): st.session_state.agent_states[ev.agent] = "running"
            elif ev.event_type == EventType.AGENT_COMPLETE: st.session_state.agent_states[ev.agent] = "done"
            elif ev.event_type == EventType.AGENT_FAILED: st.session_state.agent_states[ev.agent] = "failed"
            elif ev.event_type == EventType.SYNTHESIS_START: st.session_state.agent_states["Manager"] = "running"
            elif ev.event_type == EventType.SYNTHESIS_COMPLETE: st.session_state.agent_states["Manager"] = "done"

        level = _LOG_LEVEL_MAP.get(ev.event_type, "info")
        st.session_state.log_lines.append((ev.fmt_time(), ev.message, level))

        try:
            with open("execution.log", "a", encoding="utf-8") as f:
                timestamp_str = ev.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                agent_str = ev.agent or "System"
                f.write(f"[{timestamp_str}] [{agent_str}] {ev.event_type.name}: {ev.message}\n")
        except Exception: pass

    return st.session_state.running

# ═══════════════════════════════════════════════════════════════════════
# UI renderers
# ═══════════════════════════════════════════════════════════════════════
def _render_agent_cards() -> None:
    st.markdown('<div class="section-header">Agent Status</div>', unsafe_allow_html=True)
    active_roles = ["Manager", "FactFinder", "Analyst", "Critic"]
    for r in st.session_state.agent_states:
        if r not in active_roles: active_roles.append(r)

    cols = st.columns(min(len(active_roles), 6))
    for col, role in zip(cols, active_roles[:6]):
        state  = st.session_state.agent_states.get(role, "idle")
        color  = get_agent_color(role)
        icon   = get_agent_icon(role)
        model = st.session_state.config.agent_models.get(role, st.session_state.config.agent_models.get("Dynamic", "—"))

        dot_cls  = {"idle": "dot-idle", "running": "dot-running", "done": "dot-done", "failed": "dot-failed"}.get(state, "dot-idle")
        card_cls = {"running": "active", "done": "done", "failed": "failed"}.get(state, "")
        bar_color = {"running": color, "done": "#2ea043", "failed": "#da3633", "idle": "#30363d"}[state]

        col.markdown(
            f"""<div class="agent-card {card_cls}" style="--agent-color:{color}">
                <div style="height:3px;background:{bar_color};border-radius:2px;margin-bottom:8px"></div>
                <div class="agent-icon">{icon}</div>
                <div style="font-size:0.78rem;font-weight:700;color:{color}">{role}</div>
                <div class="agent-model">{model.split(":")[0]}</div>
                <div style="margin-top:6px">
                  <span class="agent-status-dot {dot_cls}"></span>
                  <span style="font-size:0.65rem;color:#8b949e">{state.upper()}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    overflow = active_roles[6:]
    if overflow:
        badges = " ".join(
            f'<span style="background:#161b22;border:1px solid {get_agent_color(r)};color:{get_agent_color(r)};'
            f'padding:2px 8px;border-radius:12px;font-size:0.7rem;margin:2px">{get_agent_icon(r)} {r}</span>'
            for r in overflow
        )
        st.markdown(f'<div style="margin-top:6px">{badges}</div>', unsafe_allow_html=True)

def _render_stats() -> None:
    s = st.session_state.stats
    items = [("📋", s.total_events, "Events"), ("✅", f"{s.tasks_done}/{s.tasks_total}", "Tasks Done"), ("🔄", s.retries, "Retries"), ("⚡", s.contradictions, "Contradictions"), ("🔍", s.gaps_found, "Gaps Found")]
    cols = st.columns(5)
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"]
    for col, (icon, val, label), color in zip(cols, items, colors):
        col.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{color}">{val}</div><div class="metric-label">{icon} {label}</div></div>""", unsafe_allow_html=True)

def _render_log() -> None:
    lines = st.session_state.log_lines[-100:] 
    if not lines:
        st.markdown('<div class="log-container"><span style="color:#484f58">Waiting for research to start…</span></div>', unsafe_allow_html=True)
        return
    inner = ""
    for ts, msg, level in lines:
        cls = {"success": "log-success", "warning": "log-warning", "error": "log-error"}.get(level, "log-info")
        safe_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        inner += f'<div class="log-line"><span class="log-time">[{ts}]</span><span class="{cls}">{safe_msg}</span></div>'
    st.markdown(f'<div class="log-container">{inner}</div>', unsafe_allow_html=True)

def _render_report() -> None:
    report: Optional[FinalReport] = st.session_state.report
    if report is None:
        st.markdown('<div style="color:#484f58;text-align:center;padding:40px">📄 Report will appear here when research is complete</div>', unsafe_allow_html=True)
        return
    md = report.to_markdown()
    st.download_button(label="⬇ Download .md", data=md, file_name=f"research_report_{report.run_id}.md", mime="text/markdown", key="dl_report")
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(md)
    st.markdown('</div>', unsafe_allow_html=True)

def _render_overview() -> None:
    report = st.session_state.report
    if report is None:
        st.info("Overview available after research completes.")
        return
    st.markdown("### 🤖 Agent Contributions")
    for agent, contrib in report.agent_contributions.items():
        color = get_agent_color(agent)
        st.markdown(f'<div style="background:#161b22;border-left:3px solid {color};padding:10px 14px;margin:6px 0;border-radius:0 6px 6px 0"><strong style="color:{color}">{agent}</strong>: {contrib}</div>', unsafe_allow_html=True)
    if report.key_findings:
        st.markdown("### 🔑 Key Findings")
        for i, f in enumerate(report.key_findings, 1):
            st.markdown(f'<div style="background:#161b22;padding:8px 14px;margin:4px 0;border-radius:6px;border:1px solid #30363d"><span style="color:#58a6ff;font-weight:700">{i}.</span> {f}</div>', unsafe_allow_html=True)
    if report.sections:
        st.markdown(f"### 📊 Domain Sections ({len(report.sections)} covered)")
        cols = st.columns(3)
        for i, (domain, content) in enumerate(report.sections.items()):
            word_count = len(content.split())
            cols[i % 3].markdown(f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin:4px 0;text-align:center"><div style="color:#58a6ff;font-weight:700;font-size:0.8rem">{domain}</div><div style="color:#484f58;font-size:0.7rem">{word_count} words</div></div>', unsafe_allow_html=True)

def _status_badge() -> str:
    if st.session_state.error: return '<span class="badge badge-error">⚠ Error</span>'
    if st.session_state.running: return '<span class="badge badge-running">⬤ Running</span>'
    if st.session_state.report: return '<span class="badge badge-complete">✓ Research Complete</span>'
    return '<span class="badge badge-idle">○ Idle</span>'

# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════
def _render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        st.markdown("### 🦙 Ollama")
        cfg = st.session_state.config
        cfg.ollama_base_url = st.text_input("Ollama URL", value=cfg.ollama_base_url, key="s_ollama_url")
        cfg.ollama_timeout = st.slider("Timeout (s)", 30, 300, getattr(cfg, "ollama_timeout", 120), 10, key="s_timeout")
        cfg.max_tokens = st.slider("Max tokens per call", min_value=128, max_value=1024, value=getattr(cfg, "worker_max_tokens", 900), step=64, help="256–512 recommended for CPU. Higher = better quality but slower.", key="s_tokens")

        st.markdown("### 🤖 Agent → Model")
        model_list = list(RECOMMENDED_MODELS.keys())
        all_assignable = ["Manager", "FactFinder", "Analyst", "Critic", "Dynamic"]
        for role in all_assignable:
            icon = get_agent_icon(role) if role != "Dynamic" else "✨"
            label = f"{icon} {role}" + (" (New Agents)" if role == "Dynamic" else "")
            current_model = cfg.agent_models.get(role, "llama3:8b")
            default_idx = model_list.index(current_model) if current_model in model_list else 0
            chosen = st.selectbox(label, options=model_list, index=default_idx, format_func=lambda m: f"{m}  —  {RECOMMENDED_MODELS[m].split('·')[0].strip()}", key=f"s_model_{role}")
            cfg.agent_models[role] = chosen

        st.markdown("### 🎯 Quality Gate")
        cfg.quality_threshold = st.slider("Min quality score", 0.0, 1.0, cfg.quality_threshold, 0.05, key="s_quality")
        cfg.max_retries = st.slider("Max retries per task", 0, 4, cfg.max_retries, 1, key="s_retries")

        st.markdown("---")
        st.markdown("### 📚 Local RAG")
        use_rag = st.checkbox("Enable Local RAG", key="s_use_rag")
        local_file_paths: List[str] = []

        if use_rag:
            embed_backends = ["ollama", "sentence-transformers"]
            cfg.embed_backend = st.selectbox("Embedding backend", embed_backends, index=embed_backends.index(cfg.embed_backend), key="s_embed_backend")
            embed_model_list = list(EMBEDDING_MODELS.keys())
            default_em = embed_model_list.index(cfg.embed_model) if cfg.embed_model in embed_model_list else 0
            cfg.embed_model = st.selectbox("Embedding model", embed_model_list, index=default_em, format_func=lambda m: f"{m}", key="s_embed_model")
            cfg.top_k_rag = st.slider("Top-K chunks per query", 1, 20, cfg.top_k_rag, 1, key="s_topk")

            uploaded = st.file_uploader("Upload documents (PDF / TXT / DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True, key="s_files")
            if uploaded:
                for uf in uploaded:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        local_file_paths.append(tmp.name)
                st.success(f"📂 {len(local_file_paths)} file(s) ready to ingest")

        st.markdown("---")
        st.markdown("### 🌐 Web Scraping")
        use_scraping = st.checkbox("Enable Web Scraping", key="s_use_scraping")
        scrape_urls: List[str] = []

        if use_scraping:
            cfg.firecrawl_api_key = st.text_input("Firecrawl API Key (optional)", value=cfg.firecrawl_api_key or "", type="password", help="Leave blank to use free BeautifulSoup scraping", key="s_fc_key") or None
            if cfg.firecrawl_api_key: st.success("🔑 Firecrawl active")
            else: st.info("ℹ️ Using free BS4 scraper")
            urls_raw = st.text_area("URLs to scrape (one per line)", height=100, key="s_urls")
            scrape_urls = [u.strip() for u in urls_raw.splitlines() if u.strip()]
            if scrape_urls: st.caption(f"✅ {len(scrape_urls)} URL(s) queued")

        if use_rag or use_scraping:
            st.markdown("---")
            cfg.chroma_path = st.text_input("ChromaDB path", value=cfg.chroma_path, key="s_chroma")
            if st.button("🗑 Clear Knowledge Base", key="s_clear_kb"):
                try:
                    from rag.rag import VectorStore
                    VectorStore(path=cfg.chroma_path).clear()
                    st.success("Knowledge base cleared.")
                except Exception as e: st.error(f"Error: {e}")

        st.markdown("---")
        st.caption("💡 **Recommended models for i7-1265U / 32GB:**\n- Orchestrator: `llama3:8b`\n- Workers: `mistral:7b`\n- Lightweight: `phi3:mini`\n\nPull models: `ollama pull llama3:8b`")
    return use_rag, use_scraping, local_file_paths, scrape_urls

# ═══════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    _init_state()
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    c1, c2 = st.columns([5, 2])
    with c1: st.markdown("# 🧠 Multi-Agent Research System\n<small style='color:#8b949e'>Powered by Ollama · Dynamic Experts · Full orchestration transparency</small>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div style='text-align:right;padding-top:12px'><span style='background:#1f2d1f;color:#3fb950;border:1px solid #2ea043;padding:3px 10px;border-radius:20px;font-size:0.7rem;font-weight:600'>● Ollama Local</span>&nbsp;<span style='color:#484f58;font-size:0.7rem'>Session: {st.session_state.session_id}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    use_rag, use_scraping, local_file_paths, scrape_urls = _render_sidebar()

    st.markdown('<div class="section-header">Research Topic</div>', unsafe_allow_html=True)
    q_col, b1, b2 = st.columns([5, 1, 1])

    with q_col: query = st.text_input("Enter research query", label_visibility="collapsed", placeholder="e.g. Behavioral Psychology in Mobile Gaming", disabled=st.session_state.running, key="s_query")
    with b1: run_clicked = st.button("▶ Run Research", disabled=st.session_state.running or not query, use_container_width=True, key="btn_run")
    with b2:
        st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
        reset_clicked = st.button("✕ Reset", disabled=st.session_state.running, use_container_width=True, key="btn_reset")
        st.markdown('</div>', unsafe_allow_html=True)

    if run_clicked and query and not st.session_state.running:
        _reset()
        st.session_state.running   = True
        st.session_state.run_count += 1
        st.session_state.stats.tasks_total = 0
        cfg = st.session_state.config
        t = threading.Thread(target=_thread_target, args=(query, cfg, st.session_state.event_queue, local_file_paths, scrape_urls, use_rag, use_scraping), daemon=True, name=f"research-{st.session_state.run_count}")
        t.start()
        st.session_state.thread = t

    if reset_clicked:
        _reset()
        st.rerun()

    st.markdown("")
    _render_agent_cards()

    st.markdown("")
    _render_stats()

    st.markdown("---")

    s1, s2 = st.columns([4, 2])
    with s1:
        st.markdown(_status_badge(), unsafe_allow_html=True)
        evt_count = len(st.session_state.log_lines)
        if evt_count: st.markdown(f"<small style='color:#484f58'>Run #{st.session_state.run_count} · {evt_count} events logged</small>", unsafe_allow_html=True)
    with s2:
        if st.session_state.stats.duration_seconds > 0: st.markdown(f"<div style='text-align:right;color:#484f58;font-size:0.8rem'>⏱ {st.session_state.stats.duration_seconds}s</div>", unsafe_allow_html=True)

    tab_log, tab_overview, tab_report = st.tabs(["📡 Live Log", "📊 Overview", "📄 Report"])

    with tab_log: _render_log()
    with tab_overview: _render_overview()
    with tab_report: _render_report()

    if st.session_state.error:
        st.error(f"**Pipeline Error:** {st.session_state.error}")
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()

    if st.session_state.running:
        _process_events()
        time.sleep(0.4)
        st.rerun()
    else:
        if _process_events(): st.rerun()

if __name__ == "__main__":
    main()