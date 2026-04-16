"""
OllamaBaseAgent — base for all worker agents.

Fixes applied
-------------
1. Separate token budgets: worker_max_tokens (900) vs synth/manager tokens
2. Truncation detection: if response ends mid-sentence, marks low confidence
3. Larger num_ctx (4096) so the model doesn't lose context on long tasks
4. _flatten_sources / _flatten_list handle any dict shape local models return
5. CHUNK_TIMEOUT raised to 90s — large models can pause between tokens
6. RAG Enforcement: Forces workers to use retrieved context exclusively.

"""
from __future__ import annotations

import json
import queue
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from config import Config
from core.models import EventType, PipelineEvent, ResearchTask, TaskResult

def _extract_json(text: str) -> Optional[Dict]:
    text = text.strip()
    for candidate in [
        text,
        re.sub(r"^```(?:json)?\s*", "", re.sub(r"\s*```\s*$", "", text)).strip()]:
        try: return json.loads(candidate)
        except json.JSONDecodeError: pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try: return json.loads(text[s:e+1])
        except json.JSONDecodeError: pass
    return None

def _extract_float(text: str) -> Optional[float]:
    m = re.search(r"\b(0?\.\d+|1\.0|[01])\b", text)
    if m:
        try: return max(0.0, min(1.0, float(m.group())))
        except ValueError: pass
    return None

def _flatten_sources(raw: Any) -> List[str]:
    if not raw: return []
    if isinstance(raw, str): return [raw]
    result: List[str] = []
    for item in raw:
        if isinstance(item, str): result.append(item)
        elif isinstance(item, dict):
            v = next((str(v) for v in item.values() if v), str(item))
            result.append(v.strip())
        else: result.append(str(item))
    return result

def _flatten_list(raw: Any) -> List[str]:
    if not raw: return []
    if isinstance(raw, str): return [raw]
    out: List[str] = []
    for item in raw:
        if isinstance(item, str): out.append(item)
        elif isinstance(item, dict): out.append(next((str(v) for v in item.values() if v), str(item)))
        else: out.append(str(item))
    return out

def _is_truncated(text: str) -> bool:
    t = text.strip()
    if not t: return True
    
    # 🎯 FIX: Added '}' to the list of valid ending characters for JSON
    if not t[-1] in ".!?\"'`)]}": return True 
    
    if re.search(r"[a-zA-Z]{15,}$", t): return True
    return False

WORKER_SCHEMA = """\
Reply with ONLY this JSON. No prose. No code fences. No extra keys.
{
  "summary": "3-4 sentence summary of your findings",
  "key_points": ["specific finding 1", "specific finding 2", "specific finding 3", "specific finding 4"],
  "detailed_body": "write 3-4 paragraphs of substantive analysis here",
  "sources": ["relevant source or concept 1", "relevant source or concept 2"],
  "confidence": 0.8
}"""

class OllamaBaseAgent(ABC):
    CONNECT_TIMEOUT = 20
    CHUNK_TIMEOUT   = 90

    def __init__(self, config: Config, event_queue: queue.Queue) -> None:
        self._cfg   = config
        self._queue = event_queue
        self._url   = f"{config.ollama_base_url}/api/chat"

    @property
    @abstractmethod
    def role(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    def execute(self, task: ResearchTask, rag_context: str = "") -> TaskResult:
        self._emit(EventType.AGENT_START, task.task_id, f"{self.role} → {task.title}")
        
        # Uses Dynamic fallback if the agent was invented
        model = self._cfg.agent_models.get(self.role, self._cfg.agent_models.get("Dynamic", "mistral:7b"))
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt <= task.max_retries:
            attempt += 1
            try:
                self._emit(EventType.AGENT_THINKING, task.task_id, f"🤔 {self.role} thinking… (model: {model})")
                result = self._call_streaming(task, model, rag_context)
                self._emit(EventType.AGENT_COMPLETE, task.task_id, f"✅ {self.role} done | words={result.word_count} conf={result.confidence:.2f}")
                return result

            except requests.ConnectionError as exc:
                last_exc = exc
                wait = 2 ** attempt
                self._emit(EventType.AGENT_RETRY, task.task_id, f"⚠️ Ollama unreachable — retry in {wait}s")
                time.sleep(wait)
            except requests.Timeout as exc:
                last_exc = exc
                wait = 2 ** attempt
                self._emit(EventType.AGENT_RETRY, task.task_id, f"⚠️ Stream stalled — retry in {wait}s")
                time.sleep(wait)
            except Exception as exc:
                last_exc = exc
                self._emit(EventType.AGENT_RETRY, task.task_id, f"⚠️ Attempt {attempt} error: {exc}")
                time.sleep(1)

        self._emit(EventType.AGENT_FAILED, task.task_id, f"❌ {self.role} failed after {attempt} attempts")
        raise RuntimeError(f"{self.role}: {last_exc}")

    def _call_streaming(self, task: ResearchTask, model: str, rag_context: str) -> TaskResult:
        user_msg = self._build_message(task, rag_context)
        payload  = {
            "model":  model,
            "stream": True,
            "format": "json", # 🎯 STRICTLY forces valid JSON output to prevent text leakage
            "options": {
                "temperature": 0.3,
                "num_predict": self._cfg.worker_max_tokens,
                "num_ctx":     self._cfg.num_ctx,
            },
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_msg},
            ],
        }
        tokens: List[str] = []
        with requests.post(self._url, json=payload, stream=True,
                           timeout=(self.CONNECT_TIMEOUT, getattr(self._cfg, 'ollama_timeout', 120))) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line: continue
                try: chunk = json.loads(raw_line)
                except json.JSONDecodeError: continue
                t = chunk.get("message", {}).get("content", "")
                if t: tokens.append(t)
                if chunk.get("done"): break
        return self._parse("".join(tokens), task)

    def _build_message(self, task: ResearchTask, rag_context: str) -> str:
        rag_block = ""
        # 🎯 IMPROVED: Strict citation formatting rules
        if rag_context.strip():
            rag_block = (
                f"\n\n### Relevant context from knowledge base:\n{rag_context[:1500]}\n"
                f"\nCRITICAL INSTRUCTION: You MUST base your answer ONLY on the context provided above. "
                f"If the context does not contain the answer, explicitly state 'Insufficient data in knowledge base' "
                f"instead of guessing. "
                f"When using information from the context, use inline academic citations formatting the source name in brackets, e.g., [Source: filename.txt].\n"
            )
            
        domain_hint = f"\nDomain focus: {task.domain}\n" if task.domain else ""
        return f"Task: {task.title}{domain_hint}\n\n{task.description[:800]}{rag_block}\n\n{WORKER_SCHEMA}"

    def _parse(self, raw: str, task: ResearchTask) -> TaskResult:
        data = _extract_json(raw) or {}
        truncated = _is_truncated(raw)

        if not data:
            return TaskResult(
                task_id=task.task_id, agent_role=self.role, domain=task.domain, title=task.title,
                summary=raw[:400].strip(), key_points=[], detailed_body=raw.strip(),
                sources=[], confidence=0.3 if truncated else 0.45, word_count=len(raw.split()),
            )

        body       = str(data.get("detailed_body", ""))
        summary    = str(data.get("summary", ""))
        confidence = float(data.get("confidence", 0.6))
        if truncated: confidence = min(confidence, 0.5)

        return TaskResult(
            task_id=task.task_id, agent_role=self.role, domain=task.domain, title=task.title,
            summary=summary, key_points=_flatten_list(data.get("key_points", [])), detailed_body=body,
            sources=_flatten_sources(data.get("sources", [])), confidence=confidence,
            word_count=len((summary + " " + body).split()),
        )

    def _emit(self, et: EventType, tid: Optional[str] = None, msg: str = "") -> None:
        self._queue.put(PipelineEvent(event_type=et, agent=self.role, task_id=tid, message=msg))