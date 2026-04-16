"""
ManagerAgent — the Orchestrator.

True multi-agent orchestration:
  1. ASSESS  — classify query complexity (simple/moderate/complex/expert)
  2. PLAN    — extract domains from the query; create 1 task per domain
  3. ROUTE   — assign each domain task to the BEST-FIT specialized agent
  4. EXECUTE — workers run sequentially; Manager quality-gates each
  5. REVIEW  — detect & RESOLVE contradictions, fill gaps by RE-QUERYING
  6. SYNTHESISE — write each domain section independently, then add
                  executive summary + conclusion
ManagerAgent — the Orchestrator with Dynamic Agent Generation.
"""
from __future__ import annotations

import json
import queue
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import Config
from agents.base import _extract_json, _extract_float, _is_truncated
from agents.workers import create_worker
from core.models import (
    EventType, ExecutionStats, FinalReport,
    PipelineEvent, ResearchTask, TaskResult, TaskStatus,
)

COMPLEXITY_TASKS = {"simple": 3, "moderate": 4, "complex": 6, "expert": 8}

_DECOMPOSE_PROMPT = """You are the Manager Agent orchestrating a research system.
Break this query into {n} research sub-tasks. Each task must cover a DIFFERENT domain.

CRITICAL INSTRUCTION:
You MUST INVENT a specific specialist role for each task. 
(Examples of specialists: "UrbanPlanner", "MarineBiologist", "Astro-physicist" - DO NOT copy these examples, invent your own based on the specific query!)

For each, write a 2-3 sentence `system_prompt` instructing them how to analyze the data.
Tell them to focus on qualitative analysis and real mechanics. Explicitly tell them NOT to invent statistics.

Reply with ONLY a JSON array, no fences. Match this exact format:
[
  {{
    "title": "Gameplay Loop Analysis", 
    "description": "Analyze the progression mechanics.", 
    "assigned_to": "GameDesigner", 
    "system_prompt": "You are a Game Designer. Analyze reward loops. Do not invent fake statistics.", 
    "domain": "Game Mechanics"
  }}
]"""

_QUALITY_PROMPT = """You are a strict QA grading bot. 
Score this research output 0.0 to 1.0.
- > 0.6: contains substantive analysis, accurate concepts, and logical reasoning. 3+ sentences.
- < 0.4: vague, very short, or relies on obviously fabricated statistics/fake studies.

You must reply with ONLY the raw float value. Do not include ANY other text. 
Example response: 0.85"""

_EXEC_SUMMARY_PROMPT = """Write an Executive Summary for this research report.
Focus on the core themes, real mechanics, and most important insights. 
CRITICAL: Do NOT invent, hallucinate, or fabricate statistics, percentages, or fake studies. 
Write 2-3 paragraphs. Be direct and substantive."""

_SECTION_PROMPT = """You are a research editor synthesising findings for a specific domain section.
Domain: {domain}
Agents' findings:
{content}

Write a well-structured section for the research report on this domain.
- Start with the most important insight. Focus on actual systems, mechanics, and qualitative analysis.
- CRITICAL: DO NOT invent, hallucinate, or fabricate statistics, percentages, or fake studies (e.g., do not say "85% of players").
- Write 2-3 substantive paragraphs.
- Do NOT start with "In this section" or generic phrases.
Write the section now:"""

_CONCLUSION_PROMPT = """Write a Conclusion for this research report on: {query}
Key findings from all domains:
{findings_list}

Write a 2-paragraph conclusion that synthesises the most critical cross-domain insights and ends with actionable next steps. 
Do not invent fake data."""


class ManagerAgent:
    def __init__(self, config: Config, event_queue: queue.Queue) -> None:
        self._cfg   = config
        self._queue = event_queue
        self._model = config.agent_models.get("Manager", "llama3:8b")
        self._url   = f"{config.ollama_base_url}/api/chat"
        self.stats  = ExecutionStats()

    def run(self, query: str, retriever=None) -> Tuple[FinalReport, ExecutionStats]:
        t0 = time.monotonic()
        self._emit(EventType.PIPELINE_START, message=f"🧠 Manager starting pipeline: '{query[:80]}'")

        n_tasks, complexity = self._assess_complexity(query)
        tasks = self._plan_tasks(query, n_tasks, complexity)
        self.stats.tasks_total = len(tasks)

        results: List[TaskResult] = []
        for i, task in enumerate(tasks, 1):
            self._emit(EventType.TASK_ASSIGNED, task.task_id, f"📤 [{i}/{len(tasks)}] → {task.assigned_to}: {task.title}")
            result = self._execute_one(task, retriever)
            if result:
                results.append(result)
                self.stats.tasks_done += 1

        if not results: raise RuntimeError("All worker tasks failed.")

        # 🎯 Active Resolution and Gap Filling for Rubric
        self._detect_and_resolve_contradictions(results)
        self._detect_and_fill_gaps(tasks, results, retriever)

        self._emit(EventType.SYNTHESIS_START, message=f"📝 Manager synthesising {len(results)} domain results…")
        report = self._synthesise(query, results, tasks)

        self.stats.duration_seconds = round(time.monotonic() - t0, 1)
        self._emit(EventType.PIPELINE_COMPLETE,
                   message=f"✅ Done in {self.stats.duration_seconds}s | {len(results)}/{len(tasks)} tasks | {len(report.sections)} sections")
        return report, self.stats

    def _assess_complexity(self, query: str) -> Tuple[int, str]:
        # Add the strict guardrails back to the prompt!
        prompt = (
            "Classify this research query as: simple, moderate, complex, or expert.\n"
            "simple = straightforward factual topic\n"
            "moderate = 2-3 distinct angles\n"
            "complex = 4-6 distinct domains\n"
            "expert = 7+ highly technical domains\n\n"
            "Reply with ONLY one word."
        )
        try:
            raw  = self._ollama_call(prompt, f"Query: {query}", max_tokens=5)
            word = raw.strip().lower().split()[0] if raw.strip() else "moderate"
            if word not in COMPLEXITY_TASKS: word = "moderate"
        except Exception:
            word = "moderate"
            
        n = COMPLEXITY_TASKS[word]
        self._emit(EventType.LOG, message=f"🗂 Query complexity: [{word.upper()}] → {n} tasks")
        return n, word

    def _plan_tasks(self, query: str, n_tasks: int, complexity: str) -> List[ResearchTask]:
        self._emit(EventType.LOG, message=f"🗂 Planning {n_tasks} dynamically generated tasks…")
        raw_list: List[Dict] = []
        for attempt in range(2):
            try:
                prompt = _DECOMPOSE_PROMPT.format(n=n_tasks)
                raw    = self._ollama_call(prompt, f"Research query:\n{query}")
                raw_list = _parse_task_list(raw)
                if len(raw_list) >= 3: break
            except Exception as exc:
                self._emit(EventType.LOG, message=f"⚠️ Decompose error: {exc}")

        if len(raw_list) < 3:
            raw_list = [
                {"title": "Background Facts", "description": f"Gather key facts for: {query[:200]}", "assigned_to": "FactFinder", "domain": "Background"},
                {"title": "Strategic Analysis", "description": f"Analyse trends for: {query[:200]}", "assigned_to": "Analyst", "domain": "Analysis"},
                {"title": "Risks & Issues", "description": f"Identify risks for: {query[:200]}", "assigned_to": "Critic", "domain": "Risks"}
            ]

        tasks: List[ResearchTask] = []
        for item in raw_list[:n_tasks]:
            task = ResearchTask(
                title       = str(item.get("title", "Research Task")),
                description = str(item.get("description", query)),
                assigned_to = str(item.get("assigned_to", "FactFinder")),
                system_prompt = str(item.get("system_prompt", "")),
                domain      = str(item.get("domain", item.get("title", "General"))),
                max_retries = self._cfg.max_retries,
            )
            tasks.append(task)
            tag = "⚡ Dynamic" if task.system_prompt else "Core"
            self._emit(EventType.TASK_CREATED, task.task_id, f"📋 [{tag}] {task.assigned_to}: {task.title}")

        return tasks

    def _execute_one(self, task: ResearchTask, retriever=None) -> Optional[TaskResult]:
        task.status = TaskStatus.IN_PROGRESS
        role_name = task.assigned_to
        worker = create_worker(role_name, self._cfg, self._queue, custom_prompt=task.system_prompt)
        best:  Optional[TaskResult] = None

        for attempt in range(task.max_retries + 1):
            rag_ctx = ""
            if retriever:
                self._emit(EventType.RAG_QUERY, task.task_id, "🔍 Querying knowledge base…")
                rag_ctx = retriever.retrieve(task.description)
                self.stats.rag_queries += 1

            try:
                result = worker.execute(task, rag_ctx)
            except Exception as exc:
                self._emit(EventType.AGENT_FAILED, task.task_id, f"❌ Worker error attempt {attempt+1}: {exc}")
                time.sleep(1)
                continue

            score = self._evaluate_quality(task, result)
            result.quality_score = score

            if best is None or score > (best.quality_score or 0): best = result
            if score >= self._cfg.quality_threshold:
                self._emit(EventType.QUALITY_PASS, task.task_id, f"✅ Accepted (q={score:.2f}, words={result.word_count})")
                task.status = TaskStatus.COMPLETE
                return result

            self.stats.retries += 1
            self._emit(EventType.QUALITY_FAIL, task.task_id, f"⚠️ Low quality (q={score:.2f}) — attempt {attempt+1}/{task.max_retries+1}")

            if attempt < task.max_retries:
                new_role = self._rotate_role(role_name)
                self._emit(EventType.AGENT_RETRY, task.task_id, f"🔄 Reassigning: {role_name} → {new_role}")
                role_name = new_role
                worker = create_worker(new_role, self._cfg, self._queue)

        if best:
            task.status = TaskStatus.COMPLETE
            self._emit(EventType.QUALITY_FAIL, task.task_id, f"⚠️ Accepting best result (q={best.quality_score:.2f})")
            return best

        task.status = TaskStatus.FAILED
        return None

    def _evaluate_quality(self, task: ResearchTask, result: TaskResult) -> float:
        total = len(result.summary.split()) + len(result.detailed_body.split())
        if total < 30: return 0.1
        if total < 60: return 0.35
        if not result.key_points: return 0.45
        self._emit(EventType.QUALITY_CHECK, task.task_id, "🔎 Evaluating quality…")
        snippet = f"Task: {task.title}\nSummary: {result.summary[:250]}\nPoints: {'; '.join(result.key_points[:3])}"
        try:
            raw = self._ollama_call(_QUALITY_PROMPT, snippet, max_tokens=5)
            s   = _extract_float(raw)
            return s if s is not None else 0.65
        except Exception: return 0.65

    def _detect_and_resolve_contradictions(self, results: List[TaskResult]) -> None:
        neg = {"not ", "never ", "no evidence", "disproven", "incorrect", "false"}
        for i, r1 in enumerate(results):
            b1 = (r1.summary + " " + r1.detailed_body).lower()
            for r2 in results[i+1:]:
                b2 = (r2.summary + " " + r2.detailed_body).lower()
                common_words = set(r1.title.lower().split()) & set(r2.title.lower().split())
                common = {w for w in common_words if len(w) > 4}
                if common:
                    for w in neg:
                        if (w in b1 and w not in b2) or (w in b2 and w not in b1):
                            self.stats.contradictions += 1
                            topic = ", ".join(list(common)[:2])
                            self._emit(EventType.CONTRADICTION,
                                       message=f"⚡ Contradiction on '{topic}': {r1.agent_role} vs {r2.agent_role}. Resolving...")
                            
                            prompt = (
                                f"Two research agents provided conflicting information on the topic '{topic}'.\n\n"
                                f"Agent 1 ({r1.agent_role}): {r1.summary}\n\n"
                                f"Agent 2 ({r2.agent_role}): {r2.summary}\n\n"
                                f"Analyze both claims and write a definitive, objective 2-paragraph resolution that finds the truth."
                            )
                            resolution = self._ollama_call(system="You are an expert consensus builder and fact-checker.", user=prompt, max_tokens=self._cfg.synth_max_tokens)
                            
                            if len(resolution.split()) > 10:
                                r1.conflict_resolution = f"**[Manager's Conflict Resolution vs {r2.agent_role}]:**\n{resolution}"
                                self._emit(EventType.LOG, message=f"✅ Conflict resolved for '{topic}'.")
                            break

    def _detect_and_fill_gaps(self, tasks: List[ResearchTask], results: List[TaskResult], retriever) -> None:
        done_ids = {r.task_id for r in results}
        for t in tasks:
            if t.task_id not in done_ids:
                self.stats.gaps_found += 1
                self._emit(EventType.GAP_FOUND, message=f"🔍 Gap: '{t.title}' failed. Re-querying FactFinder to fill gap...")
                
                worker = create_worker("FactFinder", self._cfg, self._queue)
                try:
                    t.max_retries = 0 
                    t.assigned_to = "FactFinder"
                    rag_ctx = retriever.retrieve(t.description) if retriever else ""
                    res = worker.execute(t, rag_ctx)
                    results.append(res)
                    self._emit(EventType.LOG, message=f"✅ Gap filled successfully for '{t.title}'.")
                except Exception as e:
                    self._emit(EventType.LOG, message=f"❌ Could not fill gap for '{t.title}': {e}")

    def _synthesise(self, query: str, results: List[TaskResult], tasks: List[ResearchTask]) -> FinalReport:
        sections: Dict[str, str] = {}
        key_findings: List[str]  = []
        contributions: Dict[str, str] = {}

        domain_results: Dict[str, List[TaskResult]] = {}
        for r in results:
            key = r.domain or r.title or r.agent_role
            if key not in domain_results:
                domain_results[key] = []
            domain_results[key].append(r)

        for domain, r_list in domain_results.items():
            agent_names = ", ".join(sorted(set(r.agent_role for r in r_list)))
            self._emit(EventType.SYNTHESIS_SECTION, message=f"📝 Synthesising section: {domain} [{agent_names}]")

            combined_content = ""
            for r in r_list:
                resolution_text = f"\n\n{getattr(r, 'conflict_resolution', '')}" if getattr(r, 'conflict_resolution', None) else ""
                combined_content += f"--- Findings from {r.agent_role} ---\nSummary: {r.summary}\n"
                if r.key_points: combined_content += "Key points:\n" + "\n".join(f"- {p}" for p in r.key_points) + "\n"
                combined_content += f"Detailed findings:\n{r.detailed_body}{resolution_text}\n\n"

            prompt = _SECTION_PROMPT.format(domain=domain, agent=agent_names, content=combined_content[:2000])
            section_text = self._synth_section(user=prompt, fallback=self._fallback_section(r_list[0]))
            sections[domain] = section_text

            for r in r_list:
                for pt in r.key_points[:2]:
                    if pt and len(pt.split()) > 4: key_findings.append(pt)
                
                clean_summary = r.summary.replace("```json", "").replace("```", "").replace("{", "").strip()
                if len(clean_summary) > 200:
                    clean_summary = clean_summary[:197].rsplit(' ', 1)[0] + "..."
                contributions[r.agent_role] = clean_summary if clean_summary else r.title

        seen = set()
        unique_findings = []
        for f in key_findings:
            if f not in seen:
                seen.add(f)
                unique_findings.append(f)
        key_findings = unique_findings[:10]

        self._emit(EventType.LOG, message="📝 Writing executive summary…")
        all_summaries = "\n\n".join(f"[{r.agent_role} — {r.domain}]: {r.summary}" for r in results[:6])
        exec_summary = self._synth_section(
            user=f"Query: {query}\n\nResearch summaries by domain:\n{all_summaries[:1600]}",
            system_override=_EXEC_SUMMARY_PROMPT, fallback=self._fallback_exec_summary(query, results)
        )

        self._emit(EventType.LOG, message="📝 Writing conclusion…")
        findings_list = "\n".join(f"- {f}" for f in key_findings[:8])
        conclusion = self._synth_section(
            user=_CONCLUSION_PROMPT.format(query=query, findings_list=findings_list),
            fallback=self._fallback_conclusion(query, results)
        )

        self._emit(EventType.SYNTHESIS_COMPLETE, message=f"📄 Report complete: {len(sections)} sections, {len(key_findings)} key findings")
        self.stats.total_events += 1

        return FinalReport(
            query=query, sections=sections, key_findings=key_findings,
            agent_contributions=contributions, executive_summary=exec_summary, conclusion=conclusion
        )

    _DEFAULT_SYNTH_SYSTEM = (
        "You are a senior research editor. Write clear, substantive prose based on the research provided. "
        "Be specific: use numbers, examples, and direct recommendations. Do not use filler phrases. Write at least 2 solid paragraphs."
    )

    def _synth_section(self, user: str, fallback: str, system_override: str = "") -> str:
        system = system_override or self._DEFAULT_SYNTH_SYSTEM
        try:
            text = self._ollama_call(system=system, user=user, max_tokens=self._cfg.synth_max_tokens).strip()
            if len(text.split()) >= 20: return text
        except Exception as exc:
            self._emit(EventType.LOG, message=f"⚠️ Synthesis call failed: {exc}")
        return fallback

    @staticmethod
    def _fallback_section(r: TaskResult) -> str:
        parts = [r.summary] if r.summary else []
        if r.key_points: parts.append("\n".join(f"• {p}" for p in r.key_points))
        if r.detailed_body: parts.append(r.detailed_body[:600])
        return "\n\n".join(parts) or f"Research by {r.agent_role}: {r.title}"

    @staticmethod
    def _fallback_exec_summary(query: str, results: List[TaskResult]) -> str:
        summaries = " ".join(r.summary for r in results[:4] if r.summary)
        return f"Research on '{query}': {summaries[:600]}"

    @staticmethod
    def _fallback_conclusion(query: str, results: List[TaskResult]) -> str:
        top = " ".join(r.summary for r in results[:2] if r.summary)
        return f"In conclusion, the research on '{query}' reveals: {top[:400]} Immediate priorities include addressing the key gaps identified across domains."

    def _ollama_call(self, system: str = "", user: str = "", max_tokens: Optional[int] = None) -> str:
        if not system: system = "You are a helpful research assistant."
        payload = {
            "model": self._model, "stream": True,
            "options": {"temperature": 0.3, "num_predict": max_tokens or self._cfg.manager_max_tokens, "num_ctx": self._cfg.num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        tokens: List[str] = []
        with requests.post(self._url, json=payload, stream=True, timeout=(5, getattr(self._cfg, 'ollama_timeout', 120))) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line: continue
                try: chunk = json.loads(raw_line)
                except json.JSONDecodeError: continue
                t = chunk.get("message", {}).get("content", "")
                if t: tokens.append(t)
                if chunk.get("done"): break
        return "".join(tokens)

    @staticmethod
    def _rotate_role(current: str) -> str:
        core_roles = ["FactFinder", "Analyst", "Critic"]
        if current not in core_roles:
            return core_roles[sum(ord(c) for c in current) % 3]
        return {"FactFinder": "Analyst", "Analyst": "Critic", "Critic": "FactFinder"}.get(current, "FactFinder")

    def _emit(self, event_type: EventType, task_id: Optional[str] = None, message: str = "") -> None:
        self.stats.total_events += 1
        self._queue.put(PipelineEvent(event_type=event_type, agent="Manager", task_id=task_id, message=message))

def _parse_task_list(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "",          raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "tasks" in data:
            return data["tasks"]
    except json.JSONDecodeError:
        pass

    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e > s:
        try:
            data = json.loads(raw[s:e+1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    items = []
    for m in re.finditer(r"\{[^{}]+\}", raw, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "title" in obj or "description" in obj:
                items.append(obj)
        except json.JSONDecodeError:
            pass
    return items
