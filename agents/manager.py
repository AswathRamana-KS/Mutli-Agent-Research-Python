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
"""
from __future__ import annotations

import json
import queue
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import AgentRole, Config
from agents.base import _extract_json, _extract_float, _is_truncated
from agents.workers import create_worker, available_roles
from core.models import (
    EventType, ExecutionStats, FinalReport,
    PipelineEvent, ResearchTask, TaskResult, TaskStatus,
)

# ═════════════════════════════════════════════════════════════════════
# Domain → Agent role routing table
# ═════════════════════════════════════════════════════════════════════

DOMAIN_ROUTE: List[Tuple[List[str], AgentRole]] = [
    (["economic", "gdp", "job", "employment", "market", "industrial revolution",
      "sector", "forecast", "growth rate", "displacement"],     AgentRole.ECONOMIST),
    (["policy", "governance", "regulation", "law", "act",
      "framework", "eu ai act", "nist", "legislation",
      "compliance", "oversight"],                               AgentRole.POLICY_ANALYST),
    (["compute", "gpu", "infrastructure", "model training",
      "talent", "researcher", "engineer", "research output",
      "publication", "patent", "ml", "machine learning",
      "deep learning", "technology strategy", "capability",
      "benchmark"],                                             AgentRole.ML_ENGINEER),
    (["ethic", "bias", "fairness", "surveillance", "inequality",
      "trust", "stakeholder", "social impact", "accountability",
      "transparency", "human rights", "misuse"],                AgentRole.ETHICIST),
    (["environment", "climate", "energy", "carbon", "emission",
      "sustainability", "renewable", "data centre", "water",
      "green", "net zero", "co2"],                              AgentRole.CLIMATE_SCIENTIST),
    (["geopolit", "alliance", "trade", "sovereignty", "arms race",
      "china", "united states", "us-china", "diplomacy",
      "international", "superpower", "competition"],            AgentRole.GEOPOLITICAL),
    (["execution", "roadmap", "phase", "milestone", "timeline",
      "plan", "0-2 year", "3-5 year", "6-10 year",
      "implementation", "failure scenario"],                    AgentRole.ANALYST),
    (["risk", "counter", "challenge", "threat", "limitation",
      "worst case", "failure"],                                 AgentRole.CRITIC),
    (["fact", "statistic", "data", "history", "evidence"],     AgentRole.FACT_FINDER),
]

COMPLEXITY_TASKS = {"simple": 3, "moderate": 4, "complex": 6, "expert": 8}

# ═════════════════════════════════════════════════════════════════════
# Prompts
# ═════════════════════════════════════════════════════════════════════

_COMPLEXITY_PROMPT = """Classify this research query as: simple, moderate, complex, or expert.
simple   = one factual topic, 3 tasks enough
moderate = 2-3 angles, 4 tasks
complex  = 4-6 distinct domains
expert   = 7+ explicit domains/requirements

Reply with ONLY one word."""

_DECOMPOSE_PROMPT = """You are the Manager Agent orchestrating a research system.
Break this query into {n} research sub-tasks. Each task must cover a DIFFERENT domain.

Available agents (assign the best fit for each task):
  FactFinder      - facts, statistics, history, evidence
  Analyst         - trends, patterns, strategic insights, execution plans
  Critic          - risks, counterarguments, failure modes, stress-testing
  Economist       - GDP, jobs, economic modeling, sector analysis
  PolicyAnalyst   - regulations, governance, legal frameworks, policy proposals
  MLEngineer      - compute infrastructure, talent, AI capabilities, technical gaps
  Ethicist        - bias, inequality, surveillance, public trust, stakeholder views
  ClimateScientist - energy demands, environmental impact, sustainability
  GeopoliticalAnalyst - alliances, trade, sovereignty, AI arms race, diplomacy

Rules:
- Create EXACTLY {n} tasks.
- Each task description must be specific and actionable (2-3 sentences).
- No overlap between tasks.
- Cover ALL explicit domains mentioned in the query.

Respond with ONLY a JSON array, no fences:
[
  {{"title": "short title", "description": "specific 2-3 sentence instructions", "assigned_to": "AgentName", "domain": "Domain Label"}},
  ...
]"""

_QUALITY_PROMPT = """Score this research output 0.0-1.0.
Good output (score > 0.6): contains specific facts/data, substantive analysis, 3+ sentences.
Poor output (score < 0.4): vague, very short, no specific data.
Reply with ONLY a number like 0.7"""

_EXEC_SUMMARY_PROMPT = """Write an Executive Summary for this research report.
Be specific: include key numbers, recommendations, and the most important insight.
Write 2-3 paragraphs. Be direct and substantive."""

_SECTION_PROMPT = """You are a research editor synthesising findings for a specific domain section.

Domain: {domain}
Agent who researched this: {agent}

Agent's findings:
{content}

Write a well-structured section for the research report on this domain.
- Start with the most important insight.
- Include specific numbers, examples, and recommendations.
- Write 2-3 substantive paragraphs.
- Do NOT start with "In this section" or generic phrases.
Write the section now:"""

_CONCLUSION_PROMPT = """Write a Conclusion for this research report on: {query}

Key findings from all domains:
{findings_list}

Write a 2-paragraph conclusion that:
1. Synthesises the most critical cross-domain insights
2. Ends with clear, prioritised next steps

Be specific and actionable."""

class ManagerAgent:

    def __init__(self, config: Config, event_queue: queue.Queue) -> None:
        self._cfg   = config
        self._queue = event_queue
        self._model = config.agent_models[AgentRole.MANAGER]
        self._url   = f"{config.ollama_base_url}/api/chat"
        self.stats  = ExecutionStats()

    def run(self, query: str, retriever=None) -> Tuple[FinalReport, ExecutionStats]:
        t0 = time.monotonic()
        self._emit(EventType.PIPELINE_START,
                   message=f"🧠 Manager starting pipeline: '{query[:80]}'")

        n_tasks, complexity = self._assess_complexity(query)
        tasks = self._plan_tasks(query, n_tasks, complexity)
        self.stats.tasks_total = len(tasks)

        results: List[TaskResult] = []
        for i, task in enumerate(tasks, 1):
            self._emit(EventType.TASK_ASSIGNED, task.task_id,
                       f"📤 [{i}/{len(tasks)}] → {task.assigned_to}: {task.title}")
            result = self._execute_one(task, retriever)
            if result:
                results.append(result)
                self.stats.tasks_done += 1

        if not results:
            raise RuntimeError("All worker tasks failed.")

        # 4. Review: ACTIVE CONFLICT RESOLUTION AND GAP FILLING
        self._detect_and_resolve_contradictions(results)
        self._detect_and_fill_gaps(tasks, results, retriever)

        # 5. Synthesise
        self._emit(EventType.SYNTHESIS_START,
                   message=f"📝 Manager synthesising {len(results)} domain results…")
        report = self._synthesise(query, results, tasks)

        self.stats.duration_seconds = round(time.monotonic() - t0, 1)
        self._emit(EventType.PIPELINE_COMPLETE,
                   message=f"✅ Done in {self.stats.duration_seconds}s | "
                           f"{len(results)}/{len(tasks)} tasks | "
                           f"{len(report.sections)} sections in report")
        return report, self.stats

    def _assess_complexity(self, query: str) -> Tuple[int, str]:
        try:
            raw  = self._ollama_call(_COMPLEXITY_PROMPT, f"Query: {query}", max_tokens=5)
            word = raw.strip().lower().split()[0] if raw.strip() else "moderate"
            if word not in COMPLEXITY_TASKS:
                word = "moderate"
        except Exception:
            word = "moderate"

        q_lower = query.lower()
        domain_hits = sum(
            1 for keywords, _ in DOMAIN_ROUTE
            if any(kw in q_lower for kw in keywords)
        )
        if domain_hits >= 6 and word in ("simple", "moderate"):
            word = "complex"
        if domain_hits >= 8:
            word = "expert"

        n = COMPLEXITY_TASKS[word]
        self._emit(EventType.LOG,
                   message=f"🗂 Query complexity: [{word.upper()}] → {n} tasks "
                           f"({domain_hits} domain keywords detected)")
        return n, word

    def _plan_tasks(self, query: str, n_tasks: int, complexity: str) -> List[ResearchTask]:
        self._emit(EventType.LOG, message=f"🗂 Planning {n_tasks} domain-specific tasks…")

        raw_list: List[Dict] = []
        for attempt in range(2):
            try:
                prompt = _DECOMPOSE_PROMPT.format(n=n_tasks)
                raw    = self._ollama_call(prompt, f"Research query:\n{query}", max_tokens=self._cfg.manager_max_tokens)
                raw_list = _parse_task_list(raw)
                if len(raw_list) >= 3:
                    break
                self._emit(EventType.LOG,
                           message=f"⚠️ Decompose attempt {attempt+1}: only {len(raw_list)} tasks")
            except Exception as exc:
                self._emit(EventType.LOG, message=f"⚠️ Decompose error: {exc}")

        if len(raw_list) < 3:
            self._emit(EventType.LOG, message="⚠️ Using domain-detection fallback")
            raw_list = self._domain_fallback_tasks(query, n_tasks)

        valid_roles = {r.value.lower(): r for r in available_roles()}
        tasks: List[ResearchTask] = []

        for item in raw_list[:n_tasks]:
            assigned_str = str(item.get("assigned_to", "FactFinder"))
            role = valid_roles.get(assigned_str.lower().replace(" ", "").replace("_", ""))
            if role is None:
                role = self._route_by_domain(
                    item.get("domain", "") + " " + item.get("description", "")
                )

            domain = str(item.get("domain", item.get("title", "General")))
            task   = ResearchTask(
                title       = str(item.get("title", "Research Task")),
                description = str(item.get("description", query)),
                assigned_to = role.value,
                domain      = domain,
                max_retries = self._cfg.max_retries,
            )
            tasks.append(task)
            self._emit(EventType.TASK_CREATED, task.task_id,
                       f"📋 [{role.value}] {task.domain}: {task.title}")

        return tasks

    def _domain_fallback_tasks(self, query: str, n_tasks: int) -> List[Dict[str, Any]]:
        q_lower = query.lower()
        found: List[Tuple[str, AgentRole, str]] = []

        domain_labels = {
            AgentRole.ECONOMIST:         "Economic Modeling",
            AgentRole.POLICY_ANALYST:    "Policy & Governance",
            AgentRole.ML_ENGINEER:       "Technology Strategy",
            AgentRole.ETHICIST:          "Ethical & Social Impact",
            AgentRole.CLIMATE_SCIENTIST: "Environmental Constraints",
            AgentRole.GEOPOLITICAL:      "Geopolitical Implications",
            AgentRole.ANALYST:           "Execution Plan & Strategy",
            AgentRole.CRITIC:            "Risks & Counterarguments",
            AgentRole.FACT_FINDER:       "Background & Evidence",
        }

        for keywords, role in DOMAIN_ROUTE:
            if any(kw in q_lower for kw in keywords):
                label = domain_labels.get(role, role.value)
                if label not in [f[0] for f in found]:
                    found.append((label, role, f"Research the {label} aspects of: {query[:200]}"))

        defaults = [
            ("Background & Evidence",     AgentRole.FACT_FINDER,   f"Gather key facts and evidence for: {query[:200]}"),
            ("Strategic Analysis",        AgentRole.ANALYST,        f"Analyse trends and insights for: {query[:200]}"),
            ("Risks & Counterarguments",  AgentRole.CRITIC,         f"Identify risks and challenges for: {query[:200]}"),
        ]
        existing_roles = {f[1] for f in found}
        for label, role, desc in defaults:
            if role not in existing_roles:
                found.append((label, role, desc))

        return [
            {"title": label, "description": desc, "assigned_to": role.value, "domain": label}
            for label, role, desc in found[:n_tasks]
        ]

    @staticmethod
    def _route_by_domain(text: str) -> AgentRole:
        t = text.lower()
        for keywords, role in DOMAIN_ROUTE:
            if any(kw in t for kw in keywords):
                return role
        return AgentRole.FACT_FINDER

    def _execute_one(self, task: ResearchTask, retriever=None) -> Optional[TaskResult]:
        task.status = TaskStatus.IN_PROGRESS
        role   = AgentRole(task.assigned_to)
        worker = create_worker(role, self._cfg, self._queue)
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
                self._emit(EventType.AGENT_FAILED, task.task_id,
                           f"❌ Worker error attempt {attempt+1}: {exc}")
                time.sleep(1)
                continue

            score = self._evaluate_quality(task, result)
            result.quality_score = score

            if best is None or score > (best.quality_score or 0):
                best = result

            if score >= self._cfg.quality_threshold:
                self._emit(EventType.QUALITY_PASS, task.task_id,
                           f"✅ Accepted (q={score:.2f}, words={result.word_count})")
                task.status = TaskStatus.COMPLETE
                return result

            self.stats.retries += 1
            self._emit(EventType.QUALITY_FAIL, task.task_id,
                       f"⚠️ Low quality (q={score:.2f}) — attempt {attempt+1}/{task.max_retries+1}")

            if attempt < task.max_retries:
                new_role = self._rotate_role(role)
                self._emit(EventType.AGENT_RETRY, task.task_id,
                           f"🔄 Reassigning: {role.value} → {new_role.value}")
                role   = new_role
                worker = create_worker(new_role, self._cfg, self._queue)

        if best:
            task.status = TaskStatus.COMPLETE
            self._emit(EventType.QUALITY_FAIL, task.task_id,
                       f"⚠️ Accepting best result (q={best.quality_score:.2f})")
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
        except Exception:
            return 0.65

    # 🎯 FIX: Active Conflict Resolution (uses Manager LLM to find the truth)
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
                            
                            # Manager steps in to resolve the conflict
                            prompt = (
                                f"Two research agents provided conflicting information on the topic '{topic}'.\n\n"
                                f"Agent 1 ({r1.agent_role}): {r1.summary}\n\n"
                                f"Agent 2 ({r2.agent_role}): {r2.summary}\n\n"
                                f"Analyze both claims and write a definitive, objective 2-paragraph resolution that finds the truth."
                            )
                            resolution = self._ollama_call(system="You are an expert consensus builder and fact-checker.", user=prompt, max_tokens=self._cfg.synth_max_tokens)
                            
                            if len(resolution.split()) > 10:
                                # BUG 5 FIX: Save to dedicated field instead of mutating detailed_body
                                r1.conflict_resolution = f"**[Manager's Conflict Resolution vs {r2.agent_role}]:**\n{resolution}"
                                self._emit(EventType.LOG, message=f"✅ Conflict resolved for '{topic}'.")
                            break

    # 🎯 FIX: Active Gap Filling (spins up a FactFinder to get missing data)
    def _detect_and_fill_gaps(self, tasks: List[ResearchTask], results: List[TaskResult], retriever) -> None:
        done_ids = {r.task_id for r in results}
        for t in tasks:
            if t.task_id not in done_ids:
                self.stats.gaps_found += 1
                self._emit(EventType.GAP_FOUND,
                           message=f"🔍 Gap: '{t.title}' failed. Re-querying FactFinder to fill gap...")
                
                worker = create_worker(AgentRole.FACT_FINDER, self._cfg, self._queue)
                try:
                    t.max_retries = 0 
                    t.assigned_to = AgentRole.FACT_FINDER.value
                    
                    rag_ctx = ""
                    if retriever:
                        rag_ctx = retriever.retrieve(t.description)
                        
                    res = worker.execute(t, rag_ctx)
                    results.append(res)
                    self._emit(EventType.LOG, message=f"✅ Gap filled successfully for '{t.title}'.")
                except Exception as e:
                    self._emit(EventType.LOG, message=f"❌ Could not fill gap for '{t.title}': {e}")


    def _synthesise(
        self,
        query:   str,
        results: List[TaskResult],
        tasks:   List[ResearchTask],
    ) -> FinalReport:

        sections:    Dict[str, str] = {}
        key_findings: List[str]     = []
        contributions: Dict[str, str] = {}

        domain_results: Dict[str, TaskResult] = {}
        for r in results:
            key = r.domain or r.title or r.agent_role
            domain_results[key] = r

        for domain, r in domain_results.items():
            self._emit(EventType.SYNTHESIS_SECTION,
                       message=f"📝 Synthesising section: {domain} [{r.agent_role}]")

            # BUG 5 FIX: Inject the conflict resolution if it exists
            resolution_text = f"\n\n{r.conflict_resolution}" if r.conflict_resolution else ""
            
            raw_content = (
                f"Summary: {r.summary}\n\n"
                f"Key points:\n" + "\n".join(f"- {p}" for p in r.key_points) +
                f"\n\nDetailed findings:\n{r.detailed_body}"
                f"{resolution_text}"
            )

            prompt = _SECTION_PROMPT.format(
                domain  = domain,
                agent   = r.agent_role,
                content = raw_content[:1400],
            )

            section_text = self._synth_section(
                user     = prompt,
                fallback = self._fallback_section(r),
            )
            sections[domain] = section_text

            for pt in r.key_points[:2]:
                if pt and len(pt.split()) > 4:
                    key_findings.append(pt)

            contributions[r.agent_role] = r.summary[:180] if r.summary else r.title

        seen: set = set()
        unique_findings: List[str] = []
        for f in key_findings:
            if f not in seen:
                seen.add(f)
                unique_findings.append(f)
        key_findings = unique_findings[:10]

        self._emit(EventType.LOG, message="📝 Writing executive summary…")
        all_summaries = "\n\n".join(
            f"[{r.agent_role} — {r.domain}]: {r.summary}" for r in results[:6]
        )
        exec_summary = self._synth_section(
            user = f"Query: {query}\n\nResearch summaries by domain:\n{all_summaries[:1600]}",
            system_override = _EXEC_SUMMARY_PROMPT,
            fallback = self._fallback_exec_summary(query, results),
        )

        self._emit(EventType.LOG, message="📝 Writing conclusion…")
        findings_list = "\n".join(f"- {f}" for f in key_findings[:8])
        conclusion = self._synth_section(
            user = _CONCLUSION_PROMPT.format(query=query, findings_list=findings_list),
            fallback = self._fallback_conclusion(query, results),
        )

        self._emit(EventType.SYNTHESIS_COMPLETE,
                   message=f"📄 Report complete: {len(sections)} sections, {len(key_findings)} key findings")
        self.stats.total_events += 1

        return FinalReport(
            query               = query,
            sections            = sections,
            key_findings        = key_findings,
            agent_contributions = contributions,
            executive_summary   = exec_summary,
            conclusion          = conclusion,
        )


    _DEFAULT_SYNTH_SYSTEM = (
        "You are a senior research editor. "
        "Write clear, substantive prose based on the research provided. "
        "Be specific: use numbers, examples, and direct recommendations. "
        "Do not use filler phrases. Write at least 2 solid paragraphs."
    )

    def _synth_section(
        self,
        user:            str,
        fallback:        str,
        system_override: str = "",
    ) -> str:
        system = system_override or self._DEFAULT_SYNTH_SYSTEM
        try:
            text = self._ollama_call(
                system     = system,
                user       = user,
                max_tokens = self._cfg.synth_max_tokens,
            )
            text = text.strip()
            if len(text.split()) >= 20:
                return text
        except Exception as exc:
            self._emit(EventType.LOG, message=f"⚠️ Synthesis call failed: {exc}")
        return fallback

    @staticmethod
    def _fallback_section(r: TaskResult) -> str:
        parts = [r.summary] if r.summary else []
        if r.key_points:
            parts.append("\n".join(f"• {p}" for p in r.key_points))
        if r.detailed_body:
            parts.append(r.detailed_body[:600])
        return "\n\n".join(parts) or f"Research by {r.agent_role}: {r.title}"

    @staticmethod
    def _fallback_exec_summary(query: str, results: List[TaskResult]) -> str:
        summaries = " ".join(r.summary for r in results[:4] if r.summary)
        return f"Research on '{query}': {summaries[:600]}"

    @staticmethod
    def _fallback_conclusion(query: str, results: List[TaskResult]) -> str:
        top = " ".join(r.summary for r in results[:2] if r.summary)
        return (
            f"In conclusion, the research on '{query}' reveals: {top[:400]} "
            f"Immediate priorities include addressing the key gaps identified across domains. "
            f"Further investigation is recommended to refine the execution roadmap."
        )

    def _ollama_call(
        self,
        system: str = "",
        user:   str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        if not system:
            system = "You are a helpful research assistant."
        payload = {
            "model":  self._model,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens or self._cfg.manager_max_tokens,
                "num_ctx":     self._cfg.num_ctx,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        tokens: List[str] = []
        with requests.post(self._url, json=payload, stream=True, timeout=(5, 90)) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                t = chunk.get("message", {}).get("content", "")
                if t:
                    tokens.append(t)
                if chunk.get("done"):
                    break
        return "".join(tokens)

    @staticmethod
    def _rotate_role(current: AgentRole) -> AgentRole:
        return {
            AgentRole.FACT_FINDER:       AgentRole.ANALYST,
            AgentRole.ANALYST:           AgentRole.FACT_FINDER,
            AgentRole.CRITIC:            AgentRole.ANALYST,
            AgentRole.ECONOMIST:         AgentRole.ANALYST,
            AgentRole.POLICY_ANALYST:    AgentRole.CRITIC,
            AgentRole.ML_ENGINEER:       AgentRole.FACT_FINDER,
            AgentRole.ETHICIST:          AgentRole.ANALYST,
            AgentRole.CLIMATE_SCIENTIST: AgentRole.FACT_FINDER,
            AgentRole.GEOPOLITICAL:      AgentRole.ANALYST,
        }.get(current, AgentRole.FACT_FINDER)

    def _emit(self, et: EventType, tid: Optional[str] = None, msg: str = "") -> None:
        self.stats.total_events += 1
        self._queue.put(PipelineEvent(
            event_type=et, agent=AgentRole.MANAGER.value, task_id=tid, message=msg
        ))

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
