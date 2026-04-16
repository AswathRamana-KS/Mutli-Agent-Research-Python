"""
Worker Agents — Core agents + Dynamic generation capability.
"""
from __future__ import annotations
import queue
from config import Config
from agents.base import OllamaBaseAgent

class FactFinderAgent(OllamaBaseAgent):
    @property
    def role(self) -> str:
        return "FactFinder"

    @property
    def system_prompt(self) -> str:
        return (
            "You are FactFinder — a precise, evidence-first research specialist.\n"
            "MISSION: Retrieve verifiable facts, statistics, historical data, and concrete examples.\n"
            "RULES:\n"
            "- Cite specific numbers, years, names, and sources whenever possible.\n"
            "- Every claim must be traceable to a named source or established record.\n"
            "- Flag uncertainty explicitly: if a figure is estimated, say so.\n"
            "- DO NOT analyse — just report facts with supporting evidence.\n"
            "TONE: Encyclopaedic, neutral, precise."
        )

class AnalystAgent(OllamaBaseAgent):
    @property
    def role(self) -> str:
        return "Analyst"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Analyst — a strategic thinker who finds deeper meaning in data.\n"
            "MISSION: Identify trends, patterns, causal relationships, and strategic insights.\n"
            "RULES:\n"
            "- Move BEYOND raw facts — synthesise what they mean collectively.\n"
            "- Distinguish correlation from causation. Be explicit.\n"
            "- Identify 2-3 bold strategic takeaways with clear reasoning.\n"
            "- Use data to support every claim (percentages, growth rates, comparisons).\n"
            "TONE: Insightful, data-grounded, forward-looking."
        )

class CriticAgent(OllamaBaseAgent):
    @property
    def role(self) -> str:
        return "Critic"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Critic — a rigorous devil's advocate and risk analyst.\n"
            "MISSION: Identify risks, counterarguments, hidden assumptions, and failure modes.\n"
            "RULES:\n"
            "- Challenge consensus views — what do sceptics and minority voices say?\n"
            "- Rate each risk: [HIGH], [MEDIUM], or [LOW] severity.\n"
            "- Cite historical precedents where similar things failed.\n"
            "- Surface hidden assumptions and test their validity.\n"
            "- DO NOT offer solutions — your job is stress-testing, not fixing.\n"
            "TONE: Sceptical, adversarial-but-constructive, rigorous."
        )

class DynamicSpecialistAgent(OllamaBaseAgent):
    """An agent created on-the-fly by the Manager to answer highly specific domain questions."""
    def __init__(self, role_name: str, prompt: str, config: Config, event_queue: queue.Queue):
        super().__init__(config, event_queue)
        self._role_name = role_name
        self._prompt = prompt

    @property
    def role(self) -> str:
        return self._role_name

    @property
    def system_prompt(self) -> str:
        return self._prompt

def create_worker(role_name: str, config: Config, event_queue: queue.Queue, custom_prompt: str = "") -> OllamaBaseAgent:
    core_map = {
        "FactFinder": FactFinderAgent,
        "Analyst":    AnalystAgent,
        "Critic":     CriticAgent,
    }
    
    cls = core_map.get(role_name)
    if cls:
        return cls(config=config, event_queue=event_queue)
        
    # If the manager assigned a role that isn't core, spin up a dynamic specialist!
    if not custom_prompt:
        custom_prompt = f"You are {role_name}. Provide highly specific, expert analysis for your domain."
        
    return DynamicSpecialistAgent(role_name, custom_prompt, config, event_queue)