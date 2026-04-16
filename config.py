"""
Configuration — Dynamic Multi-Agent Research
"""
from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

RECOMMENDED_MODELS: Dict[str, str] = {
    "llama3:8b":      "LLaMA 3 8B  · best reasoning / orchestration  (4.7 GB) ★ recommended",
    "mistral:7b":     "Mistral 7B  · fast, factual                   (4.1 GB) ★ recommended",
    "phi3:mini":      "Phi-3 Mini  · fastest, good for simple tasks  (2.2 GB)",
    "llama3.2:3b":    "LLaMA 3.2 3B · lightweight                   (2.0 GB)",
    "qwen2.5:7b":     "Qwen 2.5 7B · strong multi-domain reasoning  (4.4 GB)",
    "gemma2:9b":      "Gemma 2 9B  · high quality but slower        (5.5 GB)",
    "deepseek-r1:7b": "DeepSeek-R1 7B · chain-of-thought            (4.7 GB)",
    "llama3.1:8b":    "LLaMA 3.1 8B · tool-aware                   (4.7 GB)",
}

EMBEDDING_MODELS: Dict[str, str] = {
    "nomic-embed-text":  "Nomic Embed (Ollama) — no extra install needed",
    "all-MiniLM-L6-v2":  "MiniLM-L6  (sentence-transformers) — 80 MB",
    "all-mpnet-base-v2": "MPNet-Base (sentence-transformers) — better quality",
}

DEFAULT_AGENT_MODELS: Dict[str, str] = {
    "Manager":    "llama3:8b",
    "FactFinder": "mistral:7b",
    "Analyst":    "llama3:8b",
    "Critic":     "mistral:7b",
    "Dynamic":    "mistral:7b", # Default model for newly invented agents
}

CORE_AGENT_COLORS = {
    "Manager": "#4B9EFF", "FactFinder": "#4CAF50", 
    "Analyst": "#9C27B0", "Critic": "#FF9800"
}
CORE_AGENT_ICONS = {
    "Manager": "🧠", "FactFinder": "🔍", 
    "Analyst": "📊", "Critic": "⚖️"
}

def get_agent_color(role_name: str) -> str:
    if role_name in CORE_AGENT_COLORS: return CORE_AGENT_COLORS[role_name]
    # Dynamically generate a consistent pastel/bright color for new agents based on their name
    h = hashlib.md5(role_name.encode()).hexdigest()
    return f"#{h[:6]}"

def get_agent_icon(role_name: str) -> str:
    if role_name in CORE_AGENT_ICONS: return CORE_AGENT_ICONS[role_name]
    # Assign icons based on keyword hints, fallback to a specialist icon
    n = role_name.lower()
    if any(k in n for k in ["tech", "engineer", "dev"]): return "⚙️"
    if any(k in n for k in ["science", "bio", "chem"]): return "🔬"
    if any(k in n for k in ["law", "policy", "legal"]): return "📜"
    if any(k in n for k in ["econ", "market", "finance"]): return "💰"
    if any(k in n for k in ["psycho", "behave", "human", "social"]): return "🤝"
    if any(k in n for k in ["game", "play", "design"]): return "🎮"
    if any(k in n for k in ["health", "med", "doctor"]): return "⚕️"
    return "💡"

@dataclass
class Config:
    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Token budgets
    worker_max_tokens: int = 900
    synth_max_tokens:  int = 700
    manager_max_tokens: int = 800
    num_ctx: int = 4096

    # Agent models
    agent_models: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_AGENT_MODELS)
    )

    # Embeddings
    embed_model:   str = "nomic-embed-text"
    embed_backend: str = "ollama"
    chroma_path: str = "./chroma_db"
    top_k_rag:   int = 5
    firecrawl_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_API_KEY")
    )
    
    # Orchestration
    max_retries:       int   = 1
    quality_threshold: float = 0.5 
    ollama_timeout: int = 120  

    @property
    def max_tokens(self) -> int:
        return self.worker_max_tokens
        
    @max_tokens.setter
    def max_tokens(self, value: int):
        self.worker_max_tokens = value

_config: Optional[Config] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config