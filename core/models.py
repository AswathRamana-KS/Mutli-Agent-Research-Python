"""
Core Pydantic models + event system.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE    = "complete"
    FAILED      = "failed"
    RETRYING    = "retrying"

class RetrievalMode(str, Enum):
    NONE   = "none"
    LOCAL  = "local_rag"
    WEB    = "web_scrape"
    HYBRID = "hybrid"

class EventType(str, Enum):
    PIPELINE_START    = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR    = "pipeline_error"
    AGENT_START       = "agent_start"
    AGENT_THINKING    = "agent_thinking"
    AGENT_COMPLETE    = "agent_complete"
    AGENT_RETRY       = "agent_retry"
    AGENT_FAILED      = "agent_failed"
    TASK_CREATED      = "task_created"
    TASK_ASSIGNED     = "task_assigned"
    TASK_COMPLETE     = "task_complete"
    QUALITY_CHECK     = "quality_check"
    QUALITY_PASS      = "quality_pass"
    QUALITY_FAIL      = "quality_fail"
    CONTRADICTION     = "contradiction"
    GAP_FOUND         = "gap_found"
    RAG_QUERY         = "rag_query"
    RAG_RESULT        = "rag_result"
    SCRAPE_START      = "scrape_start"
    SCRAPE_DONE       = "scrape_done"
    SYNTHESIS_START   = "synthesis_start"
    SYNTHESIS_SECTION = "synthesis_section"
    SYNTHESIS_COMPLETE = "synthesis_complete"
    LOG               = "log"

class PipelineEvent(BaseModel):
    event_id:   str      = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: EventType
    agent:      Optional[str] = None
    task_id:    Optional[str] = None
    message:    str = ""
    data:       Dict[str, Any] = Field(default_factory=dict)
    timestamp:  datetime = Field(default_factory=datetime.now)

    def fmt_time(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

class ResearchTask(BaseModel):
    task_id:        str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title:          str
    description:    str
    assigned_to:    str          
    system_prompt:  str = ""     
    domain:         str = ""     
    status:         TaskStatus = TaskStatus.PENDING
    retrieval_mode: RetrievalMode = RetrievalMode.NONE
    retry_count:    int = 0
    max_retries:    int = 1
    created_at:     datetime = Field(default_factory=datetime.now)

class TaskResult(BaseModel):
    task_id:        str
    agent_role:     str
    domain:         str = ""
    title:          str
    summary:        str
    key_points:     List[str] = Field(default_factory=list)
    detailed_body:  str
    sources:        List[str] = Field(default_factory=list)
    confidence:     float = 0.7
    quality_score:  Optional[float] = None
    rag_chunks_used: int = 0
    word_count:     int = 0
    conflict_resolution: Optional[str] = None # 🎯 Ensures Manager can save conflict resolutions

class FinalReport(BaseModel):
    query:       str
    run_id:      str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    generated_at: datetime = Field(default_factory=datetime.now)
    sections: Dict[str, str] = Field(default_factory=dict)
    key_findings:        List[str] = Field(default_factory=list)
    agent_contributions: Dict[str, str] = Field(default_factory=dict)
    executive_summary:   str = ""
    conclusion:          str = ""

    def to_markdown(self) -> str:
        lines: List[str] = [f"# Research Report", f"", f"**Query:** {self.query}", f"", f"---", f""]
        
        if self.executive_summary: 
            lines += ["## Executive Summary", "", self.executive_summary, ""]
            
        if self.key_findings:
            lines += ["## Key Findings", ""]
            for i, f in enumerate(self.key_findings, 1): lines.append(f"{i}. {f}")
            lines.append("")
            
        if self.sections:
            lines += ["---", "", "## Detailed Analysis by Domain", ""]
            for domain, content in self.sections.items(): lines += [f"### {domain}", "", content, ""]
            
        if self.conclusion: 
            lines += ["---", "", "## Conclusion", "", self.conclusion, ""]
            
        if self.agent_contributions:
            lines += ["---", "", "## Agent Contributions", ""]
            for agent, contrib in self.agent_contributions.items(): lines.append(f"**{agent}**: {contrib}")
            lines.append("")
            
        lines.append(f"*Generated {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')} · Run ID: {self.run_id}*")
        return "\n".join(lines)

class ExecutionStats(BaseModel):
    total_events:     int   = 0
    tasks_done:       int   = 0
    tasks_total:      int   = 0
    retries:          int   = 0
    contradictions:   int   = 0
    gaps_found:       int   = 0
    rag_queries:      int   = 0
    scrape_calls:     int   = 0
    duration_seconds: float = 0.0