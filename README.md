# 🧠 Multi-Agent Research & Report Generator (MAR System)

## 🚀 Overview

The **Multi-Agent Research System (MAR)** is designed to solve complex research queries through coordinated collaboration between multiple AI agents.

A central **Manager Agent** orchestrates specialized **Worker Agents** to decompose tasks, execute them, evaluate outputs, and generate a structured, high-quality report.

### ✨ Core Capabilities

* Fully **local execution using Ollama** (no external APIs)
* **Dynamic task decomposition**
* **Intelligent retry and reassignment**
* **Retrieval-Augmented Generation (RAG)**
* Optional **web scraping**
* **Transparent execution logs**
* Interactive **Streamlit UI**

---

## 🎯 Objective

This system:

* Accepts a **complex research query**
* Breaks it into **smaller subtasks**
* Assigns tasks to **specialized agents**
* Evaluates outputs for quality
* Handles failures via **retry and reassignment**
* Produces a **coherent structured report**

---

## 🏗️ System Architecture

### 🧠 Manager Agent

Responsible for:

* Task decomposition
* Task classification (fact / analysis / critique)
* Agent assignment
* Quality evaluation
* Retry & reassignment
* Final synthesis

---

### 🤖 Worker Agents

| Agent          | Role                                                         |
| -------------- | ------------------------------------------------------------ |
| **FactFinder** | Retrieves factual data, statistics, and concrete information |
| **Analyst**    | Identifies trends, patterns, and insights                    |
| **Critic**     | Evaluates risks, counterarguments, and limitations           |

---

## 🔁 Orchestration Flow

```
User Query
   ↓
Manager Agent
   ↓
Task Decomposition
   ↓
Task Classification
   ↓
Agent Delegation
   ↓
Worker Execution
   ↓
Quality Evaluation
   ↓
Retry / Reassign (if needed)
   ↓
Final Synthesis
   ↓
Structured Report
```

---

## ⚙️ Key Features

### ✅ 1. Dynamic Task Decomposition

* Adapts number of tasks based on query complexity
* Covers multiple dimensions (technical, economic, ethical, etc.)

---

### ✅ 2. Intelligent Orchestration

* Quality scoring of agent outputs
* Automatic retries for low-quality responses
* Smart reassignment to better-suited agents

---

### ✅ 3. Fully Local LLMs (Ollama)

| Agent   | Model   |
| ------- | ------- |
| Manager | llama3  |
| Workers | mistral |

* No dependency on external APIs
* Runs entirely on local hardware

---

### ✅ 4. Retrieval-Augmented Generation (RAG)

Supports:

* Local document ingestion
* Vector search (e.g., Chroma)
* Context-aware responses

#### Modes:

* Local RAG
* Web Scraping
* Hybrid (RAG + Web)

---

### ✅ 5. Transparent Execution Logs

Example:

```
[12:01:02] Manager → Decomposing tasks
[12:01:05] FactFinder → Executing task
[12:01:08] Manager → Evaluating (score=0.40)
[12:01:09] Manager → Retry triggered
```

Provides full traceability for debugging and evaluation.

---

### ✅ 6. Interactive UI (Streamlit)

Features:

* Query input interface
* Mode selection (RAG / Web / Hybrid)
* Real-time execution logs
* Final report display
* Configurable parameters:

  * Max tokens
  * Timeout
  * Retrieval modes

---

## 🧪 Retry & Resilience

* Quality threshold-based evaluation
* Configurable retry attempts per task
* Automatic reassignment to alternate agents
* Prevents low-quality outputs in final report

---
## Architecture Diagram

The system follows a Manager-Worker multi-agent architecture:

                +----------------------+
                |      User Input      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    Streamlit UI      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    Manager Agent     |
                |----------------------|
                | - Task Decomposition |
                | - Classification     |
                | - Delegation         |
                | - Evaluation         |
                | - Retry Logic        |
                | - Final Synthesis    |
                +----------+-----------+
                           |
        -----------------------------------------
        |                  |                    |
        v                  v                    v
    +---------------+  +---------------+  +---------------+
    |  FactFinder   |  |    Analyst    |  |    Critic     |
    |---------------|  |---------------|  |---------------|
    | - Data fetch  |  | - Insights    |  | - Risks       |
    | - Stats       |  | - Trends      |  | - Weaknesses  |
    +-------+-------+  +-------+-------+  +-------+-------+
            |                  |                  |
            ---------------------------------------
                               |
                               v
                    +----------------------+
                    |   Shared Context     |
                    | (RAG / Web Data)     |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Manager Agent      |
                    |   (Synthesis Phase)  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Final Report       |
                    +----------------------+
---
[ User Interface (Streamlit) ]
       │
       ▼ (Query + Files + URLs)
[ Research Pipeline (Background Thread) ]
       │
       ├──► 1. Data Ingestion Phase
       │       ├─► Web Scraper (Firecrawl API / BS4) ──┐
       │       ├─► Local File Parser (PDF/TXT/DOCX) ───┼─► [ Document Chunker ]
       │                                               │
       │                                               ▼
       │                                       [ Ollama Embeddings ]
       │                                               │
       │                                               ▼
       │                                     [( ChromaDB Vector Store )]
       │
       ├──► 2. Orchestration Phase
               │
               ▼
        [ Manager Agent (LLaMA 3) ] ◄── (Orchestrates)
               │
               ├──► Assess Complexity (1-5 Tasks)
               ├──► Plan & Invent Roles (e.g., "MarineBiologist")
               │
               ├──► 3. Execution Phase (Parallel/Sequential)
               │       ├─► Worker 1 ◄── (Queries) ── [( ChromaDB )]
               │       ├─► Worker 2 ◄── (Queries) ── [( ChromaDB )]
               │       └─► Worker N ◄── (Queries) ── [( ChromaDB )]
               │
               ├──► 4. Quality & Consensus Phase
               │       ├─► Quality Gate (Pass/Fail/Retry)
               │       ├─► Gap Filler (Spins up FactFinder for missing data)
               │       └─► Conflict Resolution (Manager debates contradictions)
               │
               └──► 5. Synthesis Phase
                       ├─► Draft Domain Sections
                       ├─► Draft Executive Summary
                       └─► Draft Conclusion
                               │
                               ▼
                    [ Final Markdown Report ] ──► (Sent back to Streamlit UI)
---

## 📄 Output Format

### 1. Structured Report (Markdown)

Includes:

* Executive Summary
* Key Findings
* Analysis
* Risks & Counterarguments
* Conclusion

Each section is **agent-attributed**.

---

### 2. Execution Logs

* Full pipeline trace
* Timestamped steps
* Debug-friendly output

---

## 🧩 Multi-Agent Coordination

* Shared task context across agents
* Manager-driven orchestration
* Iterative feedback loops

### Conflict Handling

* Detects low-quality or conflicting outputs
* Triggers re-evaluation or reassignment

---

## 🔋 RAG Document Support

Add domain-specific documents in:

```
/data/
```

Example:

```
india_ai_roadmap.txt
```

---

## 🛠️ Setup Instructions

### 1. Clone Repository

```bash
git clone <repo_url>
cd multi-agent-research
```

---

### 2. Create Virtual Environment

```bash
python -m venv mrvenv
mrvenv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Ollama

Download:
https://ollama.com

---

### 5. Pull Models

```bash
ollama pull llama3
ollama pull mistral
ollama pull phi3:mini
ollama pull gemma
```

---

### 6. Run Application

```bash
streamlit run app.py
```

---

## ⚡ Configuration

Configurable via UI:

* Max tokens
* Timeout
* RAG toggle
* Web scraping toggle

---

## 📈 Example Query

```
Design a 10-year roadmap for India to become a global AI superpower
```

---

## 📉 Known Limitations

* RAG may retrieve irrelevant chunks if documents are poorly structured
* No real-time streaming (blocking execution)
* Limited inter-agent conflict detection
* UI can be enhanced (graphs, visual workflows)
* Performance depends on local hardware

---

## 🚀 Future Improvements

* Advanced conflict detection
* Semantic re-ranking for RAG
* Streaming responses
* Visual workflow graphs
* Query result caching
* Multi-modal support (PDFs, images)

---

## 📊 Evaluation Mapping

| Requirement        | Implementation              |
| ------------------ | --------------------------- |
| Task Decomposition | Dynamic Manager logic       |
| Delegation         | Agent routing               |
| Retry Logic        | Quality-based retries       |
| Logging            | Timestamped logs            |
| Specialized Agents | FactFinder, Analyst, Critic |
| Synthesis          | Manager aggregation         |

---

## 📌 Design Decisions

* **Python** for rapid development and ML ecosystem
* **Ollama** for local LLM inference
* **Streamlit** for quick UI development
* Modular agent design for extensibility

---

## 🧠 Key Insight

> Effective AI systems are not just about models, but about **orchestration, reasoning, and coordination between agents**

---

## 👨‍💻 Author

**Aswath Ramana KS**
