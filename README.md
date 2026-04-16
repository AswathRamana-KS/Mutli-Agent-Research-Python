\# 🧠 Multi-Agent Research \& Report Generator (MAR System)



\## 🚀 Overview



This project implements a \*\*Multi-Agent Research System\*\* where a \*\*Manager Agent\*\* orchestrates multiple specialized \*\*Worker Agents\*\* to perform complex research tasks and generate a structured, high-quality report.



The system supports:



\* Fully \*\*local execution using Ollama (no external APIs)\*\*

\* \*\*Dynamic task decomposition\*\*

\* \*\*Retry and reassignment logic\*\*

\* \*\*Local RAG (Retrieval-Augmented Generation)\*\*

\* \*\*Web scraping (optional)\*\*

\* \*\*Interactive UI with full transparency (Streamlit)\*\*



\---



\## 🎯 Objective



As per the assignment requirements , the system:



\* Accepts a \*\*complex research query\*\*

\* Decomposes it into \*\*subtasks\*\*

\* Delegates tasks to \*\*specialized agents\*\*

\* Collects and evaluates responses

\* Handles failures via \*\*retry/reassignment\*\*

\* Synthesizes a \*\*coherent final report\*\*



\---



\## 🏗️ System Architecture



\### 🧠 Manager Agent



Responsible for:



\* Task decomposition (dynamic)

\* Task classification (fact / analysis / critique)

\* Agent assignment

\* Quality evaluation

\* Retry \& reassignment

\* Final synthesis



\---



\### 🤖 Worker Agents



| Agent          | Role                                                         |

| -------------- | ------------------------------------------------------------ |

| \*\*FactFinder\*\* | Retrieves factual data, statistics, and concrete information |

| \*\*Analyst\*\*    | Identifies trends, patterns, and insights                    |

| \*\*Critic\*\*     | Evaluates risks, counterarguments, and limitations           |



\---



\### 🔁 Orchestration Flow



```

User Query

&#x20;  ↓

Manager Agent

&#x20;  ↓

Task Decomposition

&#x20;  ↓

Task Classification

&#x20;  ↓

Agent Delegation

&#x20;  ↓

Worker Execution

&#x20;  ↓

Quality Evaluation

&#x20;  ↓

Retry / Reassign (if needed)

&#x20;  ↓

Final Synthesis

&#x20;  ↓

Structured Report

```



\---



\## ⚙️ Key Features



\### ✅ 1. Dynamic Task Decomposition



\* Number of tasks depends on query complexity

\* Covers multiple dimensions (economic, technical, ethical, etc.)



\---



\### ✅ 2. Intelligent Orchestration



\* Manager evaluates responses using quality scoring

\* Automatically retries low-quality outputs

\* Reassigns tasks to better-suited agents



\---



\### ✅ 3. Fully Local LLMs (Ollama)



\* No external APIs required

\* Runs on local machine



| Agent   | Model   |

| ------- | ------- |

| Manager | llama3  |

| Workers | mistral |



\---



\### ✅ 4. Retrieval-Augmented Generation (RAG)



Supports:



\* Local document ingestion

\* Vector search (Chroma or similar)

\* Context-aware responses



Modes:



\* \*\*Local RAG\*\*

\* \*\*Web Scraping\*\*

\* \*\*Hybrid (both)\*\*



\---



\### ✅ 5. Transparent Execution Logs



Every step is logged:



\* Task creation

\* Agent execution

\* Quality scores

\* Retries

\* Final synthesis



\---



\### ✅ 6. Interactive UI (Streamlit)



Features:



\* Query input

\* Mode selection (RAG / Web / Hybrid)

\* Live execution logs

\* Final report display

\* Config controls (tokens, timeout)



\---



\## 📊 Execution Logging



Logs include:



```

\[Time] Manager → Decomposing tasks

\[Time] FactFinder → Executing task

\[Time] Manager → Evaluating (score=0.40)

\[Time] Manager → Retry triggered

```



This ensures \*\*full traceability\*\* as required by the assignment .



\---



\## 🧪 Retry \& Resilience



\* Quality threshold-based evaluation

\* Max retry attempts per task

\* Automatic reassignment to different agent

\* Prevents low-quality outputs from reaching final report



\---



\## 📄 Output Format



The system produces:



\### 1. Structured Report (Markdown)



Includes:



\* Executive Summary

\* Key Findings

\* Analysis

\* Risks \& Counterarguments

\* Conclusion



Each section is \*\*agent-attributed\*\*



\---



\### 2. Execution Logs



\* Full pipeline trace

\* Debug-friendly

\* Useful for evaluation



\---



\## 🧩 Multi-Agent Coordination



Agents collaborate via:



\* Shared task context

\* Manager-controlled orchestration

\* Iterative feedback loops



Conflict resolution:



\* Manager detects low-quality or conflicting responses

\* Triggers re-evaluation or reassignment



\---



\## 🔋 RAG Document Support



You can add domain-specific knowledge by placing files in:



```

/data/

```



Example:



```

india\_ai\_roadmap.txt

```



\---



\## 🛠️ Setup Instructions



\### 1. Clone Repository



```bash

git clone <repo\_url>

cd multi-agent-research

```



\---



\### 2. Create Virtual Environment



```bash

python -m venv mrvenv

mrvenv\\Scripts\\activate   # Windows

```



\---



\### 3. Install Dependencies



```bash

pip install -r requirements.txt

```



\---



\### 4. Install Ollama



Download and install:

👉 https://ollama.com



\---



\### 5. Pull Models



```bash

ollama pull llama3

ollama pull mistral

```



\---



\### 6. Run Application



```bash

streamlit run app.py

```



\---



\## ⚡ Configuration



Configurable via UI:



\* Max tokens

\* Timeout

\* RAG toggle

\* Web scraping toggle



\---



\## 📈 Example Query



```

Design a 10-year roadmap for India to become a global AI superpower

```



\---



\## 📉 Known Limitations



\* RAG retrieval may return irrelevant chunks if documents are not well structured

\* No real-time streaming (blocking execution)

\* Limited conflict detection between agents

\* UI visualization can be further improved (graphs, agent cards)

\* Model performance constrained by local hardware



\---



\## 🚀 Future Improvements



\* Advanced conflict detection between agents

\* Better RAG ranking (semantic reranking)

\* Streaming responses in UI

\* Visual workflow graph (agent nodes)

\* Caching repeated queries

\* Multi-modal support (PDF, images)



\---



\## 📊 Evaluation Mapping



| Requirement        | Implementation              |

| ------------------ | --------------------------- |

| Task Decomposition | Dynamic Manager logic       |

| Delegation         | Agent routing               |

| Retry Logic        | Quality-based retries       |

| Logging            | Timestamped logs            |

| Specialized Agents | FactFinder, Analyst, Critic |

| Synthesis          | Manager merges outputs      |



\---



\## 📌 Design Decisions



\* Python chosen for rapid prototyping and ML ecosystem

\* Ollama used for fully local inference

\* Streamlit for fast UI development

\* Modular agent design for extensibility



\---



\## 🧠 Key Insight



This system demonstrates that:



> Effective AI systems are not just about models, but about \*\*orchestration, reasoning, and coordination between agents\*\*



\---



\## 👨‍💻 Author



Ashwath



\---



\## 📜 License



MIT License



