# 🛰️ Multimodal Geospatial Crisis Response & Infrastructure GraphRAG

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-Tactical_HUD-00F0FF.svg)](https://streamlit.io/)
[![Dark Mode First](https://img.shields.io/badge/UI_Style-Dark_Mode_First-10B981.svg)](#ui-aesthetic--design-language)
[![Modules Covered](https://img.shields.io/badge/Modules-6.1_%7C_6.2_%7C_6.3-8B5CF6.svg)](#core-modules-breakdown)

An end-to-end **Multimodal Geospatial GraphRAG (Knowledge Graph Retrieval-Augmented Generation)** mission control pipeline engineered for disaster evacuation, resilient emergency routing, and infrastructure damage reasoning under catastrophic hazards (flash floods, structural bridge failures, seismic collapses, power outages).

Designed with a **Dark Mode First / Tech Forward** aesthetic, this system integrates high-contrast tactical map layers, dynamic knowledge subgraphs with physics simulation, automated agentic decomposition, quantitative IR & RAGAS evaluations, and explainable decision trees.

---

## 📁 Repository Directory Structure

```text
microproject/
├── data/
│   ├── sample_osm_network.json     # Nodes (intersections/hospitals/bridges/shelters) & Edges (roads, capacities, flood depth)
│   ├── disaster_reports.json       # Textual incident dispatch reports with coordinates and damage severity
│   ├── damage_images/              # Synthesized UAV drone thermal/RGB reconnaissance images with tactical HUDs
│   └── generate_images.py          # Script for generating synthetic reconnaissance imagery
├── src/
│   ├── __init__.py
│   ├── graph_store.py              # NetworkX / Neo4j spatial graph store, dynamic edge updates, Dijkstra & K-shortest paths
│   ├── multimodal_retriever.py     # Multimodal vector retriever across dispatch reports and drone damage images
│   ├── agent_workflow.py           # Module 6.1: LangGraph state machine & multi-hop query decomposition
│   ├── explainability.py           # Module 6.3: Explainable reasoning paths, counterfactual analysis & bottleneck attribution
│   └── evaluation.py               # Module 6.2: NDCG@K, MRR, Precision@K, RAGAS Faithfulness & LLM-as-a-Judge benchmark
├── app.py                          # Streamlit Tactical Mission Control Dashboard (Folium map + Pyvis subgraph trace)
├── requirements.txt                # Production dependencies
└── README.md                       # Comprehensive system architecture & usage documentation
```

---

## 🧩 Core Modules Breakdown

### 1. Module 6.1: Agentic Decomposition & Graph Traversal (`src/agent_workflow.py`, `src/graph_store.py`)
- **Query Decomposer**: Breaks unstructured crisis directives into structured subtasks:
  - Spatial entity extraction (Origin, Destination, Facility Type e.g., Trauma Level 1 ICU Hospital or Mega-Shelter).
  - Vehicle threshold constraints (`ambulance`: $\le 20\text{cm}$ flood tolerance; `high_clearance_truck`: $\le 60\text{cm}$; `rescue_boat`: $\le 999\text{cm}$).
- **Dynamic Hazard Synchronization**: Automatically links incident dispatches and UAV damage images to graph edges, updating status (`severely_damaged`, `flooded`, `blocked`, `caution`) and cost multipliers in real time.
- **Resilient Traversal Engine**:
  - Traversal cost function incorporating physical distance, base speed, elevation, water depth, and structural damage:
    $$\text{Cost} = \text{Base Time} \times \left(1 + 8 \times \text{Damage} + 4 \times \text{HazardRisk} + \frac{\text{FloodDepth}}{10} \times 1.5\right)$$
  - Evaluates **K-Shortest Paths** via Yen/Dijkstra algorithms to provide standby contingency corridors.
  - Multi-facility search finds the optimal reachable center based on both travel time and remaining ICU/triage bed capacity.

### 2. Module 6.2: IR & RAGAS Evaluation Pipeline (`src/evaluation.py`)
- **Classical Information Retrieval (IR) Metrics**:
  - $\text{NDCG@K}$ (Normalized Discounted Cumulative Gain): Evaluates ranking quality of retrieved disaster reports against graded ground truth relevance.
  - $\text{MRR}$ (Mean Reciprocal Rank): Measures reciprocal rank of first relevant hazard report.
  - $\text{Precision@K}$ and $\text{Recall@K}$.
- **RAGAS / GraphRAG Domain Metrics**:
  - **Faithfulness**: Validates whether synthesized dispatch directives and travel claims are grounded in retrieved sensor logs and verified graph node attributes.
  - **Answer Relevance**: Measures lexical and semantic alignment between user crisis intent and tactical instructions.
- **LLM-as-a-Judge Pipeline**:
  - Evaluates responses across a 4-dimensional rubric: *Hazard Avoidance (Safety)*, *Topological Feasibility*, *Resource Triage Allocation*, and *Grounding Transparency*.
  - Includes pre-packaged benchmark cases (`Q1` to `Q3`) with automated scorecard generation.

### 3. Module 6.3: Explainable Reasoning Paths (`src/explainability.py`)
- **Decision Provenance**: Step-by-step trace detailing why specific nodes and roads were chosen.
- **Hazard Elimination Log**: Transparently lists disqualified roads with exact incident IDs, water depth metrics, and structural shear percentages.
- **Counterfactual Reasoning ("Why not Path B?")**: Explicitly compares Candidate Path 1 with Candidate Path 2, highlighting delays, detours, and elevated flood risks that led to disqualification.
- **Critical Infrastructure Bottlenecks**: Identifies bridge crossings and vulnerable low-elevation coastal segments subject to aftershocks or secondary storm surges.

---

## 🎨 UI Aesthetic & Design Language: Dark Mode First / Tech Forward

The Streamlit dashboard (`app.py`) is engineered to reflect an authentic **Emergency Operations Center (EOC) / Tactical HUD**:
- **Palette**: Deep Obsidian/Slate background (`#0B0F19`, `#111827`, `#1E293B`) with high-contrast text (`#F8FAFC`, `#CBD5E1`).
- **Tactical Visual Accents**:
  - 🟢 **Emerald Green (`#10B981`)**: Safe, traversable green corridors and fully operational hospitals.
  - 🔴 **Crimson Coral (`#EF4444`)**: Submerged arterial roads, bridge shear cracks, and disqualified edges.
  - 🟡 **Amber Gold (`#F59E0B`)**: Cautious transit zones, traffic congestion, and emergency shelters.
  - 🔷 **Radiant Cyan (`#00F0FF` / `#38BDF8`)**: Active recommended evacuation path, telemetry metrics, and primary buttons.
- **Interactive Layers**:
  - **Folium OpenStreetMap Layer**: High-contrast CartoDB Dark Matter map with color-coded polylines and hospital markers.
  - **Pyvis Physics Subgraph**: Force-directed network diagram showing live topology and node connections.

---

## 🚀 Quickstart Guide

### 1. Installation
Ensure Python 3.10+ is available:
```powershell
pip install -r requirements.txt
```

### 2. Configure Gemini API (Optional for Enhanced LLM Synthesis)
The system automatically uses deterministic fallbacks if no API key is present. To enable Gemini LLM reasoning:
Create or edit `.env` in the root workspace:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Streamlit Mission Control App
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing the Modules via Command Line

Run individual verification scripts or automated benchmarks directly:

```powershell
# 1. Run Benchmark Suite (NDCG@K, MRR, Faithfulness, LLM-as-a-Judge)
python -c "from src.graph_store import CrisisSpatialGraph; from src.multimodal_retriever import MultimodalRetriever; from src.agent_workflow import CrisisResponseAgent; from src.evaluation import run_benchmark_suite; g = CrisisSpatialGraph('data/sample_osm_network.json'); r = MultimodalRetriever(); agent = CrisisResponseAgent(g, r); print(run_benchmark_suite(agent, r))"

# 2. Test Agent Pipeline with Custom Query
python -c "from src.graph_store import CrisisSpatialGraph; from src.multimodal_retriever import MultimodalRetriever; from src.agent_workflow import CrisisResponseAgent; g = CrisisSpatialGraph('data/sample_osm_network.json'); r = MultimodalRetriever(); agent = CrisisResponseAgent(g, r); res = agent.run_pipeline('Find ambulance route from Downtown to General Hospital avoiding flooded streets'); print(res['final_dispatch_order'])"
```

---

## 🗄️ Neo4j Database Integration

The system includes native Cypher statement generation. In the Streamlit sidebar, click **"Generate Cypher Script"** or execute:
```python
from src.graph_store import CrisisSpatialGraph
g = CrisisSpatialGraph("data/sample_osm_network.json")
cypher_script = g.generate_neo4j_cypher()
with open("neo4j_crisis.cypher", "w") as f:
    f.write(cypher_script)
```
Paste into Neo4j Browser or run via `cypher-shell` to instantiate the spatial graph in a Neo4j instance.

