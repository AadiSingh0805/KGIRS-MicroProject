# Aegis CrisisNav | Multimodal Geospatial Crisis Response & Infrastructure GraphRAG

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit UI](https://img.shields.io/badge/UI-Sleek_Mission_Control-00F0FF.svg)](https://streamlit.io/)
[![Dark Mode First](https://img.shields.io/badge/Style-Modern_Glassmorphism-10B981.svg)](#modern-ui-architecture)
[![Evaluation](https://img.shields.io/badge/Benchmarks-IR_%7C_RAGAS_%7C_LLM--Judge-8B5CF6.svg)](#system-verification--evaluation-benchmarks)

An end-to-end **Multimodal Geospatial GraphRAG (Knowledge Graph Retrieval-Augmented Generation)** mission control platform engineered for disaster evacuation, resilient emergency routing, and infrastructure damage reasoning under catastrophic hazards (flash floods, structural bridge failures, seismic collapses, power outages).

Designed with a **Modern, User-Friendly Mission Control UI**, this application gives incident commanders and first responders an intuitive single-window canvas featuring real-time interactive geospatial maps, force-directed graph physics, dynamic hazard avoidance, UAV drone telemetry, and explainable AI decision paths.

---

## Designed For Emergency Dispatchers & First Responders

- **Zero Clutter**: Completely eliminates the restrictive sidebar in favor of an expansive command view with a unified top-level header and inline mission parameters.
- **1-Click Crisis Presets**: Instant launch buttons for high-priority crisis scenarios:
  - **Monsoon Flood Evacuation**: Chembur Naka to Zen Hospital bypassing submerged 75cm flood at Shell Colony.
  - **Amar Mahal Detour**: Heavy emergency convoy routing avoiding a 98% beam shear fracture at Amar Mahal flyover.
  - **Casualty Triage**: Routing from RCF Sports Mega-Shelter to Surana Sethia Hospital prioritizing available ICU beds.
- **Vehicle Profiles**: Real-time water clearance adaptations:
  - **Ambulance**: $\le 20\text{cm}$ flood threshold.
  - **High-Clearance Rescue Truck**: $\le 60\text{cm}$ flood threshold.
  - **Rescue Boat**: Deep-water priority.
  - **Civilian Car**: $\le 15\text{cm}$ flood threshold.
- **Explainable Decision Intelligence**: Transparently explains *why* a route was selected, lists disqualified hazard zones with water depths, and compares candidate corridors.
- **Zero API Key Requirement**: Uses OpenStreetMap with a custom dark tactical shader for geospatial operations, requiring no third-party API keys or subscriptions.

---

## Repository Directory Structure

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
│   ├── agent_workflow.py           # LangGraph state machine & multi-hop query decomposition
│   ├── explainability.py           # Explainable reasoning paths, counterfactual analysis & bottleneck attribution
│   └── evaluation.py               # NDCG@K, MRR, Precision@K, RAGAS Faithfulness & LLM-as-a-Judge benchmark
├── app.py                          # Streamlit Tactical Mission Control Dashboard (Interactive Map, Graphs, Decisions)
├── requirements.txt                # Production dependencies
└── README.md                       # Comprehensive system architecture & usage documentation
```

---

## Core Architectural Pillars

### 1. Resilient Graph Traversal & Dynamic Hazard Synchronization (`src/graph_store.py`, `src/agent_workflow.py`)
- **Query Decomposer**: Extracts spatial endpoints (Origin, Destination) and triage facility intent.
- **Dynamic Hazard Synchronization**: Ingests disaster dispatches and UAV aerial recon, updating edge damage severities and flood depths.
- **Hazard-Penalized Traversal Engine**:
  $$\text{Cost} = \text{Base Time} \times \left(1 + 8 \times \text{Damage} + 4 \times \text{HazardRisk} + \frac{\text{FloodDepth}}{10} \times 1.5\right)$$
- **K-Shortest Paths**: Evaluates standby contingency corridors via Yen/Dijkstra algorithms.
- **Facility Capacity Optimization**: Selects destinations based on both proximity and live emergency ICU/triage bed availability.

### 2. Explainable Decision Intelligence & Attribution (`src/explainability.py`)
- **Decision Provenance**: Transparent step-by-step trace explaining every road selection.
- **Hazard Elimination Log**: Identifies disqualified roads with exact water depths, damage percentages, and sensor citations.
- **Counterfactual Reasoning ("Why not Path B?")**: Directly compares the primary route with candidate alternatives, detailing delay penalties and flood hazards.
- **Critical Infrastructure Bottlenecks**: Highlights single-span bridges and low-elevation coastal zones vulnerable to secondary surges.

### 3. Quantitative IR & RAGAS Evaluation Pipeline (`src/evaluation.py`)
- **Information Retrieval (IR)**: NDCG@3, Mean Reciprocal Rank (MRR), Precision@3, Recall@3.
- **RAGAS Domain Metrics**: Faithfulness (grounded in graph and sensor telemetry) and Answer Relevance.
- **LLM-as-a-Judge Rubric**: Assesses Hazard Avoidance (Safety), Topological Feasibility, Resource Triage Allocation, and Grounding Transparency.

---

## Modern UI Architecture

The Streamlit dashboard (`app.py`) is organized into 5 cohesive operational views:

1. **Tactical Mission & Map**:
   - Telemetry KPI strip (Corridor Status, Transit Time, Distance, AI Confidence, Hazards Bypassed).
   - High-contrast responsive OpenStreetMap centered on Chembur, Mumbai with custom tactical dark shader, glowing route overlay, and hospital capacity markers.
   - Actionable field dispatch directive and turn-by-turn road navigation cards.
   - Backup contingency corridors (K-shortest paths).

2. **Decision Intelligence**:
   - Disqualified hazard elimination log with water depth and structural damage details.
   - Counterfactual route analysis comparing alternative corridors.
   - Critical infrastructure bottleneck monitoring.
   - Multimodal sensor grounding citations and agent state machine trace.

3. **Hazard & UAV Reconnaissance**:
   - Filterable real-time incident reports (Critical, High, Moderate) with operational actions.
   - UAV drone reconnaissance thermal/RGB imagery viewer with damage HUD overlays.

4. **Knowledge Graph Topology**:
   - Force-directed interactive physics network simulation (Pyvis) highlighting the active path and facility statuses.

5. **System Quality & Verification**:
   - One-click benchmark runner with executive KPI scorecards, comparison charts, and detailed test case evaluations.

---

## Quickstart Guide

### 1. Installation
Ensure Python 3.10+ is available:
```powershell
pip install -r requirements.txt
```

### 2. Configure Gemini API (Optional for Enhanced LLM Synthesis)
The system uses deterministic fallbacks if no API key is present. To enable Gemini LLM reasoning:
Create or edit `.env` in the root workspace:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Qdrant vector storage

Report embeddings, image feature vectors, and their metadata are stored in
Qdrant. Retrieval uses Qdrant cosine search rather than in-memory Python lists.
Text embedding providers are explicit and versioned; image retrieval still uses heuristic visual features and filename embeddings.

Copy `.env.example` to `.env` and configure it as needed. By default, no server
or credentials are required: embedded Qdrant persists data under
`data/qdrant_storage/` (ignored by Git). `QDRANT_PATH` can override that location;
relative paths resolve from the repository root. `.env` is loaded for both the
app and command-line use, with existing environment variables taking precedence.

For a Qdrant server or Qdrant Cloud, set `QDRANT_URL` and, when required,
`QDRANT_API_KEY`. A configured server failure is surfaced; storage does not
silently switch to a different backend.

The dashboard indexes reports and available images at startup into collections
named `crisisnav_reports_<dimension>_<profile>` and
`crisisnav_images_<dimension>_<profile>`. The profile signature includes the
provider, pinned model/algorithm, preprocessing version, task types, normalization,
and image feature algorithm. Different providers or dimensions never reuse an
embedding space. Each stored point also carries its embedding profile, which is
checked when read. Collections must use cosine distance and the expected dimension.
Set `QDRANT_COLLECTION_PREFIX` to isolate another dataset.

Indexing synchronizes a complete snapshot per modality: stable UUID point IDs
prevent duplicates, updated records replace old payloads/vectors, and records
absent from the snapshot are removed. Passing an empty list clears that
modality's collection. Use dedicated collections: do not point unrelated
projects or concurrent dataset writers at the same prefix. Retrieval can read
persisted vectors without calling the indexing methods again.

Embedded storage supports one application process per directory; clients are
shared within the process for Streamlit reloads. Store operations are serialized
with a process-wide lock, and the embedded SQLite connection permits access
from successive Streamlit script threads. For multiple processes, use a
Qdrant server.

### Consistent embedding configuration

`EMBEDDING_PROVIDER=hash` is the offline default, even when a Gemini API key is
present for dispatch synthesis. It preserves the deterministic hashing algorithm;
it is not a learned semantic model. Set `EMBEDDING_PROVIDER=gemini` explicitly to
use pinned `gemini-embedding-001`, with `GEMINI_API_KEY` configured.
`EMBEDDING_DIM` defaults to 128 for hash and 768 for Gemini. Explicit constructor
dimensions override this setting. Gemini receives the exact requested output
dimension, uses `RETRIEVAL_DOCUMENT` for indexing and `RETRIEVAL_QUERY` for search,
and vectors are normalized. Unexpected dimensions, non-finite output, and API
errors stop the operation; vectors are never silently cropped, padded, or replaced
by another provider. Image extraction/provider errors also propagate.

Restart the app after changing `.env`. Startup reindexes the current incident
registry and available images into the matching profile collections. Earlier
collections without profile signatures are retained but never queried, because
they may contain mixed providers. Starting the app rebuilds the searchable index
without deleting those legacy collections. Standalone retrievers must call
`index_reports` and `index_images` after changing profiles. API failures during
embedding generation leave the existing snapshot untouched.

Vehicle identifiers, displayed flood limits, and routing limits share one profile
registry: ambulance 20 cm, rescue truck 60 cm, civilian car 15 cm, and the existing
simulated boat profile. Unknown identifiers raise an error, and legacy
`heavy_truck` maps to `high_clearance_truck`. These are configured simulation
thresholds, not independently validated real-world vehicle clearances.

Run storage integration tests with:
```sh
python -m unittest discover -s tests -v
```

### Incident management

Open **Incident management** above the scenario buttons to create or edit an
incident. Select its affected roads, flood depth, structural damage, source,
severity, closure flag, and active/resolved status, then choose **Save incident**.
The current route recalculates and Qdrant is synchronized on the next app run.

Operator edits are saved atomically in `data/incident_reports.json`, initialized
from the bundled reports on first save. This local registry and its lock file
are ignored by Git. Set `INCIDENT_REPORTS_PATH` to use another registry location.
Concurrent stale edits are rejected rather than overwriting
another operator's changes. Restarting the app preserves the local registry.

Routing applies the entire registry, regardless of semantic search ranking.
Resolved reports stay in history but are excluded from report search and image
indexing. Resolving an incident clears its road restrictions only when no other
active report on that road imposes them. Explicit road closures and severe
structural damage block all vehicles; flood depth uses the selected vehicle's
clearance. The old permanent junction exclusions have been removed.

### 3. Run the Streamlit Mission Control App
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Automated Testing & Verification

Run individual verification scripts or automated benchmarks directly:

```powershell
# 1. Run Pipeline Unit Tests
python test_pipeline.py

# 2. Run Quality Benchmark Suite (NDCG@3, MRR, Faithfulness, LLM-as-a-Judge)
python -c "from src.graph_store import CrisisSpatialGraph; from src.multimodal_retriever import MultimodalRetriever; from src.agent_workflow import CrisisResponseAgent; from src.evaluation import run_benchmark_suite; g = CrisisSpatialGraph('data/sample_osm_network.json'); r = MultimodalRetriever(); agent = CrisisResponseAgent(g, r); print(run_benchmark_suite(agent, r))"
```

---

## Neo4j Database Export

The application generates production-ready Cypher statements for graph databases. Click **"Export Cypher"** in the top navigation or execute:
```python
from src.graph_store import CrisisSpatialGraph
g = CrisisSpatialGraph("data/sample_osm_network.json")
cypher_script = g.generate_neo4j_cypher()
with open("neo4j_crisis.cypher", "w") as f:
    f.write(cypher_script)
```
Paste into Neo4j Browser or run via `cypher-shell` to instantiate the spatial graph in a Neo4j instance.

## Demo interface

The dashboard opens on **About** to introduce the project before the demo.
Tabs follow **About → Route planner → Incidents → Evidence → Network → Evaluation**.
**Route planner** keeps the request, vehicle selection, reference-image option,
scenario shortcuts, and map together. **Incidents** holds
the report editor; **Evidence** and **Network** explain the result. **Evaluation**
shows results only after **Run benchmark** is selected. **About** explains the
sample dataset, implementation limits, and SDG alignment. Evidence counts replace
unvalidated confidence percentages, and aerial fallbacks are labelled simulations.
This presentation update does not change routing, retrieval, storage, or scoring.
