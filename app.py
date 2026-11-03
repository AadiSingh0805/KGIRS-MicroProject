"""
Multimodal Geospatial Crisis Response & Infrastructure GraphRAG
Tactical Mission Operations Dashboard
Aesthetic: Dark Mode First / Minimalist Tech Forward / Clean High-Contrast (No Emojis)
Modules Covered: 6.1 (Agentic Traversal), 6.2 (IR & RAGAS Evaluation), 6.3 (Explainability)
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import streamlit as st
import folium
from streamlit_folium import folium_static
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image

# Ensure local imports resolve
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.graph_store import CrisisSpatialGraph
from src.multimodal_retriever import MultimodalRetriever
from src.agent_workflow import CrisisResponseAgent
from src.explainability import ReasoningChainVisualizer
from src.evaluation import run_benchmark_suite, CRISIS_BENCHMARK_DATASET

# Page Configuration
st.set_page_config(
    page_title="Crisis GraphRAG | Tactical Operations",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode First / Minimalist Tech Forward Styling
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0A0D14;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }
    
    /* Header Container */
    .mission-header {
        background-color: #111622;
        border: 1px solid #1E2638;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    
    .hud-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hud-badge {
        background-color: #172554;
        color: #38BDF8;
        border: 1px solid #1E40AF;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-family: monospace;
    }
    
    /* Telemetry Panels */
    .telemetry-card {
        background-color: #111622;
        border: 1px solid #1E2638;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-family: "JetBrains Mono", Consolas, "Fira Code", monospace;
        font-size: 26px;
        font-weight: 700;
        color: #38BDF8;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        margin-bottom: 4px;
    }
    
    /* Status Badges */
    .tag-clear {
        background: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 700;
        font-family: monospace;
    }
    
    .tag-hazard {
        background: #7F1D1D;
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 700;
        font-family: monospace;
    }
    
    .tag-caution {
        background: #78350F;
        color: #FBBF24;
        border: 1px solid #D97706;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 700;
        font-family: monospace;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0F1420;
        padding: 4px;
        border-radius: 6px;
        border: 1px solid #1E2638;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        border-radius: 4px;
        padding: 6px 14px;
        font-size: 13px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E2638 !important;
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    
    /* Code Blocks and Outputs */
    pre, code {
        background-color: #0F1420 !important;
        border: 1px solid #1E2638 !important;
        color: #E2E8F0 !important;
        font-family: "JetBrains Mono", Consolas, monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# Cached System Initialization
@st.cache_resource
def load_crisis_system():
    data_dir = os.path.join(current_dir, "data")
    osm_path = os.path.join(data_dir, "sample_osm_network.json")
    reports_path = os.path.join(data_dir, "disaster_reports.json")
    
    # 1. Spatial Graph
    graph = CrisisSpatialGraph(osm_path)

    # 2. Multimodal Retriever
    retriever = MultimodalRetriever(embedding_dim=128)
    if os.path.exists(reports_path):
        with open(reports_path, "r", encoding="utf-8") as f:
            reports_data = json.load(f)
        retriever.index_reports(reports_data)

        # Index Recon Images
        img_meta = []
        for rep in reports_data:
            img_rel = rep.get("image_ref")
            if img_rel:
                full_path = os.path.join(current_dir, img_rel)
                if not os.path.exists(full_path):
                    full_path = os.path.join(data_dir, "damage_images", os.path.basename(img_rel))
                if os.path.exists(full_path):
                    img_meta.append({
                        "image_path": full_path,
                        "associated_report": rep.get("report_id"),
                        "hazard_type": rep.get("incident_type")
                    })
        retriever.index_images(img_meta)
    else:
        reports_data = []

    # 3. Agent & Visualizer
    agent = CrisisResponseAgent(graph, retriever)
    visualizer = ReasoningChainVisualizer(graph)

    return graph, retriever, agent, visualizer, reports_data

graph_store, retriever, agent, visualizer, disaster_reports = load_crisis_system()


# Header Section
st.markdown("""
<div class="mission-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="hud-title">
            <span>CRISIS GRAPHRAG</span>
            <span class="hud-badge">TACTICAL SYSTEM</span>
        </div>
        <div style="font-family: monospace; font-size: 12px; color: #38BDF8;">
            TELEMETRY: ACTIVE | NODES: 15 | PROTOCOL: 6.1-6.3
        </div>
    </div>
    <div style="color: #64748B; font-size: 13px; margin-top: 6px;">
        Multimodal Geospatial GraphRAG for Resilient Evacuation, Dynamic Damage Reasoning, and Explainable Routing.
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown("### SYSTEM PARAMETERS")
    vehicle_type = st.selectbox(
        "Response Vehicle Profile",
        options=["ambulance", "high_clearance_truck", "rescue_boat", "civilian_car"],
        index=0,
        help="Adjusts water clearance thresholds and speed penalties."
    )
    avoid_hazards = st.checkbox("Strict Hazard Avoidance", value=True)

    st.markdown("---")
    st.markdown("### ACTIVE INCIDENTS")
    for rep in disaster_reports[:4]:
        tag = "tag-hazard" if rep.get("severity") == "CRITICAL" else ("tag-caution" if rep.get("severity") == "HIGH" else "tag-clear")
        st.markdown(f"""
        <div style="background: #111622; border-left: 2px solid #EF4444; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #F1F5F9;">{rep.get('report_id')}</span>
                <span class="{tag}">{rep.get('severity')}</span>
            </div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">{rep.get('incident_type')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### EXPORTS")
    if st.button("Export Neo4j Cypher Script", use_container_width=True):
        cypher = graph_store.generate_neo4j_cypher()
        st.download_button("Download Cypher File", cypher, file_name="neo4j_crisis.cypher", mime="text/plain", use_container_width=True)


# Session State Management
if "current_state" not in st.session_state:
    st.session_state.current_state = agent.run_pipeline(
        query="Find fastest safe ambulance route from Downtown to nearest hospital avoiding flooded streets",
        vehicle_type="ambulance",
        avoid_hazards=True
    )


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Tactical Map & Topology",
    "Agentic Traversal (Module 6.1)",
    "Reasoning Paths (Module 6.3)",
    "Evaluation Benchmarks (Module 6.2)"
])


# ==============================================================================
# TAB 1: TACTICAL MAP & SUBGRAPH
# ==============================================================================
with tab1:
    st.markdown("##### Real-Time Geospatial Map & Knowledge Subgraph")
    
    col_map, col_pyvis = st.columns([3, 2])
    
    current_state = st.session_state.current_state
    primary_route = current_state.get("primary_route")
    route_nodes = primary_route.get("path", []) if primary_route else []
    route_node_set = set(route_nodes)

    with col_map:
        center_lat, center_lon = 37.7800, -122.4200
        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="CartoDB dark_matter",
            control_scale=True
        )

        # Plot Edges
        drawn_edges = set()
        for u, v, data in graph_store.graph.edges(data=True):
            eid = data.get("id", "")
            if eid.endswith("_REV") or eid in drawn_edges:
                continue
            drawn_edges.add(eid)

            coords = data.get("geometry", [])
            status = data.get("status", "open")
            dmg = data.get("damage_severity", 0.0)
            fld = data.get("flood_depth_cm", 0.0)
            name = data.get("name", eid)

            if status in ["severely_damaged", "blocked"]:
                color = "#EF4444"
                weight = 3
                dash_array = "5, 5"
            elif status == "flooded":
                color = "#3B82F6"
                weight = 4
                dash_array = "4, 4"
            elif status == "caution":
                color = "#F59E0B"
                weight = 3
                dash_array = None
            else:
                color = "#10B981"
                weight = 2
                dash_array = None

            if coords:
                popup_html = f"""
                <div style='font-family: monospace; font-size: 11px; color: #111;'>
                    <b>{name}</b> ({eid})<br/>
                    Status: {status.upper()}<br/>
                    Distance: {data.get('distance_km')} km<br/>
                    Damage: {int(dmg*100)}% | Water: {fld} cm
                </div>
                """
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=weight,
                    opacity=0.75,
                    dash_array=dash_array,
                    popup=folium.Popup(popup_html, max_width=240)
                ).add_to(fmap)

        # Highlight Active Route
        if primary_route and primary_route.get("segments"):
            for seg in primary_route["segments"]:
                seg_coords = seg.get("geometry", [])
                if seg_coords:
                    folium.PolyLine(
                        locations=seg_coords,
                        color="#38BDF8",
                        weight=6,
                        opacity=0.95,
                        tooltip=f"Selected Corridor: {seg.get('road_name')}"
                    ).add_to(fmap)

        # Plot Nodes with validation
        for nid, ndata in graph_store.graph.nodes(data=True):
            lat = ndata.get("lat")
            lon = ndata.get("lon")
            if lat is None or lon is None:
                continue

            ntype = ndata.get("type", "intersection")
            name = ndata.get("name", nid)
            status = ndata.get("operational_status", "active")
            avail = ndata.get("available_capacity", 0)
            total = ndata.get("capacity", 0)

            if ntype == "hospital":
                icon_color = "green" if status == "active" else "red"
                icon_name = "medkit"
            elif ntype == "shelter":
                icon_color = "orange"
                icon_name = "home"
            elif ntype == "bridge":
                icon_color = "purple" if status == "active" else "red"
                icon_name = "road"
            elif ntype == "supply_depot":
                icon_color = "blue"
                icon_name = "th"
            else:
                icon_color = "gray"
                icon_name = "circle"

            popup_text = f"""
            <div style='font-family: monospace; font-size: 11px; color: #111;'>
                <b>{name}</b> [{nid}]<br/>
                Type: {ntype.upper()}<br/>
                Status: {status}<br/>
                Capacity: {avail}/{total}
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=240),
                tooltip=f"{name} ({ntype})",
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa")
            ).add_to(fmap)

        folium_static(fmap, width=650, height=520)

    with col_pyvis:
        st.markdown("**Knowledge Subgraph Topology (Physics Sim)**")
        pyvis_html = graph_store.export_pyvis_network(highlight_path=route_nodes, height="520px")
        components.html(pyvis_html, height=530, scrolling=False)


# ==============================================================================
# TAB 2: AGENTIC WORKFLOW (MODULE 6.1)
# ==============================================================================
with tab2:
    st.markdown("##### Module 6.1: Agentic Decomposition & Graph Traversal")
    
    preset_cols = st.columns(3)
    preset_query = None
    with preset_cols[0]:
        if st.button("Preset: Flood Bypass", use_container_width=True):
            preset_query = "Find fastest safe ambulance route from Downtown to nearest hospital avoiding flooded streets"
    with preset_cols[1]:
        if st.button("Preset: Bridge Detour", use_container_width=True):
            preset_query = "Route heavy emergency convoy from Downtown to North Bay Span inspection depot avoiding fractured bridge"
    with preset_cols[2]:
        if st.button("Preset: Casualty Triage", use_container_width=True):
            preset_query = "Evacuate casualties from Civic Auditorium Shelter to General Hospital with ICU trauma beds"

    user_query = st.text_input(
        "Incident Dispatch Directive:",
        value=preset_query if preset_query else st.session_state.current_state.get("user_query", ""),
        help="Enter route requests with vehicle, location, or facility parameters."
    )

    recon_col1, recon_col2 = st.columns([2, 1])
    with recon_col1:
        uploaded_img = st.file_uploader("Optional Reconnaissance Image Upload", type=["png", "jpg", "jpeg"])
    with recon_col2:
        preset_img = st.selectbox(
            "Or Select Verified Drone Telemetry:",
            options=["None", "flash_flood_embarcadero.png", "bridge_pier_crack.png", "hospital_submerged_basement.png", "landslide_downed_lines.png", "clear_corridor_highway.png"]
        )

    if st.button("Execute Traversal Engine", type="primary", use_container_width=True):
        img_target = None
        if uploaded_img:
            temp_path = os.path.join(current_dir, "data", "damage_images", "temp_upload.png")
            with open(temp_path, "wb") as f_out:
                f_out.write(uploaded_img.read())
            img_target = temp_path
        elif preset_img != "None":
            img_target = os.path.join(current_dir, "data", "damage_images", preset_img)

        with st.spinner("Processing decomposition and dynamic graph traversal..."):
            new_state = agent.run_pipeline(
                query=user_query,
                image_path=img_target,
                vehicle_type=vehicle_type,
                avoid_hazards=avoid_hazards
            )
            st.session_state.current_state = new_state
            st.rerun()

    state = st.session_state.current_state
    route = state.get("primary_route")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-label">Route Status</div>
            <div class="metric-value" style="color: {'#34D399' if route and route.get('status') == 'CLEAR' else '#F87171'};">
                {route.get('status') if route else 'HALTED'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-label">Estimated Transit Time</div>
            <div class="metric-value">{route.get('estimated_travel_time_min', 0.0) if route else 0.0} <span style="font-size:14px;color:#64748B;">min</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-label">Corridor Distance</div>
            <div class="metric-value">{route.get('total_distance_km', 0.0) if route else 0.0} <span style="font-size:14px;color:#64748B;">km</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="metric-label">Confidence Score</div>
            <div class="metric-value">{int(state.get('confidence_score', 0.0)*100)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("State Machine Trajectory (Subtasks & State Transitions)", expanded=True):
        for step in state.get("reasoning_trace", []):
            st.markdown(f"""
            <div style="background:#111622; border-left:3px solid #38BDF8; padding:8px 12px; margin-bottom:8px; border-radius:4px;">
                <b style="color:#38BDF8; font-family: monospace;">{step.get('step')}</b> <span style="color:#64748B; font-size:11px;">[{step.get('module')}]</span>
                <div style="color:#94A3B8; font-size:12px; margin-top:4px;">{step.get('details')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Tactical Dispatch Directive")
    st.markdown(state.get("final_dispatch_order", "No active dispatch directive."))

    alt_routes = state.get("alternative_routes", [])
    if alt_routes:
        st.markdown("### Candidate Contingency Corridors (K-Shortest Paths)")
        alt_data = []
        for a in alt_routes:
            alt_data.append({
                "Rank": f"Corridor #{a['rank']}",
                "Transit Path": " -> ".join(a["path"]),
                "ETA (min)": a["eta_min"],
                "Distance (km)": a["distance_km"],
                "Cost Score": a["cost_score"]
            })
        st.dataframe(pd.DataFrame(alt_data), use_container_width=True)


# ==============================================================================
# TAB 3: EXPLAINABILITY (MODULE 6.3)
# ==============================================================================
with tab3:
    st.markdown("##### Module 6.3: Explainable Reasoning Paths & Decision Attribution")
    
    state = st.session_state.current_state
    explanation = visualizer.explain_route_decision(
        primary_route=state.get("primary_route"),
        alternative_routes=state.get("alternative_routes", []),
        retrieved_incidents=state.get("retrieved_incidents", []),
        target_facility_info=state.get("target_facility_info")
    )

    exp_col1, exp_col2 = st.columns([3, 2])

    with exp_col1:
        st.markdown(visualizer.format_markdown_explanation(explanation))

    with exp_col2:
        st.markdown("### Multimodal Sensor Reconnaissance")
        retrieved_imgs = state.get("retrieved_images", [])
        if retrieved_imgs:
            for im in retrieved_imgs:
                p = im.get("path")
                if os.path.exists(p):
                    st.image(p, caption=f"Evidence: {im.get('image_id')} ({im.get('hazard_type')})", use_container_width=True)
        else:
            sample_img = os.path.join(current_dir, "data", "damage_images", "flash_flood_embarcadero.png")
            if os.path.exists(sample_img):
                st.image(sample_img, caption="UAV Recon Telemetry: Flood Impact Area", use_container_width=True)


# ==============================================================================
# TAB 4: EVALUATION BENCHMARKS (MODULE 6.2)
# ==============================================================================
with tab4:
    st.markdown("##### Module 6.2: Information Retrieval & RAGAS / LLM-as-a-Judge Evaluation")
    st.markdown("Automated evaluation across ranking metrics (NDCG@K, MRR, Precision@K), RAGAS faithfulness, and LLM-as-a-Judge safety rubric.")

    if st.button("Run Full Crisis Evaluation Benchmark (Q1 - Q3)", use_container_width=True):
        with st.spinner("Benchmarking IR and RAGAS metrics..."):
            bench_results = run_benchmark_suite(agent, retriever)
            st.session_state.benchmark_results = bench_results
            st.success("Benchmark Execution Complete.")

    if "benchmark_results" in st.session_state:
        results = st.session_state.benchmark_results
        
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        with bm1:
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="metric-label">NDCG@3</div>
                <div class="metric-value">{results['mean_ndcg@3']}</div>
            </div>
            """, unsafe_allow_html=True)
        with bm2:
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="metric-label">MRR</div>
                <div class="metric-value">{results['mrr']}</div>
            </div>
            """, unsafe_allow_html=True)
        with bm3:
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="metric-label">Precision@3</div>
                <div class="metric-value">{results['mean_precision@3']}</div>
            </div>
            """, unsafe_allow_html=True)
        with bm4:
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="metric-label">Faithfulness</div>
                <div class="metric-value">{results['mean_faithfulness']}</div>
            </div>
            """, unsafe_allow_html=True)
        with bm5:
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="metric-label">Judge Score</div>
                <div class="metric-value">{results['mean_judge_score']} <span style="font-size:14px;color:#64748B;">/10</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Benchmark Evaluation Scorecard per Query")
        df_bench = pd.DataFrame(results["detailed_results"])
        st.dataframe(df_bench[["query_id", "query", "ndcg@3", "mrr", "precision@3", "recall@3", "faithfulness", "answer_relevance", "judge_composite", "judge_critique"]], use_container_width=True)

        chart_data = pd.DataFrame({
            "Metric": ["NDCG@3", "MRR", "Precision@3", "Recall@3", "Faithfulness", "Answer Relevance"],
            "Score": [
                results['mean_ndcg@3'],
                results['mrr'],
                results['mean_precision@3'],
                results['mean_recall@3'],
                results['mean_faithfulness'],
                results['mean_answer_relevance']
            ]
        }).set_index("Metric")
        st.bar_chart(chart_data)
    else:
        st.info("Execute 'Run Full Crisis Evaluation Benchmark' to run the evaluation suite.")
