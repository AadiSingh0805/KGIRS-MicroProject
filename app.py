"""
Aegis CrisisNav | Resilient Geospatial GraphRAG
Tactical Emergency Operations & Resilient Evacuation Navigation (Mumbai Chembur Corridor)
A clean, modern, user-friendly mission control app for crisis dispatchers and first responders.
"""

import os
import sys
import re
import json
import base64
import hashlib
import copy
from typing import Dict, Any, List, Optional
import streamlit as st
import folium
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image

# Ensure local imports resolve
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(current_dir, ".env"))
except Exception:
    pass

import importlib
import src.graph_store as gs_mod
import src.agent_workflow as aw_mod
import src.explainability as exp_mod
import src.multimodal_retriever as mr_mod
try:
    importlib.reload(gs_mod)
    importlib.reload(aw_mod)
    importlib.reload(exp_mod)
    importlib.reload(mr_mod)
except Exception:
    pass

from src.graph_store import CrisisSpatialGraph
from src.multimodal_retriever import MultimodalRetriever
from src.agent_workflow import CrisisResponseAgent
from src.explainability import ReasoningChainVisualizer
from src.evaluation import run_benchmark_suite, CRISIS_BENCHMARK_DATASET
from src.incident_repository import IncidentRepository
from src.incident_ui import render_incident_manager
from src.vehicle_profiles import VEHICLE_PROFILES
from src.presentation import render_about
from src.dispatch_brief import build_dispatch_brief

# ==============================================================================
# STREAMLIT CONFIGURATION - CLEAN FULL-CANVAS LAYOUT (NO SIDEBAR, NO EMOJIS)
# ==============================================================================
st.set_page_config(
    page_title="Aegis CrisisNav | Mumbai Chembur Emergency GraphRAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Tech-Forward Sleek Design System
st.markdown("""
<style>
    /* 1. Base App & Canvas */
    .stApp {
        background-color: #080C14;
        background-image: radial-gradient(at 50% 0%, #0F172A 0%, #080C14 75%);
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }

    /* Completely hide Streamlit sidebar and extra controls for clean microproject look */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.0rem !important;
        max-width: 1480px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Prominent Route Planner PDF export action */
    [data-testid="stDownloadButton"] button[kind="secondary"] {
        min-height: 44px;
        border: 1px solid #F59E0B !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #92400E 0%, #B45309 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        box-shadow: 0 0 18px rgba(245, 158, 11, 0.28);
    }

    /* 2. Top Executive Header Bar */
    .brand-icon {
        background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%);
        color: #FFFFFF;
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.5px;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
    }

    .brand-title {
        font-size: 23px;
        font-weight: 750;
        letter-spacing: -0.4px;
        color: #F8FAFC;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 11px;
        font-weight: 500;
        color: #94A3B8;
        letter-spacing: 0.2px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.4px;
        font-family: "JetBrains Mono", monospace;
    }

    .pill-green {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .pill-cyan {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .pill-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* 3. Sleek Metric KPI Cards */
    .metric-card {
        background: #0E1526;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }

    .metric-caption {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
    }

    .metric-number {
        font-family: "JetBrains Mono", Consolas, monospace;
        font-size: 24px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 2px;
        line-height: 1.1;
    }

    .metric-unit {
        font-size: 12px;
        font-weight: 500;
        color: #64748B;
        margin-left: 2px;
    }

    .metric-sub {
        font-size: 10.5px;
        color: #64748B;
        margin-top: 2px;
    }

    /* 4. Modern Cards & Surfaces */
    .glass-panel {
        background: #0E1526;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .panel-heading {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #CBD5E1;
        margin-bottom: 8px;
    }

    /* 5. Hazard & Corridor Tag Badges */
    .badge-clear {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        font-family: monospace;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        font-family: monospace;
    }
    .badge-caution {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid #D97706;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        font-family: monospace;
    }

    /* 6. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #0B111E;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 12.5px;
        font-weight: 600;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }

    /* 7. Input fields & buttons */
    .stTextInput > div > div > input {
        background-color: #0E1526 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
        font-size: 13px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38BDF8 !important;
    }
    .stSelectbox > div > div {
        background-color: #0E1526 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
    }

    /* 8. Sleek turn-by-turn road list */
    .turn-item {
        background: #0B111E;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* 9. Styled Field Directive Box */
    .directive-card {
        background: #0B111E;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .directive-header {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #38BDF8;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .directive-route {
        font-size: 13.5px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .directive-stats {
        display: flex;
        gap: 12px;
        font-size: 11px;
        color: #94A3B8;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 8px;
    }
    .directive-item {
        font-size: 11.5px;
        color: #CBD5E1;
        line-height: 1.4;
        margin-bottom: 4px;
        display: flex;
        gap: 6px;
    }

    /* Demo presentation: readable surfaces, restrained emphasis, compact navigation. */
    .planner-title { font-size: 25px !important; letter-spacing: -0.7px; margin: 0 !important; padding: 0 !important; }
    [data-testid="stVerticalBlock"] { gap: .65rem; }
    .metric-card { box-shadow: none !important; padding: 13px 15px !important; min-height: 82px; border-radius: 10px; }
    .metric-number { font-size: 26px !important; line-height: 1.25; }
    .metric-caption { font-size: 11px !important; color: #A8B5C9 !important; letter-spacing: 0.5px; }
    .metric-sub { font-size: 11px; color: #A0AEC0; }
    .panel-heading { font-size: 13px; text-transform: none; letter-spacing: 0; color: #E2E8F0; margin-top: 5px; }
    .directive-card, .glass-panel { box-shadow: none !important; }
    .directive-route { font-size: 15px; line-height: 1.5; }
    .directive-item { font-size: 12px; line-height: 1.6; }
    .stTabs [data-baseweb="tab-list"] { margin-top: 8px; margin-bottom: 8px; gap: 3px; overflow-x: auto; }
    .stTabs [data-baseweb="tab"] { padding: 7px 17px; white-space: nowrap; }
    div[data-testid="stForm"] { border-color: #263449; border-radius: 12px; padding: 14px 18px; }
    button[data-testid="stBaseButton-primaryFormSubmit"], button[data-testid="stBaseButton-primary"] { background: #0EA5E9 !important; border-color: #0EA5E9 !important; color: #07111F !important; font-weight: 700; }
    button:focus-visible, input:focus-visible { outline: 2px solid #7DD3FC !important; outline-offset: 2px; }
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        .brand-title { font-size: 20px; }
        .metric-number { font-size: 22px !important; }
        .stTabs [data-baseweb="tab"] { padding: 6px 10px; }
    }
</style>
""", unsafe_allow_html=True)
# ==============================================================================
# AEGIS CRISISNAV - PREMIUM UI OVERRIDES
# ==============================================================================
st.markdown("""
<style>
/* Smooth interface */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Main canvas */
.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(14,165,233,0.12), transparent 40%),
        radial-gradient(ellipse at 85% 10%, rgba(37,99,235,0.08), transparent 35%),
        #060B14 !important;
}

/* Main content spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1600px !important;
}

/* Premium glass cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #101C2D, #0B1422);
    border: 1px solid rgba(56,189,248,0.20);
    padding: 18px 20px;
    border-radius: 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,0.55);
    transform: translateY(-3px);
    box-shadow: 0 10px 35px rgba(14,165,233,0.12);
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.7px;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: #E0F2FE !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}

/* Section headings */
h1, h2, h3 {
    color: #F1F5F9 !important;
    letter-spacing: -0.5px;
}

h1 {
    font-weight: 850 !important;
}

h2 {
    font-weight: 750 !important;
}

h3 {
    font-weight: 700 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #94A3B8 !important;
    font-weight: 650 !important;
    padding: 12px 22px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom-color: #38BDF8 !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: #38BDF8 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    min-height: 42px;
    font-weight: 650 !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    background: linear-gradient(135deg, #0F2238, #102A43) !important;
    color: #E0F2FE !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.18);
    transform: translateY(-1px);
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284C7, #2563EB) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 5px 20px rgba(37,99,235,0.25);
}

/* Inputs and select boxes */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: #0B1524 !important;
    color: #E2E8F0 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] > div {
    background-color: #0B1524 !important;
    border-color: #23364D !important;
    border-radius: 10px !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background: rgba(15,28,45,0.65);
    border: 1px solid #1E334A;
    border-radius: 12px;
    overflow: hidden;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #1E334A;
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(56,189,248,0.2);
}

/* Horizontal separators */
hr {
    border-color: rgba(56,189,248,0.15) !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #060B14;
}

::-webkit-scrollbar-thumb {
    background: #1E3A5F;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #0284C7;
}

/* Subtle animation */
@keyframes aegisFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.block-container {
    animation: aegisFadeIn 0.45s ease-out;
}

/* ==========================================
   AEGIS TELEMETRY CARDS - PREMIUM REDESIGN
   ========================================== */

.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 142px;
    padding: 22px 20px;
    background: linear-gradient(145deg, #101D30 0%, #0A1220 100%);
    border: 1px solid rgba(56, 189, 248, 0.20);
    border-radius: 16px;
    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.25),
        inset 0 1px 0 rgba(255,255,255,0.025);
    transition: all 0.28s ease;
}

.metric-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 2px;
    width: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        #38BDF8,
        #6366F1,
        transparent
    );
    opacity: 0.75;
}

.metric-card::after {
    content: "";
    position: absolute;
    width: 110px;
    height: 110px;
    right: -40px;
    bottom: -45px;
    background: radial-gradient(
        circle,
        rgba(56,189,248,0.12),
        transparent 70%
    );
    pointer-events: none;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56,189,248,0.55);
    box-shadow:
        0 14px 35px rgba(0,0,0,0.35),
        0 0 22px rgba(56,189,248,0.07);
}

.metric-caption {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.metric-number {
    font-size: 28px;
    font-weight: 850;
    letter-spacing: -0.8px;
    line-height: 1.15;
    color: #F1F5F9;
    text-shadow: 0 0 18px rgba(56,189,248,0.08);
    white-space: nowrap;
}

.metric-unit {
    font-size: 13px;
    font-weight: 600;
    color: #64748B;
    margin-left: 6px;
    letter-spacing: 0;
}

.metric-sub {
    color: #64748B;
    font-size: 11px;
    font-weight: 500;
    margin-top: 12px;
    letter-spacing: 0.15px;
}

/* ==========================================
   AEGIS FINAL VISUAL POLISH
   ========================================== */

/* Fix download buttons: PDF and graph export */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #0F2238, #102A43) !important;
    color: #E0F2FE !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    font-weight: 700 !important;
    min-height: 42px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #075985, #1D4ED8) !important;
    border-color: #38BDF8 !important;
    box-shadow: 0 0 22px rgba(56,189,248,0.22) !important;
    transform: translateY(-2px);
}

/* Improve all navigation tab contrast */
button[data-baseweb="tab"] {
    color: #A8B9CF !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: #38BDF8 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    background: rgba(14,165,233,0.08) !important;
    border-radius: 8px 8px 0 0 !important;
}

/* Better labels and help text contrast */
label, [data-testid="stWidgetLabel"] {
    color: #CBD5E1 !important;
}

/* Scenario buttons: refined hover */
.stButton > button {
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #12304D, #153B59) !important;
}

/* Better panel heading */
.panel-heading {
    color: #38BDF8 !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
}

/* Subtle glow around map and reconnaissance panels */
iframe {
    border-radius: 14px !important;
}

/* More readable secondary text */
.metric-sub {
    color: #94A3B8 !important;
}

.metric-unit {
    color: #94A3B8 !important;
}

/* Reduce harsh white borders on focused controls */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 1px rgba(56,189,248,0.25) !important;
}

</style>
""", unsafe_allow_html=True)


incident_reports_path = os.getenv("INCIDENT_REPORTS_PATH", "").strip() or os.path.join(current_dir, "data", "incident_reports.json")
if not os.path.isabs(incident_reports_path):
    incident_reports_path = os.path.join(current_dir, incident_reports_path)

# ==============================================================================
# SYSTEM INITIALIZATION & CACHING (Chembur, Mumbai Network Mesh)
# ==============================================================================
def get_data_fingerprint() -> str:
    """Computes a hash of the OSM network and disaster reports data files to auto-invalidate stale caches."""
    h = hashlib.sha256()
    for fname in ["sample_osm_network.json", "disaster_reports.json"]:
        p = os.path.join(current_dir, "data", fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    if os.path.exists(incident_reports_path):
        with open(incident_reports_path, "rb") as f:
            h.update(f.read())
    for name in ("EMBEDDING_PROVIDER", "EMBEDDING_DIM"):
        h.update(os.getenv(name, "").encode())
    return h.hexdigest()[:16]

@st.cache_resource(show_spinner=False)
def load_crisis_system(mesh_fingerprint: str):
    data_dir = os.path.join(current_dir, "data")
    osm_path = os.path.join(data_dir, "sample_osm_network.json")
    reports_path = incident_reports_path
    if not os.path.exists(reports_path):
        reports_path = os.path.join(data_dir, "disaster_reports.json")

    # 1. Spatial Graph Engine
    graph = CrisisSpatialGraph(osm_path)

    # 2. Multimodal Telemetry Retriever & Agent
    api_key = os.getenv("GEMINI_API_KEY")
    retriever = MultimodalRetriever(api_key=api_key)
    if os.path.exists(reports_path):
        with open(reports_path, "r", encoding="utf-8") as f:
            reports_data = json.load(f)
        retriever.index_reports(reports_data)

        # Index Recon Images
        img_meta = []
        for rep in reports_data:
            if rep.get("incident_status", "active") == "resolved":
                continue
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

    graph.apply_incident_reports(reports_data)

    # 3. Agent & Reasoning Engine
    agent = CrisisResponseAgent(graph, retriever, api_key=api_key)
    try:
        visualizer = ReasoningChainVisualizer(graph, api_key=api_key)
    except TypeError:
        visualizer = ReasoningChainVisualizer(graph)

    return graph, retriever, agent, visualizer, reports_data

mesh_fingerprint = get_data_fingerprint()
graph_store, retriever, agent, visualizer, disaster_reports = load_crisis_system(mesh_fingerprint)
# Cached Qdrant is shared safely; each session gets its own mutable route graph.
graph_store = copy.deepcopy(graph_store)
agent = CrisisResponseAgent(graph_store, retriever, api_key=os.getenv("GEMINI_API_KEY"))
visualizer = ReasoningChainVisualizer(graph_store, api_key=os.getenv("GEMINI_API_KEY"))
if st.session_state.get("mesh_fingerprint") != mesh_fingerprint:
    st.session_state.pop("current_state", None)
    st.session_state.pop("benchmark_results", None)
    st.session_state.mesh_fingerprint = mesh_fingerprint

# ==============================================================================
# SESSION STATE AUTO-SANITIZER (Purge stale legacy session memory)
# ==============================================================================
need_reset = False
if "current_state" in st.session_state:
    cs = st.session_state.current_state
    origin_check = cs.get("extracted_origin", "")
    route_check = cs.get("primary_route") or {}
    path_nodes = route_check.get("path", [])
    if (
        origin_check not in graph_store.graph.nodes
        or "DOWNTOWN" in str(origin_check)
        or any("DOWNTOWN" in str(n) or "VANNESS" in str(n) or "STJUDE" in str(n) for n in path_nodes)
    ):
        need_reset = True

if "directive_input" in st.session_state:
    if "downtown" in st.session_state.directive_input.lower() or "van ness" in st.session_state.directive_input.lower():
        need_reset = True

if need_reset:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ==============================================================================
# SESSION STATE INITIALIZATION (Default: Chembur Naka to Zen Hospital)
# ==============================================================================
def get_node_satellite_image(node_id: str) -> str:
    """Maps canonical Chembur node IDs to high-resolution orbital satellite images."""
    mapping = {
        "N_CAMPUS_VESIT": "sat_vesit_campus.png",
        "N_HOSP_ZEN": "sat_zen_hospital.png",
        "N_HOSP_SURANA": "sat_surana_hospital.png",
        "N_CHEMBUR_NAKA": "sat_chembur_naka.png",
        "N_BRIDGE_AMAR_MAHAL": "sat_tilak_nagar.png",
        "N_SHELTER_RCF": "sat_rcf_shelter.png",
        "N_HOSP_INLAKS": "sat_inlaks_hospital.png",
        "N_INT_SHELL_COLONY": "sat_chembur_naka.png",
        "N_INT_TILAK_NAGAR": "sat_tilak_nagar.png",
        "N_DEPOT_MANKHURD": "sat_mankhurd_depot.png",
        "N_SHELTER_ACRES": "sat_surana_hospital.png",
        "N_INT_SUMAN_NAGAR": "sat_disaster_amar_mahal_collapse.png",
        "N_INT_VN_PURAV": "sat_disaster_shell_colony_flood.png",
        "N_INT_SINDHI_SOC": "sat_vesit_campus.png",
        "N_BRIDGE_FREEWAY": "sat_rcf_shelter.png",
        "N_INT_GOVANDI": "sat_mankhurd_depot.png"
    }
    return mapping.get(node_id, "sat_chembur_naka.png")

if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = "flood_bypass"

if "directive_input" not in st.session_state:
    st.session_state.directive_input = "Find fastest safe ambulance route from Chembur Naka to Zen Multi Speciality Hospital avoiding flooded streets"

if "vehicle_type" not in st.session_state:
    st.session_state.vehicle_type = "ambulance"

if "avoid_hazards" not in st.session_state:
    st.session_state.avoid_hazards = True

if "selected_image_name" not in st.session_state:
    st.session_state.selected_image_name = "sat_shell_colony.png"

if "current_state" not in st.session_state:
    img_path = os.path.join(current_dir, "data", "damage_images", st.session_state.selected_image_name)
    st.session_state.current_state = agent.run_pipeline(
        query=st.session_state.directive_input,
        image_path=img_path if os.path.exists(img_path) else None,
        vehicle_type=st.session_state.vehicle_type,
        avoid_hazards=st.session_state.avoid_hazards
    )


# ==============================================================================
# TOP EXECUTIVE HEADER BAR (Clean, Modern, Chembur Mission Control)
# ==============================================================================
top_col_left, top_col_right = st.columns([1.35, 1.65])

with top_col_left:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 14px; padding: 2px 0;">
        <svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.55)); flex-shrink: 0;">
            <defs>
                <linearGradient id="shieldGrad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#0284C7"/>
                    <stop offset="50%" stop-color="#0EA5E9"/>
                    <stop offset="100%" stop-color="#38BDF8"/>
                </linearGradient>
                <linearGradient id="coreGrad" x1="16" y1="16" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#38BDF8"/>
                    <stop offset="100%" stop-color="#0284C7"/>
                </linearGradient>
            </defs>
            <path d="M24 3L42 9V22C42 33.2 34.3 43.1 24 46C13.7 43.1 6 33.2 6 22V9L24 3Z" fill="#0A1120" stroke="url(#shieldGrad)" stroke-width="2.2" stroke-linejoin="round"/>
            <circle cx="24" cy="24" r="13" stroke="rgba(56, 189, 248, 0.28)" stroke-width="1.2" stroke-dasharray="2 3"/>
            <line x1="24" y1="11" x2="24" y2="37" stroke="rgba(56, 189, 248, 0.45)" stroke-width="1.2"/>
            <line x1="11" y1="24" x2="37" y2="24" stroke="rgba(56, 189, 248, 0.45)" stroke-width="1.2"/>
            <line x1="24" y1="24" x2="33" y2="15" stroke="#38BDF8" stroke-width="1.8"/>
            <line x1="24" y1="24" x2="15" y2="31" stroke="#FBBF24" stroke-width="1.8"/>
            <line x1="24" y1="24" x2="31" y2="31" stroke="#34D399" stroke-width="1.8"/>
            <circle cx="33" cy="15" r="2.8" fill="#38BDF8" stroke="#FFFFFF" stroke-width="1"/>
            <circle cx="15" cy="31" r="2.8" fill="#FBBF24" stroke="#FFFFFF" stroke-width="1"/>
            <circle cx="31" cy="31" r="2.8" fill="#34D399" stroke="#FFFFFF" stroke-width="1"/>
            <circle cx="24" cy="11" r="2" fill="#38BDF8"/>
            <circle cx="24" cy="24" r="5" fill="url(#coreGrad)" stroke="#FFFFFF" stroke-width="1.5"/>
            <circle cx="24" cy="24" r="2" fill="#FFFFFF"/>
        </svg>
        <div>
            <div class="brand-title">Aegis CrisisNav</div>
            <div class="brand-subtitle">Emergency routing · GraphRAG demonstration</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col_right:
    pills_box, btn_box = st.columns([3.4, 1.1])
    with pills_box:
        st.markdown(f"""
        <div style="display: flex; gap: 8px; justify-content: flex-end; align-items: center; height: 100%; padding-top: 5px;">
            <span class="status-pill pill-green"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #34D399; box-shadow: 0 0 6px #34D399;"></span>DEMO DATA</span>
            <span class="status-pill pill-cyan"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #38BDF8;"></span>{len(graph_store.graph.nodes)} LOCATIONS</span>
            <span class="status-pill pill-amber"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #FBBF24;"></span>{sum(1 for r in disaster_reports if r.get("incident_status", "active") != "resolved" and (r.get("flood_depth_cm", 0) > 0 or r.get("damage_severity", 0) > 0.25 or r.get("road_closed", False)))} ACTIVE HAZARDS</span>
        </div>
        """, unsafe_allow_html=True)
    with btn_box:
        if st.button("Reload data", help="Reload the current dataset and recalculate the route", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# Navigation separates planning, data editing, and project context.
tab_about, tab_mission, tab_incidents, tab_decision, tab_graph, tab_eval = st.tabs([
    "About", "Route planner", "Incidents", "Evidence", "Network", "Evaluation"
], default="Route planner" if st.query_params.get("view") == "planner" else "About")

with tab_incidents:
    st.subheader("Incident registry")
    st.caption("Add a road restriction, update its measurements, or resolve it to recalculate routes.")
    incident_roads = {
        data["id"]: {"name": data["name"], "nodes": (u, v)}
        for u, v, data in graph_store.graph.edges(data=True)
        if not data["id"].endswith("_REV")
    }
    render_incident_manager(IncidentRepository(
        incident_reports_path,
        os.path.join(current_dir, "data", "disaster_reports.json"), incident_roads,
    ), expanded=True)

with tab_about:
    render_about(graph_store, disaster_reports, VEHICLE_PROFILES, current_dir)

with tab_mission:
    st.markdown('<h2 class="planner-title">Plan an emergency route</h2>', unsafe_allow_html=True)
    latest_report_time = max((r.get("timestamp", "") for r in disaster_reports), default="Not available")
    st.caption(f"Chembur, Mumbai · Demo dataset · Latest report: {latest_report_time}")
    # 2. Custom Directive Input & Parameter Bar (Form-backed for guaranteed input synchronization)
    with st.form("dispatch_directive_form", clear_on_submit=False):
        cmd_col1 = st.container()
        cmd_col2, cmd_col3, cmd_col4 = st.columns([1.2, 1.6, 1.0], vertical_alignment="bottom")

        with cmd_col1:
            query_val = st.text_input(
                "Where should the vehicle go?",
                value=st.session_state.directive_input,
                placeholder="e.g. Find fastest safe ambulance route from VESIT to Zen Multi Speciality Hospital avoiding flooded streets..."
            )

        with cmd_col2:
            vehicle_opts = list(VEHICLE_PROFILES)
            v_idx = vehicle_opts.index(st.session_state.vehicle_type) if st.session_state.vehicle_type in vehicle_opts else 0
            v_val = st.selectbox(
                "Vehicle",
                options=vehicle_opts,
                index=v_idx,
                format_func=lambda x: VEHICLE_PROFILES[x]["label"]
            )

        with cmd_col3:
            sat_imgs = [
                "None",
                "sat_disaster_shell_colony_flood.png",
                "sat_disaster_amar_mahal_collapse.png",
                "sat_inlaks_hospital.png",
                "sat_vesit_campus.png",
                "sat_surana_hospital.png",
                "sat_zen_hospital.png",
                "sat_chembur_naka.png",
                "sat_rcf_shelter.png",
                "sat_tilak_nagar.png",
                "sat_mankhurd_depot.png"
            ]
            img_idx = sat_imgs.index(st.session_state.selected_image_name) if st.session_state.selected_image_name in sat_imgs else 0
            img_val = st.selectbox(
                "Reference image (optional)",
                options=sat_imgs,
                index=img_idx,
                format_func=lambda x: {
                    "None": "No image",
                    "sat_disaster_shell_colony_flood.png": "VN Purav Marg Central Arterial (75cm Flood Hazard)",
                    "sat_disaster_amar_mahal_collapse.png": "Suman Nagar Flyover (Expansion Shear Hazard)",
                    "sat_inlaks_hospital.png": "Inlaks Hospital (Postal Colony)",
                    "sat_vesit_campus.png": "VESIT Campus (Collector Colony Safe Zone)",
                    "sat_surana_hospital.png": "Surana Sethia Hospital (Trauma ICU)",
                    "sat_zen_hospital.png": "Zen Multi Speciality Hospital",
                    "sat_chembur_naka.png": "Chembur Naka Transit Hub",
                    "sat_rcf_shelter.png": "RCF Sports Complex Mega-Shelter",
                    "sat_tilak_nagar.png": "Tilak Nagar Station Junction",
                    "sat_mankhurd_depot.png": "Mankhurd Supply Depot"
                }.get(x, x),
                help="Choose a bundled reference image for the existing image-retrieval workflow"
            )

        with cmd_col4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            calc_submitted = st.form_submit_button("Find route", type="primary", use_container_width=True)

    if calc_submitted:
        st.session_state.directive_input = query_val
        st.session_state.vehicle_type = v_val
        st.session_state.selected_image_name = img_val

        img_target = None
        if st.session_state.selected_image_name != "None":
            img_target = os.path.join(current_dir, "data", "damage_images", st.session_state.selected_image_name)

        with st.spinner("Finding a route using the current incident reports..."):
            new_state = agent.run_pipeline(
                query=query_val,
                image_path=img_target,
                vehicle_type=v_val,
                avoid_hazards=st.session_state.avoid_hazards
            )
            st.session_state.current_state = new_state

            # Synchronize tactical reconnaissance hub with chosen telemetry or destination
            image_to_node = {
                "sat_zen_hospital.png": "N_HOSP_ZEN",
                "sat_surana_hospital.png": "N_HOSP_SURANA",
                "sat_inlaks_hospital.png": "N_HOSP_INLAKS",
                "sat_disaster_shell_colony_flood.png": "N_INT_VN_PURAV",
                "sat_disaster_amar_mahal_collapse.png": "N_INT_SUMAN_NAGAR",
                "sat_vesit_campus.png": "N_CAMPUS_VESIT",
                "sat_rcf_shelter.png": "N_SHELTER_RCF",
                "sat_chembur_naka.png": "N_CHEMBUR_NAKA",
                "sat_tilak_nagar.png": "N_INT_TILAK_NAGAR",
                "sat_mankhurd_depot.png": "N_DEPOT_MANKHURD"
            }
            target_recon = image_to_node.get(img_val) or new_state.get("extracted_destination", "N_HOSP_ZEN")
            st.session_state.recon_marker_selector = target_recon
            st.session_state.selected_recon_node = target_recon
            st.rerun()

    # 1. Preset Crisis Scenarios (Quick 1-Click Launch for Mumbai Chembur)
    st.caption("Try a demo scenario")
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns([1.0, 1.0, 1.0, 1.0, 1.15, 0.75])

    with p_col1:
        if st.button("Flood detour", use_container_width=True, help="Calculate the current Chembur Naka to Zen Hospital route"):
            st.session_state.active_scenario = "flood_bypass"
            st.session_state.directive_input = "Find fastest safe ambulance route from Chembur Naka to Zen Multi Speciality Hospital avoiding flooded streets"
            st.session_state.vehicle_type = "ambulance"
            st.session_state.avoid_hazards = True
            st.session_state.selected_image_name = "sat_disaster_shell_colony_flood.png"
            img_p = os.path.join(current_dir, "data", "damage_images", "sat_disaster_shell_colony_flood.png")
            st.session_state.current_state = agent.run_pipeline(
                query=st.session_state.directive_input,
                image_path=img_p if os.path.exists(img_p) else None,
                vehicle_type="ambulance",
                avoid_hazards=True
            )
            st.session_state.recon_marker_selector = "N_INT_VN_PURAV"
            st.session_state.selected_recon_node = "N_INT_VN_PURAV"
            st.rerun()

    with p_col2:
        if st.button("Bridge closure", use_container_width=True, help="Calculate the current convoy route to Surana Hospital"):
            st.session_state.active_scenario = "bridge_detour"
            st.session_state.directive_input = "Route heavy emergency convoy from Chembur Naka to Surana Sethia Hospital avoiding fractured Suman Nagar flyover"
            st.session_state.vehicle_type = "high_clearance_truck"
            st.session_state.avoid_hazards = True
            st.session_state.selected_image_name = "sat_disaster_amar_mahal_collapse.png"
            img_p = os.path.join(current_dir, "data", "damage_images", "sat_disaster_amar_mahal_collapse.png")
            st.session_state.current_state = agent.run_pipeline(
                query=st.session_state.directive_input,
                image_path=img_p if os.path.exists(img_p) else None,
                vehicle_type="high_clearance_truck",
                avoid_hazards=True
            )
            st.session_state.recon_marker_selector = "N_INT_SUMAN_NAGAR"
            st.session_state.selected_recon_node = "N_INT_SUMAN_NAGAR"
            st.rerun()

    with p_col3:
        if st.button("Campus transfer", use_container_width=True, help="Evacuation transit from VESIT Campus (Collector Colony) to Zen Hospital"):
            st.session_state.active_scenario = "vesit_evac"
            st.session_state.directive_input = "Find fastest safe ambulance route from Vivekanand Education Society's Institute of Technology (VESIT) to Zen Multi Speciality Hospital avoiding flooded streets"
            st.session_state.vehicle_type = "ambulance"
            st.session_state.avoid_hazards = True
            st.session_state.selected_image_name = "sat_vesit_campus.png"
            img_p = os.path.join(current_dir, "data", "damage_images", "sat_vesit_campus.png")
            st.session_state.current_state = agent.run_pipeline(
                query=st.session_state.directive_input,
                image_path=img_p if os.path.exists(img_p) else None,
                vehicle_type="ambulance",
                avoid_hazards=True
            )
            st.rerun()

    with p_col4:
        if st.button("Hospital transfer", use_container_width=True, help="Evacuate casualties from RCF Mega-Shelter to Surana Hospital ICU"):
            st.session_state.active_scenario = "casualty_triage"
            st.session_state.directive_input = "Evacuate casualties from RCF Sports Complex Shelter to Surana Sethia Hospital with ICU trauma beds"
            st.session_state.vehicle_type = "ambulance"
            st.session_state.avoid_hazards = True
            st.session_state.selected_image_name = "sat_surana_hospital.png"
            img_p = os.path.join(current_dir, "data", "damage_images", "sat_surana_hospital.png")
            st.session_state.current_state = agent.run_pipeline(
                query=st.session_state.directive_input,
                image_path=img_p if os.path.exists(img_p) else None,
                vehicle_type="ambulance",
                avoid_hazards=True
            )
            st.rerun()

    with p_col5:
        if st.button("VN Purav transfer", use_container_width=True, help="Calculate the existing evacuation scenario from VN Purav to Zen Hospital using current restrictions"):
            st.session_state.active_scenario = "air_force_medevac"
            st.session_state.directive_input = "Critical emergency evacuation from submerged VN Purav intersection to Zen Hospital rooftop helipad"
            st.session_state.vehicle_type = "ambulance"
            st.session_state.avoid_hazards = True
            st.session_state.selected_image_name = "sat_disaster_shell_colony_flood.png"
            img_p = os.path.join(current_dir, "data", "damage_images", "sat_disaster_shell_colony_flood.png")
            st.session_state.current_state = agent.run_pipeline(
                query=st.session_state.directive_input,
                image_path=img_p if os.path.exists(img_p) else None,
                vehicle_type="ambulance",
                avoid_hazards=True
            )
            st.session_state.recon_marker_selector = "N_INT_VN_PURAV"
            st.session_state.selected_recon_node = "N_INT_VN_PURAV"
            st.rerun()

    with p_col6:
        cypher_code = graph_store.generate_neo4j_cypher()
        st.download_button(
            label="Export graph",
            data=cypher_code,
            file_name="neo4j_chembur_graph.cypher",
            mime="text/plain",
            use_container_width=True,
            help="Download complete Neo4j Cypher script to replicate the spatial graph in Neo4j"
        )

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

current_state = st.session_state.current_state
primary_route = current_state.get("primary_route")
route_nodes = primary_route.get("path", []) if primary_route else []
route_node_set = set(route_nodes)


# ==============================================================================
# TAB 1: ACTIVE MISSION & MAP (Command Center View)
# ==============================================================================
with tab_mission:
    # 1. Real-Time Telemetry KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    r_status = primary_route.get("status", "HALTED") if primary_route else "HALTED"
    r_time = primary_route.get("estimated_travel_time_min", 0.0) if primary_route else 0.0
    r_dist = primary_route.get("total_distance_km", 0.0) if primary_route else 0.0
    evidence_count = len(current_state.get("retrieved_incidents", []))
    bypassed_count = sum(
        1 for u, v, data in graph_store.graph.edges(data=True)
        if not data["id"].endswith("_REV")
        and graph_store.calculate_edge_weight(u, v, data, current_state.get("vehicle_type", "ambulance")) == float("inf")
    )
    is_air_evac = bool(primary_route and primary_route.get("is_aerial_evacuation"))

    origin_id = current_state.get("extracted_origin", "N_CHEMBUR_NAKA")
    dest_id = current_state.get("extracted_destination", "N_HOSP_ZEN")
    origin_label = graph_store.graph.nodes.get(origin_id, {}).get("name", origin_id)
    dest_label = graph_store.graph.nodes.get(dest_id, {}).get("name", dest_id)

    with kpi1:
        if is_air_evac:
            status_color = "#F59E0B"
            status_text = "NO GROUND ROUTE"
            status_sub = "Simulated aerial alternative"
        else:
            status_color = "#34D399" if r_status == "CLEAR" else ("#FBBF24" if r_status == "CAUTION" else "#F87171")
            status_text = "ROUTE FOUND" if r_status == "CLEAR" else r_status
            status_sub = "Within configured limits" if r_status == "CLEAR" else "Review route restrictions"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Route status</div>
            <div class="metric-number" style="color: {status_color}; font-size: 20px;">{status_text}</div>
            <div class="metric-sub">{status_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        time_sub = "Simulated flight estimate" if is_air_evac else "Includes hazard penalties"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Route estimate</div>
            <div class="metric-number" style="color: #38BDF8;">{r_time}<span class="metric-unit">min</span></div>
            <div class="metric-sub">{time_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        dist_sub = "Direct flight vector" if is_air_evac else "Physical road length"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Distance</div>
            <div class="metric-number" style="color: #F8FAFC;">{r_dist}<span class="metric-unit">km</span></div>
            <div class="metric-sub">{dist_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Retrieved evidence</div>
            <div class="metric-number" style="color: #A78BFA;">{evidence_count}<span class="metric-unit">reports</span></div>
            <div class="metric-sub">Supporting search results</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Restricted roads</div>
            <div class="metric-number" style="color: #F87171;">{bypassed_count}<span class="metric-unit">roads</span></div>
            <div class="metric-sub">Across the demo network</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    if is_air_evac or not primary_route:
        st.warning(f"No usable ground route from {origin_label} to {dest_label} under the current constraints. Any aerial corridor shown is a simulated alternative; no dispatch has been sent.")

    # 2. Interactive Map & Tactical Satellite Surveillance Hub (Side-by-Side)
    map_col, recon_col = st.columns([2.2, 1.0])

    with map_col:
        map_heading_col, dispatch_button_col = st.columns([2.0, 1.0], vertical_alignment="center")
        with map_heading_col:
            st.markdown('<div class="panel-heading">Route map · Chembur, Mumbai</div>', unsafe_allow_html=True)
        with dispatch_button_col:
            dispatch_pdf = build_dispatch_brief(
                current_state,
                graph_store,
                {
                    "vehicle_type": current_state.get("vehicle_type", st.session_state.vehicle_type),
                    "avoid_hazards": st.session_state.avoid_hazards,
                    "query": st.session_state.directive_input,
                },
            )
            st.download_button(
                label="Download PDF report",
                data=dispatch_pdf,
                file_name="aegis_crisisnav_dispatch_brief.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download the current route map, directions, settings, and supporting evidence as a PDF",
                key="download_dispatch_pdf_report",
            )

        # Center on Chembur, Mumbai
        center_lat, center_lon = 19.0575, 72.8988

        # Leaflet Folium Map with Dual Satellite and Street Basemaps
        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles=None,
            control_scale=True
        )
        # Explicitly lock viewport bounds to Chembur, Mumbai metropolitan network
        fmap.fit_bounds([[19.042, 72.882], [19.072, 72.935]])

        # 1. Esri World Imagery (Authentic High-Resolution Orbital Satellite Photography)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite basemap (Esri)",
            show=False,
            overlay=False,
            control=True
        ).add_to(fmap)

        # 2. OpenStreetMap (Street Network)
        folium.TileLayer(
            tiles="OpenStreetMap",
            attr="OpenStreetMap",
            name="Street Network (OSM)",
            overlay=False,
            control=True
        ).add_to(fmap)

        folium.LayerControl(position="topleft").add_to(fmap)

        # Plot all road edges with hazard color-coding
        drawn_edges = set()
        for u, v, data in graph_store.graph.edges(data=True):
            eid = data.get("id", "")
            if eid.endswith("_REV") or eid in drawn_edges:
                continue
            drawn_edges.add(eid)

            coords = data.get("geometry", [])
            # Filter strictly to Mumbai Chembur coordinate bounding box
            if coords and not (18.5 <= coords[0][0] <= 19.5 and 72.5 <= coords[0][1] <= 73.5):
                continue

            status = data.get("status", "open")
            dmg = data.get("damage_severity", 0.0)
            fld = data.get("flood_depth_cm", 0.0)
            name = data.get("name", eid)

            if status in ["severely_damaged", "blocked"]:
                color = "#EF4444"
                weight = 3.5
                dash_array = "6, 6"
            elif status == "flooded":
                color = "#38BDF8"
                weight = 4
                dash_array = "5, 5"
            elif status == "caution":
                color = "#F59E0B"
                weight = 3
                dash_array = None
            else:
                color = "#10B981"
                weight = 2.5
                dash_array = None

            if coords:
                popup_html = f"""
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; min-width: 170px; color: #E2E8F0; padding: 2px;">
                    <div style="font-size: 12px; font-weight: 700; color: #FFFFFF; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 4px; margin-bottom: 5px;">
                        {name}
                    </div>
                    <div style="font-size: 10px; color: #64748B; margin-bottom: 2px;">SEGMENT: <span style="color: #38BDF8; font-family: monospace;">{eid}</span></div>
                    <div style="font-size: 10px; color: #64748B; margin-bottom: 2px;">STATUS: <span style="color: {'#34D399' if status=='open' else ('#38BDF8' if status=='flooded' else '#EF4444')}; font-weight: 700; text-transform: uppercase;">{status}</span></div>
                    <div style="font-size: 10px; color: #64748B; margin-bottom: 2px;">LENGTH: <span style="color: #E2E8F0;">{data.get('distance_km')} km</span></div>
                    <div style="font-size: 10px; color: #64748B;">DAMAGE: <span style="color: {'#EF4444' if dmg>0.3 else '#E2E8F0'}; font-weight: 600;">{int(dmg*100)}%</span> | WATER: <span style="color: {'#38BDF8' if fld>15 else '#E2E8F0'}; font-weight: 600;">{fld} cm</span></div>
                </div>
                """
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=weight,
                    opacity=0.75,
                    dash_array=dash_array,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(fmap)

        # Overlay Glowing Active Safe Route or Tactical Aerial Corridor
        if primary_route and primary_route.get("segments"):
            for seg in primary_route["segments"]:
                seg_coords = seg.get("geometry", [])
                if seg_coords and (18.5 <= seg_coords[0][0] <= 19.5 and 72.5 <= seg_coords[0][1] <= 73.5):
                    if is_air_evac:
                        # Halo line for glowing flight vector
                        folium.PolyLine(
                            locations=seg_coords,
                            color="#F59E0B",
                            weight=9.0,
                            opacity=0.35,
                            dash_array="10, 10"
                        ).add_to(fmap)
                        # Core tactical flight line
                        folium.PolyLine(
                            locations=seg_coords,
                            color="#FBBF24",
                            weight=4.5,
                            opacity=0.95,
                            dash_array="10, 10",
                            tooltip=f"Simulated aerial corridor: {seg.get('road_name')} ({seg.get('distance_km')} km)"
                        ).add_to(fmap)

                        # Add midway airborne helicopter beacon
                        if len(seg_coords) >= 2:
                            mid_lat = (seg_coords[0][0] + seg_coords[-1][0]) / 2.0
                            mid_lon = (seg_coords[0][1] + seg_coords[-1][1]) / 2.0
                            air_icon_html = """
                            <div style="width: 34px; height: 34px; border-radius: 50%; background: radial-gradient(circle, #F59E0B 0%, #B45309 70%, #78350F 100%); border: 2px solid #FEF08A; box-shadow: 0 0 16px #F59E0B, 0 0 30px rgba(245, 158, 11, 0.7); display: flex; align-items: center; justify-content: center; cursor: pointer; animation: pulse-flood 1.5s infinite;">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 2v4M4 6h16M7 10h10a4 4 0 0 1 4 4v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2a4 4 0 0 1 4-4zM8 18v2M16 18v2M5 20h14"></path>
                                </svg>
                            </div>
                            """
                            folium.Marker(
                                location=[mid_lat, mid_lon],
                                popup=folium.Popup(f"<div style='font-family: sans-serif; color: #E2E8F0; padding: 3px;'><b style='color: #F59E0B;'>Simulated aerial corridor</b><br><span style='font-size: 11px;'>Flight ETA: {r_time} min | Air Dist: {r_dist} km</span><br><span style='font-size: 10px; color: #94A3B8;'>Asset: Mil Mi-17 V5 Tactical Medevac</span></div>", max_width=240),
                                tooltip="Simulated aerial alternative",
                                icon=folium.DivIcon(
                                    html=air_icon_html,
                                    icon_size=(34, 34),
                                    icon_anchor=(17, 17),
                                    popup_anchor=(0, -17),
                                    class_name="tactical-div-icon"
                                )
                            ).add_to(fmap)
                    else:
                        folium.PolyLine(
                            locations=seg_coords,
                            color="#00F0FF",
                            weight=7.0,
                            opacity=0.95,
                            tooltip=f"Selected Safe Corridor: {seg.get('road_name')} ({seg.get('distance_km')} km)"
                        ).add_to(fmap)

        # Inject Custom CSS for Glowing Tactical Beacons and Dark Glass Popups
        fmap_css = """
        <style>
        @keyframes pulse-origin {
            0% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.9), 0 0 14px #00F0FF; }
            70% { box-shadow: 0 0 0 14px rgba(0, 240, 255, 0), 0 0 14px #00F0FF; }
            100% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0), 0 0 14px #00F0FF; }
        }
        @keyframes pulse-dest {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.95), 0 0 16px #10B981; }
            70% { box-shadow: 0 0 0 16px rgba(16, 185, 129, 0), 0 0 16px #10B981; }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0), 0 0 16px #10B981; }
        }
        @keyframes pulse-flood {
            0% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.95), 0 0 14px #06B6D4; transform: scale(1); }
            50% { transform: scale(1.08); }
            70% { box-shadow: 0 0 0 16px rgba(6, 182, 212, 0), 0 0 14px #06B6D4; }
            100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0), 0 0 14px #06B6D4; transform: scale(1); }
        }
        @keyframes pulse-hazard {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.95), 0 0 16px #EF4444; transform: scale(1); }
            50% { transform: scale(1.08); }
            70% { box-shadow: 0 0 0 18px rgba(239, 68, 68, 0), 0 0 16px #EF4444; }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0), 0 0 16px #EF4444; transform: scale(1); }
        }
        .tactical-div-icon {
            background: transparent !important;
            border: none !important;
        }
        .leaflet-popup-content-wrapper {
            background: #0B111E !important;
            color: #E2E8F0 !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.85) !important;
            border-radius: 8px !important;
            padding: 0 !important;
        }
        .leaflet-popup-tip {
            background: #0B111E !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
        }
        .leaflet-popup-content {
            margin: 10px 14px !important;
            line-height: 1.4 !important;
        }
        </style>
        """
        fmap.get_root().html.add_child(folium.Element(fmap_css))

        # Resolve active extraction origin and designated target from routing telemetry
        active_origin = (
            (primary_route.get("origin") if primary_route else None)
            or (primary_route.get("source") if primary_route else None)
            or current_state.get("extracted_origin")
            or "N_CHEMBUR_NAKA"
        )
        active_target = (
            (primary_route.get("target") if primary_route else None)
            or (primary_route.get("destination") if primary_route else None)
            or current_state.get("extracted_destination")
            or "N_HOSP_ZEN"
        )

        # Plot Nodes with Vibrant, High-Tech Tactical Glowing Beacons
        for nid, ndata in graph_store.graph.nodes(data=True):
            lat = ndata.get("lat")
            lon = ndata.get("lon")
            if lat is None or lon is None:
                continue
            # Ensure only coordinates in the Mumbai Chembur metropolitan region are plotted
            if not (18.5 <= lat <= 19.5 and 72.5 <= lon <= 73.5):
                continue

            ntype = ndata.get("type", "intersection")
            name = ndata.get("name", nid)
            status = ndata.get("operational_status", "active")
            avail = ndata.get("available_capacity", 0)
            total = ndata.get("capacity", 0)
            icu = ndata.get("icu_beds", 0)

            # Node role identification - dynamically anchored to active query source and destination
            is_origin = bool(active_origin and nid == active_origin)
            is_target = bool(active_target and nid == active_target)

            hazards = graph_store.node_hazards(nid)
            is_flood_hazard = hazards["flood_depth_cm"] > 20
            is_structural_hazard = hazards["damage_severity"] >= 0.85

            # Select high-tech icon design and dimensions
            if is_origin:
                size = 34
                icon_html = """
                <div style="width: 32px; height: 32px; border-radius: 50%; background: radial-gradient(circle, #00F0FF 0%, #0284C7 70%, #0369A1 100%); border: 2px solid #FFFFFF; box-shadow: 0 0 14px #00F0FF, 0 0 28px rgba(0, 240, 255, 0.6); display: flex; align-items: center; justify-content: center; animation: pulse-origin 2s infinite; cursor: pointer;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="22" y1="12" x2="18" y2="12"></line>
                        <line x1="6" y1="12" x2="2" y2="12"></line>
                        <line x1="12" y1="6" x2="12" y2="2"></line>
                        <line x1="12" y1="22" x2="12" y2="18"></line>
                    </svg>
                </div>
                """
            elif is_target:
                size = 36
                icon_html = """
                <div style="width: 34px; height: 34px; border-radius: 50%; background: radial-gradient(circle, #34D399 0%, #10B981 60%, #047857 100%); border: 2px solid #FFFFFF; box-shadow: 0 0 16px #10B981, 0 0 30px rgba(16, 185, 129, 0.7); display: flex; align-items: center; justify-content: center; animation: pulse-dest 1.8s infinite; cursor: pointer;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"></path>
                        <line x1="4" y1="22" x2="4" y2="15"></line>
                    </svg>
                </div>
                """
            elif is_flood_hazard:
                size = 34
                icon_html = """
                <div style="width: 32px; height: 32px; border-radius: 50%; background: radial-gradient(circle, #38BDF8 0%, #0284C7 65%, #0C4A6E 100%); border: 2px solid #67E8F9; box-shadow: 0 0 14px #00F0FF, 0 0 26px rgba(56, 189, 248, 0.7); display: flex; align-items: center; justify-content: center; animation: pulse-flood 1.6s infinite; cursor: pointer;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="#FFFFFF">
                        <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
                    </svg>
                </div>
                """
            elif is_structural_hazard:
                size = 34
                icon_html = """
                <div style="width: 32px; height: 32px; border-radius: 50%; background: radial-gradient(circle, #F87171 0%, #EF4444 65%, #7F1D1D 100%); border: 2px solid #FCA5A5; box-shadow: 0 0 14px #EF4444, 0 0 28px rgba(239, 68, 68, 0.75); display: flex; align-items: center; justify-content: center; animation: pulse-hazard 1.4s infinite; cursor: pointer;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                </div>
                """
            elif ntype == "hospital" and not is_target:
                size = 28
                icon_html = """
                <div style="width: 26px; height: 26px; border-radius: 6px; background: #064E3B; border: 1.5px solid #10B981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.55); display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="#34D399">
                        <path d="M19 10.5h-5.5V5c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v5.5H5c-.83 0-1.5.67-1.5 1.5s.67 1.5 1.5 1.5h5.5V19c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5v-5.5H19c.83 0 1.5-.67 1.5-1.5s-.67-1.5-1.5-1.5z"/>
                    </svg>
                </div>
                """
            elif ntype == "shelter" and not is_target:
                size = 26
                icon_html = """
                <div style="width: 24px; height: 24px; border-radius: 6px; background: #451A03; border: 1.5px solid #F59E0B; box-shadow: 0 0 8px rgba(245, 158, 11, 0.5); display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                </div>
                """
            elif ntype == "campus" and not is_origin:
                size = 26
                icon_html = """
                <div style="width: 24px; height: 24px; border-radius: 6px; background: #3B0764; border: 1.5px solid #A855F7; box-shadow: 0 0 8px rgba(168, 85, 247, 0.55); display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#C084FC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 10v6M2 10l10-5 10 5-10 5z"></path>
                        <path d="M6 12v5c3 3 9 3 12 0v-5"></path>
                    </svg>
                </div>
                """
            elif (ntype == "transit_hub" or nid == "N_CHEMBUR_NAKA") and not is_origin:
                size = 28
                icon_html = """
                <div style="width: 26px; height: 26px; border-radius: 50%; background: #082F49; border: 1.5px solid #0284C7; box-shadow: 0 0 8px rgba(2, 132, 199, 0.6); display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.5">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M12 2v3m0 14v3M2 12h3m14 0h3"></path>
                    </svg>
                </div>
                """
            elif ntype == "bridge":
                size = 26
                icon_html = """
                <div style="width: 24px; height: 24px; border-radius: 6px; background: #1E1B4B; border: 1.5px solid #818CF8; box-shadow: 0 0 9px rgba(129, 140, 248, 0.65); display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C7D2FE" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 19V9a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v10"></path>
                        <line x1="2" y1="19" x2="22" y2="19"></line>
                        <line x1="8" y1="7" x2="8" y2="19"></line>
                        <line x1="16" y1="7" x2="16" y2="19"></line>
                    </svg>
                </div>
                """
            elif ntype == "supply_depot":
                size = 24
                icon_html = """
                <div style="width: 22px; height: 22px; border-radius: 5px; background: #172554; border: 1.5px solid #3B82F6; box-shadow: 0 0 7px rgba(59, 130, 246, 0.5); display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <div style="width: 8px; height: 8px; background: #60A5FA; border-radius: 2px;"></div>
                </div>
                """
            else:
                # Open Road Intersections (including Shell Colony, Tilak Nagar, Govandi, Sindhi Society)
                size = 18
                icon_html = """
                <div style="width: 18px; height: 18px; border-radius: 50%; background: #0B111E; border: 2px solid #F97316; box-shadow: 0 0 8px rgba(249, 115, 22, 0.7); display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <div style="width: 6px; height: 6px; border-radius: 50%; background: #FDBA74; box-shadow: 0 0 4px #F97316;"></div>
                </div>
                """

            popup_text = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; min-width: 190px; color: #E2E8F0; padding: 2px;">
                <div style="font-size: 12.5px; font-weight: 800; color: #FFFFFF; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 5px; margin-bottom: 6px;">
                    {name}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                    <span style="color: #64748B; font-weight: 600;">NODE ID:</span>
                    <span style="color: #38BDF8; font-family: monospace; font-weight: 700;">{nid}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                    <span style="color: #64748B; font-weight: 600;">CLASSIFICATION:</span>
                    <span style="color: #A78BFA; font-weight: 700; text-transform: uppercase;">{ntype}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                    <span style="color: #64748B; font-weight: 600;">STATUS:</span>
                    <span style="color: {'#34D399' if status == 'active' else '#F87171'}; font-weight: 800; text-transform: uppercase;">{status}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                    <span style="color: #64748B; font-weight: 600;">ELEVATION:</span>
                    <span style="color: #E2E8F0; font-weight: 600;">{ndata.get('elevation_m', 0)} m</span>
                </div>
                {f'''<div style="display: flex; justify-content: space-between; font-size: 10px; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255,255,255,0.1);">
                    <span style="color: #64748B; font-weight: 600;">TRIAGE INTAKE:</span>
                    <span style="color: #34D399; font-weight: 700;">{avail}/{total} beds (ICU: {icu})</span>
                </div>''' if ntype == 'hospital' else ''}
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=260),
                tooltip=f"{name} ({ntype.upper()})",
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(size, size),
                    icon_anchor=(size // 2, size // 2),
                    popup_anchor=(0, -(size // 2)),
                    class_name="tactical-div-icon"
                )
            ).add_to(fmap)

        # In-Map Tactical HUD Legend (Positioned top-right inside map canvas)
        in_map_legend_html = """
        <div id="tactical-map-legend" style="position:absolute;bottom:28px;left:12px;z-index:1000;background:rgba(11,17,30,.94);border:1px solid #334155;border-radius:8px;padding:10px 12px;color:#E2E8F0;font:11px/1.8 system-ui;max-width:185px">
            <b>Map key</b><br>
            <span style="color:#00F0FF">━</span> Selected route<br>
            <span style="color:#10B981">━</span> Open road<br>
            <span style="color:#38BDF8">┄</span> Flood report<br>
            <span style="color:#EF4444">┄</span> Damage / closure<br>
            <span style="color:#F59E0B">┄</span> Simulated air route
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(in_map_legend_html))

        # Responsive rendering of the map
        map_html = fmap.get_root().render()
        components.html(map_html, height=550, scrolling=False)

    with recon_col:
        st.markdown('<div class="panel-heading">Location details</div>', unsafe_allow_html=True)

        # All available nodes for inspection
        all_nodes_list = list(graph_store.graph.nodes(data=True))
        all_node_ids = [nid for nid, _ in all_nodes_list]

        # Default to active route destination or target
        dest_node_id = current_state.get("extracted_destination", "N_HOSP_ZEN")
        orig_node_id = current_state.get("extracted_origin", "N_CHEMBUR_NAKA")

        if "recon_marker_selector" not in st.session_state or st.session_state.recon_marker_selector not in all_node_ids:
            st.session_state.recon_marker_selector = dest_node_id if dest_node_id in all_node_ids else all_node_ids[0]

        def get_recon_label(nid):
            nd = graph_store.graph.nodes.get(nid, {})
            name = nd.get("name", nid)
            ntype = nd.get("type", "").upper()
            if graph_store.node_hazards(nid)["flood_depth_cm"] > 20:
                return f"{name} [FLOOD ON ADJACENT ROAD]"
            elif graph_store.node_hazards(nid)["damage_severity"] >= 0.85:
                return f"{name} [STRUCTURAL HAZARD ON ADJACENT ROAD]"
            elif nid == "N_CAMPUS_VESIT":
                return f"{name} [ACADEMIC CAMPUS]"
            elif nd.get("type") == "hospital":
                return f"{name} [HOSPITAL]"
            elif nd.get("type") == "shelter":
                return f"{name} [SHELTER]"
            return f"{name} ({ntype})"

        default_idx = all_node_ids.index(st.session_state.recon_marker_selector) if st.session_state.recon_marker_selector in all_node_ids else 0

        chosen_marker = st.selectbox(
            "Inspect a location",
            options=all_node_ids,
            format_func=get_recon_label,
            key="recon_marker_selector",
            help="Select a location to view stored coordinates, capacity, and reference imagery"
        )
        st.session_state.selected_recon_node = chosen_marker

        sel_node_id = chosen_marker
        sel_data = graph_store.graph.nodes.get(sel_node_id, {})
        sel_img_name = get_node_satellite_image(sel_node_id)
        sel_img_path = os.path.join(current_dir, "data", "damage_images", sel_img_name)

        # Telemetry styling & badges
        is_flood = graph_store.node_hazards(sel_node_id)["flood_depth_cm"] > 20
        is_struct = graph_store.node_hazards(sel_node_id)["damage_severity"] >= 0.85
        is_campus = (sel_node_id == "N_CAMPUS_VESIT")
        is_hosp = (sel_data.get("type") == "hospital")

        if is_flood:
            status_border = "#EF4444"
            status_pill = '<span class="badge-critical">CRITICAL FLOOD HAZARD</span>'
        elif is_struct:
            status_border = "#EF4444"
            status_pill = '<span class="badge-critical">CRITICAL STRUCTURAL HAZARD</span>'
        elif is_campus:
            status_border = "#A855F7"
            status_pill = '<span class="status-pill pill-cyan" style="background: rgba(168,85,247,0.25); color: #C084FC; border-color: rgba(168,85,247,0.4);">CAMPUS SAFE ZONE</span>'
        elif is_hosp:
            status_border = "#10B981"
            status_pill = f'<span class="status-pill pill-green">{sel_data.get("operational_status", "ACTIVE").upper()}</span>'
        else:
            status_border = "#0284C7"
            status_pill = f'<span class="status-pill pill-cyan">{sel_data.get("operational_status", "ACTIVE").upper()}</span>'

        capacity_str = f"<span>Capacity: <b style='color: #34D399;'>{sel_data.get('available_capacity', 0)}</b>/{sel_data.get('capacity', 0)} beds</span>" if sel_data.get('capacity') else ""
        icu_str = f"<span>ICU Beds: <b style='color: #38BDF8;'>{sel_data.get('icu_beds')}</b></span>" if sel_data.get('icu_beds') else ""
        telem_html = (
            f'<div class="glass-panel" style="padding: 10px 12px; margin-bottom: 8px; border-left: 3px solid {status_border};">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
            f'<span style="font-family: monospace; font-size: 10px; font-weight: 800; color: #38BDF8;">REFERENCE LOCATION</span>'
            f'{status_pill}</div>'
            f'<div style="font-size: 13.5px; font-weight: 700; color: #F8FAFC;">{sel_data.get("name", sel_node_id)}</div>'
            f'<div style="font-size: 10.5px; color: #94A3B8; margin-top: 4px; display: flex; flex-wrap: wrap; gap: 8px 14px;">'
            f'<span>GPS: <b style="color: #F1F5F9;">{sel_data.get("lat", 0):.4f} N, {sel_data.get("lon", 0):.4f} E</b></span>'
            f'<span>Elevation: <b style="color: #38BDF8;">{sel_data.get("elevation_m", 0)}m</b></span>'
            f'{capacity_str}{icu_str}</div></div>'
        )
        st.markdown(telem_html, unsafe_allow_html=True)

        if os.path.exists(sel_img_path):
            st.image(sel_img_path, caption=f"Reference image: {sel_data.get('name', sel_node_id)}", use_container_width=True)
        else:
            st.info("No bundled reference image is available for this location.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Route summary & Turn-by-Turn Navigation (Full-Width Row Below Map)
    disp_col1, disp_col2 = st.columns([1.1, 1.0])

    with disp_col1:
        st.markdown('<div class="panel-heading">Route summary</div>', unsafe_allow_html=True)

        target_info = current_state.get("target_facility_info")
        target_name = target_info.get("facility_name", "Designated Emergency Facility") if target_info else "Designated Emergency Facility"
        avail_beds = target_info.get("available_capacity", "N/A") if target_info else "N/A"
        total_beds = target_info.get("total_capacity", "N/A") if target_info else "N/A"
        icu_beds = target_info.get("icu_beds", "N/A") if target_info else "N/A"

        st.markdown(f"""
        <div class="glass-panel" style="border-left: 3px solid #38BDF8; padding: 12px 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <b style="color: #38BDF8; font-size: 11.5px; letter-spacing: 0.5px;">Destination facility</b>
                <span class="badge-clear">ACTIVE</span>
            </div>
            <div style="font-size: 15px; font-weight: 700; color: #F8FAFC;">{target_name}</div>
            <div style="font-size: 11.5px; color: #94A3B8; margin-top: 3px;">
                Emergency Capacity: <b style="color: #34D399;">{avail_beds}</b> / {total_beds} beds | ICU: <b style="color: #38BDF8;">{icu_beds}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if evidence_count == 0:
            st.info("No supporting incident reports were retrieved. The route is based on the current graph and incident registry snapshot.")

        origin_id = current_state.get("extracted_origin", "N_CHEMBUR_NAKA")
        dest_id = current_state.get("extracted_destination", "N_HOSP_ZEN")
        origin_label = graph_store.graph.nodes.get(origin_id, {}).get("name", origin_id)
        dest_label = graph_store.graph.nodes.get(dest_id, {}).get("name", dest_id)

        corridor_path_nodes = [graph_store.graph.nodes.get(n, {}).get("name", n) for n in route_nodes]
        corridor_path_str = " -> ".join(corridor_path_nodes) if corridor_path_nodes else "Direct Arterial Path"

        if is_air_evac:
            st.markdown(f"""
            <div class="directive-card">
                <div class="directive-header">Simulated aerial alternative</div>
                <div class="directive-route">{origin_label} → {dest_label}</div>
                <div class="directive-stats">Distance: {r_dist} km · Model estimate: {r_time} min</div>
                <div class="directive-item">The routing engine found no ground route for the selected vehicle. This air corridor illustrates a fallback scenario.</div>
                <div class="directive-item">Aircraft availability and landing sites are not verified. No external dispatch or hospital notification is performed.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="directive-card">
                <div class="directive-header">
                    <span>Recommended route</span>
                    <span class="badge-clear">{r_status} CORRIDOR</span>
                </div>
                <div class="directive-route">
                    {origin_label} -> {dest_label}
                </div>
                <div class="directive-stats">
                    <span>Transit: <b style="color: #38BDF8;">{r_time} min</b></span>
                    <span>Distance: <b style="color: #F8FAFC;">{r_dist} km</b></span>
                    <span>Evidence: <b style="color: #34D399;">{evidence_count} reports</b></span>
                </div>
                <div style="font-size: 11px; text-transform: uppercase; color: #64748B; font-weight: 700; margin-bottom: 4px;">
                    Selected route:
                </div>
                <div style="font-size: 11px; color: #A78BFA; font-family: monospace; line-height: 1.4; margin-bottom: 10px;">
                    {corridor_path_str}
                </div>
                <div class="directive-item">
                    <span style="color: #34D399; font-weight: bold;">[OK]</span>
                    <span>Selected corridor satisfies the configured vehicle restrictions in the current incident snapshot.</span>
                </div>
                <div class="directive-item">
                    <span style="color: #FBBF24; font-weight: bold;">[BYPASS]</span>
                    <span>Active road restrictions are applied from all incident reports; resolved incidents no longer block roads.</span>
                </div>
                <div class="directive-item">
                    <span style="color: #38BDF8; font-weight: bold;">[TRIAGE]</span>
                    <span>Destination availability comes from the demo dataset; no bed booking or notification has been sent.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with disp_col2:
        st.markdown('<div class="panel-heading">Route segments</div>', unsafe_allow_html=True)
        if primary_route and primary_route.get("segments"):
            st.markdown('<div style="max-height: 185px; overflow-y: auto;">', unsafe_allow_html=True)
            for idx, seg in enumerate(primary_route["segments"], 1):
                seg_status = seg.get("status", "open")
                if is_air_evac:
                    tag_class = "badge-critical"
                    status_lbl = "AIR CORRIDOR"
                    tag_style = 'style="background: #F59E0B; color: #000; font-weight: 800;"'
                else:
                    tag_class = "badge-clear" if seg_status == "open" else ("badge-caution" if seg_status == "caution" else "badge-critical")
                    status_lbl = seg_status.upper()
                    tag_style = ''

                from_label = graph_store.graph.nodes.get(seg.get('from_node'), {}).get('name', seg.get('from_node'))
                to_label = graph_store.graph.nodes.get(seg.get('to_node'), {}).get('name', seg.get('to_node'))
                if is_air_evac:
                    from_label += " [Extraction LZ]"
                    to_label += " [Helipad Reception]"

                st.markdown(f"""
                <div class="turn-item">
                    <div>
                        <span style="font-family: monospace; color: {'#F59E0B' if is_air_evac else '#38BDF8'}; font-size: 11px; font-weight: 700;">#{idx}</span>
                        <b style="color: #F1F5F9; font-size: 12px; margin-left: 6px;">{seg.get('road_name')}</b>
                        <div style="font-size: 11px; color: #64748B; margin-top: 2px;">
                            {from_label} -> {to_label}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span class="{tag_class}" {tag_style}>{status_lbl}</span>
                        <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">{seg.get('distance_km')} km</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No route segments are available. The current restrictions may block every safe corridor. Review the incident registry or select another vehicle or destination.")

        # Contingency Corridors (K-Shortest Paths)
        alt_routes = current_state.get("alternative_routes", [])
        if alt_routes and len(alt_routes) > 1:
            with st.expander("Contingency Corridors (K-Shortest Paths)", expanded=False):
                for alt in alt_routes[1:]:
                    alt_node_names = [graph_store.graph.nodes.get(n, {}).get("name", n) for n in alt.get("path", [])]
                    st.markdown(f"""
                    <div style="background: #0B111E; border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 8px 10px; margin-bottom: 6px; font-size: 11px;">
                        <div style="display: flex; justify-content: space-between;">
                            <b style="color: #A78BFA;">Corridor #{alt['rank']}</b>
                            <span style="color: #38BDF8;">ETA: {alt['eta_min']} min ({alt['distance_km']} km)</span>
                        </div>
                        <div style="color: #64748B; font-family: monospace; margin-top: 3px; font-size: 10px; word-break: break-all;">
                            {' -> '.join(alt_node_names)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: DECISION INTELLIGENCE (Explainability & Attribution)
# ==============================================================================
with tab_decision:
    explanation = visualizer.explain_route_decision(
        primary_route=current_state.get("primary_route"),
        alternative_routes=current_state.get("alternative_routes", []),
        retrieved_incidents=current_state.get("retrieved_incidents", []),
        target_facility_info=current_state.get("target_facility_info")
    )

    # 1. LLM Cognitive Decision Attribution (Hero Component)
    llm_attribution_text = explanation.get("llm_attribution") or explanation.get("summary", "No attribution available.")
    llm_badge = explanation.get("llm_source", "GEMINI 3.1 FLASH LITE")

    # Clean and split bullet points to prevent raw markdown asterisks or detached bullet dots
    raw_lines = [l.strip() for l in llm_attribution_text.split("\n") if l.strip()]
    cleaned_bullets = []
    current_bullet = []
    for l in raw_lines:
        is_bullet_start = bool(re.match(r'^[*\-•\d]+[.)]?\s+', l))
        if is_bullet_start and current_bullet:
            cleaned_bullets.append(" ".join(current_bullet))
            current_bullet = []
        cleaned = re.sub(r'^[*\-•\d]+[.)]?\s*', '', l).strip()
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'<b style="color: #F8FAFC;">\1</b>', cleaned)
        if cleaned:
            current_bullet.append(cleaned)
    if current_bullet:
        cleaned_bullets.append(" ".join(current_bullet))

    if not cleaned_bullets:
        cleaned_bullets = [llm_attribution_text]

    bullet_html_list = []
    for b in cleaned_bullets:
        bullet_html_list.append(
            f'<div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 10px;">'
            f'<span style="color: #A78BFA; font-size: 16px; line-height: 1.4; user-select: none;">&#8226;</span>'
            f'<div style="font-size: 14px; color: #E2E8F0; line-height: 1.6; font-weight: 450;">{b}</div>'
            f'</div>'
        )
    rendered_bullets = "".join(bullet_html_list)

    st.markdown(
        f'<div class="glass-panel" style="border-left: 3px solid #8B5CF6; margin-bottom: 16px; background: rgba(139, 92, 246, 0.05); padding: 18px 22px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">'
        f'<span style="font-family: monospace; font-size: 13px; font-weight: 800; color: #C4B5FD; letter-spacing: 0.8px;">GENERATED ROUTE EXPLANATION</span>'
        f'<span class="status-pill pill-cyan" style="background: rgba(139, 92, 246, 0.25); color: #DDD6FE; border-color: rgba(139, 92, 246, 0.5); font-size: 11px; font-weight: 700; padding: 3px 10px;">SOURCE: {llm_badge}</span>'
        f'</div>'
        f'<div>{rendered_bullets}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # 2. Executive Routing Telemetry Strip
    pr = current_state.get("primary_route")
    if pr:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="telemetry-card"><div class="metric-label" style="font-size: 11px;">Distance</div>'
                f'<div class="metric-value" style="font-size: 22px;">{pr.get("total_distance_km", 0):.2f} <span style="font-size: 13px; color: #64748B;">km</span></div></div>',
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f'<div class="telemetry-card"><div class="metric-label" style="font-size: 11px;">Estimated Transit</div>'
                f'<div class="metric-value" style="font-size: 22px;">{pr.get("estimated_travel_time_min", 0):.1f} <span style="font-size: 13px; color: #64748B;">min</span></div></div>',
                unsafe_allow_html=True
            )
        with m3:
            elim_cnt = len(explanation.get("eliminated_edges", []))
            st.markdown(
                f'<div class="telemetry-card"><div class="metric-label" style="font-size: 11px;">Hazard Edges Disqualified</div>'
                f'<div class="metric-value" style="font-size: 22px; color: #F87171;">{elim_cnt} <span style="font-size: 13px; color: #64748B;">zones</span></div></div>',
                unsafe_allow_html=True
            )
        with m4:
            t_info = current_state.get("target_facility_info", {})
            beds = t_info.get("available_capacity", "--") if t_info else "--"
            st.markdown(
                f'<div class="telemetry-card"><div class="metric-label" style="font-size: 11px;">Target Intake Capacity</div>'
                f'<div class="metric-value" style="font-size: 22px; color: #34D399;">{beds} <span style="font-size: 13px; color: #64748B;">beds</span></div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    dec_col1, dec_col2 = st.columns([1.25, 1.0])

    with dec_col1:
        # A. Disqualified Hazards (Clean Compact Matrix)
        st.markdown('<div class="panel-heading" style="font-size: 13px;">Road restriction evidence</div>', unsafe_allow_html=True)
        eliminated = explanation.get("eliminated_edges", [])
        if eliminated:
            table_rows = []
            for el in eliminated:
                r_name = el.get('road_name', el.get('edge_id'))
                row_html = (
                    '<tr style="border-bottom: 1px solid rgba(255,255,255,0.08);">'
                    f'<td style="padding: 12px 14px;"><b style="color: #F87171; font-size: 13.5px;">{r_name}</b><br/><span style="color: #64748B; font-family: monospace; font-size: 10.5px;">{el.get("edge_id")}</span></td>'
                    f'<td style="padding: 12px 14px; white-space: nowrap;"><b style="color: #38BDF8; font-size: 13.5px;">{el.get("flood_depth")}</b><br/><span style="color: #F87171; font-size: 12px;">{el.get("damage_severity")} Damage</span></td>'
                    f'<td style="padding: 12px 14px; color: #E2E8F0; font-size: 13px; font-weight: 450;">{el.get("disqualification_rationale")}</td>'
                    f'<td style="padding: 12px 14px; text-align: right;"><span class="badge-critical" style="font-size: 11px; padding: 4px 8px;">{el.get("reason")}</span></td>'
                    '</tr>'
                )
                table_rows.append(row_html)
            rows_joined = "".join(table_rows)
            table_html = (
                '<div class="glass-panel" style="padding: 0; overflow: hidden; margin-bottom: 14px;">'
                '<table style="width: 100%; border-collapse: collapse; text-align: left;">'
                '<thead><tr style="background: rgba(255,255,255,0.04); border-bottom: 1px solid rgba(255,255,255,0.1); color: #94A3B8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">'
                '<th style="padding: 10px 14px;">Corridor Segment</th>'
                '<th style="padding: 10px 14px;">Telemetry</th>'
                '<th style="padding: 10px 14px;">Disqualification Reason</th>'
                '<th style="padding: 10px 14px; text-align: right;">Verdict</th>'
                f'</tr></thead><tbody>{rows_joined}</tbody></table></div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="glass-panel" style="color: #94A3B8; font-size: 13px; margin-bottom: 14px;">'
                'All arterial road segments along selected corridor are within safe clearance limits.</div>',
                unsafe_allow_html=True
            )

        # B. Critical Infrastructure Bottlenecks
        st.markdown('<div class="panel-heading" style="font-size: 13px;">Infrastructure bottlenecks</div>', unsafe_allow_html=True)
        bottlenecks = explanation.get("bottlenecks", [])
        if bottlenecks:
            for b in bottlenecks:
                st.markdown(
                    f'<div class="glass-panel" style="border-left: 3px solid #F59E0B; padding: 10px 14px; margin-bottom: 8px;">'
                    f'<div style="color: #FDE68A; font-size: 13px; line-height: 1.5; font-weight: 500;">{b}</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="glass-panel" style="color: #94A3B8; font-size: 13px;">'
                'No single-span bridges or bottleneck vulnerabilities detected on this corridor.</div>',
                unsafe_allow_html=True
            )

    with dec_col2:
        # C. Counterfactual Path Comparison ("Why Not Path B?")
        st.markdown('<div class="panel-heading" style="font-size: 13px;">Why this route?</div>', unsafe_allow_html=True)
        cf = explanation.get("counterfactual_analysis", {})
        if isinstance(cf, dict) and cf.get("details"):
            for d in cf["details"]:
                st.markdown(
                    f'<div class="glass-panel" style="border-left: 3px solid #8B5CF6; padding: 12px 14px; margin-bottom: 10px;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                    f'<b style="color: #C4B5FD; font-size: 13px;">Candidate Corridor #{d.get("alternative_rank")}</b>'
                    f'<span class="badge-caution" style="font-size: 10.5px; padding: 2px 7px;">REJECTED</span></div>'
                    f'<div style="font-size: 13px; color: #F1F5F9; line-height: 1.45; font-weight: 450;">{d.get("rejection_rationale")}</div>'
                    f'<div style="display: flex; gap: 14px; font-size: 12px; margin-top: 8px;">'
                    f'<span style="background: rgba(248,113,113,0.15); color: #F87171; padding: 2px 8px; border-radius: 4px; font-weight: 600;">+{d.get("eta_penalty_min")} min delay</span>'
                    f'<span style="background: rgba(251,191,36,0.15); color: #FCD34D; padding: 2px 8px; border-radius: 4px; font-weight: 600;">+{d.get("distance_penalty_km")} km detour</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="glass-panel" style="color: #94A3B8; font-size: 13px; margin-bottom: 14px;">'
                'Selected corridor is the singularly viable path under active hazard constraints.</div>',
                unsafe_allow_html=True
            )

        # D. Multimodal Satellite Evidence Grounding
        st.markdown('<div class="panel-heading" style="font-size: 13px;">Related reports and images</div>', unsafe_allow_html=True)
        citations = explanation.get("evidence_citations", [])
        if citations:
            for cit in citations:
                img_ev = cit.get("image_evidence")
                img_name = os.path.basename(img_ev) if img_ev else "Sensor Mesh Grounding"
                st.markdown(
                    f'<div class="glass-panel" style="border-left: 2px solid #06B6D4; padding: 12px 14px; margin-bottom: 8px;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
                    f'<b style="color: #38BDF8; font-family: monospace; font-size: 12px;">{cit.get("report_id")}</b>'
                    f'<span style="font-size: 11px; color: #94A3B8;">{cit.get("source")}</span></div>'
                    f'<div style="font-size: 13px; color: #F1F5F9; line-height: 1.4; font-weight: 500;">{cit.get("incident_type")}</div>'
                    f'<div style="font-size: 11.5px; color: #94A3B8; margin-top: 5px;">Satellite Telemetry: <code style="color: #38BDF8; font-size: 11px;">{img_name}</code></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="glass-panel" style="color: #94A3B8; font-size: 13px;">'
                'No external satellite incident reports triggered for this corridor.</div>',
                unsafe_allow_html=True
            )





# ==============================================================================
# TAB 3: KNOWLEDGE GRAPH TOPOLOGY
# ==============================================================================
with tab_graph:
    st.markdown('<div class="panel-heading">Road network and selected route</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 10px;">
        Force-directed spatial knowledge graph of Chembur, Mumbai. Traversed emergency corridor is highlighted in <b>Cyber Cyan (#00F0FF)</b>,
        and emergency aerial medevac flight vectors in <b>Dashed Gold (#F59E0B)</b>. Real-time physics engine dynamically balances nodes and edges based on topological tension.
    </div>
    """, unsafe_allow_html=True)

    # Tactical HUD Legend Pill Bar for Knowledge Graph
    st.markdown("""
    <div style="background: rgba(11, 17, 30, 0.9); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 11px; align-items: center; box-shadow: 0 4px 16px rgba(0,0,0,0.4);">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #00F0FF; border: 1.5px solid #FFFFFF;"></span>
            <span style="color: #00F0FF; font-weight: 700;">Extraction Origin</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #10B981; border: 1.5px solid #FEF08A;"></span>
            <span style="color: #34D399; font-weight: 700;">Target Destination</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #0284C7; border: 1.5px solid #67E8F9;"></span>
            <span style="color: #38BDF8;">Central Flood (VN Purav)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #EF4444; border: 1.5px solid #FCA5A5;"></span>
            <span style="color: #F87171;">Structural Shear (Suman)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #A855F7; border: 1.5px solid #C084FC;"></span>
            <span style="color: #C084FC;">VESIT Campus</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 9px; height: 9px; border-radius: 2px; background: #059669; border: 1px solid #34D399;"></span>
            <span style="color: #34D399;">Hospital (Zen / Surana)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 9px; height: 9px; border-radius: 2px; background: #D97706; border: 1px solid #FCD34D;"></span>
            <span style="color: #FBBF24;">Relief Shelter (Acres / RCF)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 9px; height: 9px; border-radius: 2px; background: #6366F1; border: 1px solid #A5B4FC;"></span>
            <span style="color: #A5B4FC;">Flyover (Amar Mahal / Panjarpol)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #EA580C; border: 1px solid #FDBA74;"></span>
            <span style="color: #FB923C;">Open Junction (Shell Colony, etc.)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 16px; height: 3px; background: #00F0FF; border-radius: 2px;"></span>
            <span style="color: #00F0FF;">Active Safe Corridor</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 16px; height: 0; border-top: 2.5px dashed #F59E0B;"></span>
            <span style="color: #F59E0B;">Simulated aerial alternative</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 16px; height: 0; border-top: 2px dashed #EF4444;"></span>
            <span style="color: #EF4444;">Blocked / Severed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pyvis_html = graph_store.export_pyvis_network(highlight_path=route_nodes, is_aerial=is_air_evac, height="580px")
    components.html(pyvis_html, height=590, scrolling=False)

    # Subgraph Stats
    sg1, sg2, sg3, sg4 = st.columns(4)
    with sg1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Total Spatial Nodes</div>
            <div class="metric-number">{len(graph_store.graph.nodes)}</div>
        </div>
        """, unsafe_allow_html=True)
    with sg2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Network Road Edges</div>
            <div class="metric-number">{len(graph_store.graph.edges)}</div>
        </div>
        """, unsafe_allow_html=True)
    with sg3:
        transit_unit = "vector" if is_air_evac else "nodes"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Traversed Path Length</div>
            <div class="metric-number">{len(route_nodes)} <span class="metric-unit">{transit_unit}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with sg4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-caption">Target Destination</div>
            <div class="metric-number" style="font-size: 18px; color: #10B981;">{graph_store.graph.nodes.get(primary_route.get('target', ''), {}).get('name', 'N/A') if primary_route else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Places and graph relationships</div>', unsafe_allow_html=True)

    kg_tab1, kg_tab2 = st.tabs(["Active Knowledge Triples & Semantic Relations", "Geospatial Entity Node Registry"])

    with kg_tab1:
        st.markdown("""
        <div style="font-size: 11.5px; color: #94A3B8; margin-bottom: 10px;">
            Verified crisis infrastructure triples loaded in the GraphRAG store. Nodes and dynamic hazard states dynamically penalize edge traversal cost in Dijkstra / A*.
        </div>
        """, unsafe_allow_html=True)

        # Display semantic triples in a clean tactical table
        triples_data = []
        for u, v, data in graph_store.graph.edges(data=True):
            if data.get("id", "").endswith("_REV"):
                continue
            u_name = graph_store.graph.nodes.get(u, {}).get("name", u)
            v_name = graph_store.graph.nodes.get(v, {}).get("name", v)
            status = data.get("status", "open")
            dmg = int(data.get("damage_severity", 0) * 100)
            fld = data.get("flood_depth_cm", 0)
            dist = data.get("distance_km", 0)

            if status == "flooded" or fld > 25:
                pred = "INUNDATED_CORRIDOR"
                badge = f'<span class="badge-critical" style="background: rgba(56,189,248,0.2); color: #38BDF8; border-color: #38BDF8;">FLOOD {fld}cm</span>'
            elif status in ["severely_damaged", "blocked"] or dmg > 50:
                pred = "STRUCTURAL_SEVERANCE"
                badge = f'<span class="badge-critical">DAMAGE {dmg}%</span>'
            elif status == "caution":
                pred = "RESTRICTED_ACCESS"
                badge = '<span class="badge-caution">CAUTION</span>'
            else:
                pred = "ROAD_CONNECTS"
                badge = '<span class="badge-clear">TRAVERSABLE</span>'

            triples_data.append({
                "Subject (Source)": f"{u_name} ({u})",
                "Predicate (Relationship)": pred,
                "Object (Target)": f"{v_name} ({v})",
                "Distance": f"{dist} km",
                "Status": badge
            })

        triples_df = pd.DataFrame(triples_data)
        st.write(triples_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    with kg_tab2:
        st.markdown("""
        <div style="font-size: 11.5px; color: #94A3B8; margin-bottom: 10px;">
            Canonical spatial entities within the Mumbai Chembur metropolitan disaster zone.
        </div>
        """, unsafe_allow_html=True)

        nodes_data = []
        for nid, data in graph_store.graph.nodes(data=True):
            ntype = data.get("type", "intersection").upper()
            status = data.get("operational_status", "active")
            cap = f"{data.get('available_capacity', 0)}/{data.get('capacity', 0)}" if data.get('capacity') else "N/A"
            icu = str(data.get("icu_beds", "N/A")) if data.get("icu_beds") is not None else "N/A"
            helipad = "Yes (Rooftop)" if data.get("helipad") else "No"

            nodes_data.append({
                "Canonical ID": nid,
                "Entity Name": data.get("name", nid),
                "Classification": ntype,
                "Status": status.upper(),
                "Elevation": f"{data.get('elevation_m', 0)} m",
                "Capacity (Avail/Total)": cap,
                "ICU Beds": icu,
                "Helipad LZ": helipad
            })

        nodes_df = pd.DataFrame(nodes_data)
        st.dataframe(nodes_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 5: SYSTEM QUALITY & VERIFICATION (Benchmarks & Evaluation)
# ==============================================================================
with tab_eval:
    st.markdown('<div class="panel-heading">Evaluate this dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 13px; color: #94A3B8; margin-bottom: 16px;">
        Run the existing benchmark to calculate retrieval metrics on the current dataset.
        Grounding and judge scores use custom heuristics, with an optional model judge; they are not a validated measure of operational safety.
    </div>
    """, unsafe_allow_html=True)

    # Execution Button
    eval_btn_col1, eval_btn_col2 = st.columns([1.5, 2.5])
    with eval_btn_col1:
        if st.button("Run benchmark", type="primary", use_container_width=True):
            with st.spinner("Calculating retrieval metrics and the existing evaluation rubrics..."):
                bench_results = run_benchmark_suite(agent, retriever)
                st.session_state.benchmark_results = bench_results
                st.success("Quality Benchmark Complete.")

    if "benchmark_results" not in st.session_state:
        st.info("No benchmark has been run in this session. Choose Run benchmark to calculate results for the current dataset.")
    else:
        results = st.session_state.benchmark_results
        # Metric Cards
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        with bm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-caption">Ranking NDCG@3</div>
                <div class="metric-number" style="color: #38BDF8;">{results['mean_ndcg@3']}</div>
                <div class="metric-sub">Ranked relevance gain</div>
            </div>
            """, unsafe_allow_html=True)
        with bm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-caption">Reciprocal Rank (MRR)</div>
                <div class="metric-number" style="color: #34D399;">{results['mrr']}</div>
                <div class="metric-sub">Rank-1 hazard retrieval</div>
            </div>
            """, unsafe_allow_html=True)
        with bm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-caption">Precision@3</div>
                <div class="metric-number" style="color: #F8FAFC;">{results['mean_precision@3']}</div>
                <div class="metric-sub">Relevant report ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with bm4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-caption">Grounding heuristic</div>
                <div class="metric-number" style="color: #A78BFA;">{results['mean_faithfulness']}</div>
                <div class="metric-sub">Custom heuristic</div>
            </div>
            """, unsafe_allow_html=True)
        with bm5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-caption">Judge rubric score</div>
                <div class="metric-number" style="color: #FBBF24;">{results['mean_judge_score']}<span class="metric-unit">/10</span></div>
                <div class="metric-sub">Existing evaluation rubric</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Detailed Benchmark Scorecard Table
        st.markdown("##### Scenario Benchmark Scorecard")
        df_detailed = pd.DataFrame(results["detailed_results"])
        st.dataframe(
            df_detailed[[
                "query_id", "query", "ndcg@3", "mrr", "precision@3",
                "recall@3", "faithfulness", "answer_relevance", "judge_composite", "judge_critique"
            ]],
            use_container_width=True
        )

        # Metric Comparison Chart (Clean Tactical Progress Scorecard)
        st.markdown('<div class="panel-heading" style="margin-top: 16px;">Retrieval and grounding metrics</div>', unsafe_allow_html=True)

        metrics_display = [
            ("Ranking Quality (NDCG@3)", results['mean_ndcg@3'], 1.0, "#38BDF8", "Measures graded relevance of top-3 retrieved hazard reports"),
            ("Reciprocal Rank (MRR)", results['mrr'], 1.0, "#34D399", "Speed of retrieving the single most critical blocking incident at rank 1"),
            ("Precision@3", results['mean_precision@3'], 1.0, "#38BDF8", "Ratio of retrieved documents that were strictly relevant to the corridor"),
            ("Recall@3", results['mean_recall@3'], 1.0, "#34D399", "Coverage of ground-truth hazards captured by the retriever"),
            ("Grounding heuristic", results['mean_faithfulness'], 1.0, "#A78BFA", "Custom grounding heuristic; see About for its interpretation"),
            ("Answer Relevance", results['mean_answer_relevance'], 1.0, "#A78BFA", "Degree of alignment between user query constraints and final tactical order"),
            ("Judge rubric score", results['mean_judge_score'] / 10.0, 1.0, "#FBBF24", "Existing judge rubric, using a model or deterministic fallback (scaled /10)")
        ]

        c_left, c_right = st.columns(2)
        for idx, (m_name, val, max_v, bar_color, desc) in enumerate(metrics_display):
            pct = int(min(val / max_v, 1.0) * 100)
            target_col = c_left if idx % 2 == 0 else c_right
            with target_col:
                st.markdown(f"""
                <div class="glass-panel" style="padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                        <b style="color: #F1F5F9; font-size: 11.5px;">{m_name}</b>
                        <span style="font-family: monospace; font-weight: 800; color: {bar_color}; font-size: 12px;">{val:.3f} ({pct}%)</span>
                    </div>
                    <div style="font-size: 10px; color: #64748B; margin-bottom: 6px;">{desc}</div>
                    <div style="background: rgba(255, 255, 255, 0.06); height: 5px; border-radius: 3px; overflow: hidden;">
                        <div style="background: {bar_color}; width: {pct}%; height: 100%; border-radius: 3px; box-shadow: 0 0 6px {bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
