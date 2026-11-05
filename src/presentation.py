"""Presentation components only; no routing, retrieval, or persistence logic."""
import base64
from pathlib import Path
import streamlit as st


LANDING_CSS = """
<style>
.st-key-about_hero { background: radial-gradient(ellipse at 85% 20%,rgba(14,165,233,.10),transparent 55%),linear-gradient(130deg,#101c30,#0d1625 65%); border:1px solid #26384d; border-radius:18px; padding:28px 32px 24px; margin:12px 0 8px; }
.intro-grid { display:grid; grid-template-columns:1.1fr 1fr; gap:30px; align-items:center; }
.intro-kicker { color:#65c9df; font:600 10px/1.5 system-ui; letter-spacing:2px; margin:0 0 16px; }
.intro-title { color:#f4f8ff; font-size:clamp(32px,3.7vw,50px)!important; line-height:1.12!important; font-weight:750!important; letter-spacing:-1.8px!important; margin:0 0 18px!important; padding:0!important; }
.intro-title span { color:#67d6d0; }
.intro-copy { color:#aebdd0; font-size:15px; line-height:1.75; max-width:510px; margin:0; }
.intro-action { display:block; text-align:center; background:#48abe0; color:#071423!important; text-decoration:none!important; font-size:14px; font-weight:650; padding:12px 16px; border-radius:8px; }
.intro-action:hover { background:#72c7ee; }
.intro-action:focus-visible { outline:2px solid #c1f1ff; outline-offset:3px; }
.intro-art { background:rgba(5,13,23,.40); border:1px solid rgba(94,136,163,.2); border-radius:14px; overflow:hidden; }
.intro-art svg { display:block; width:100%; height:auto; }
.intro-art-label { color:#8fa6bc; font-size:10px; letter-spacing:.4px; padding:10px 16px; border-top:1px solid #203044; }
.fact-row { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:12px 0 24px; }
.fact { padding:13px 18px; border:1px solid #223248; border-radius:10px; background:#0d1625; display:flex; align-items:center; gap:13px; }
.fact strong { font-size:27px; color:#e9f3ff; font-weight:650; letter-spacing:-1px; }
.fact span { color:#a4b4c8; font-size:12px; line-height:1.5; }
.section-kicker { color:#6c9eb4; font-size:10px; font-weight:700; letter-spacing:1.6px; margin:0 0 7px; }
.section-title { font-size:23px; color:#edf5ff; letter-spacing:-.6px; font-weight:650; margin:0 0 18px; }
.workflow-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:13px; margin-bottom:24px; }
.workflow-card { background:#0e1727; border:1px solid #223248; border-radius:12px; padding:20px; min-height:174px; }
.workflow-step { color:#63b7cc; font-size:11px; font-weight:700; letter-spacing:1px; margin-bottom:17px; }
.workflow-card h3 { color:#e6f2ff; font-size:15px!important; font-weight:650; padding:0!important; margin:0 0 9px!important; }
.workflow-card p { color:#9caec4; font-size:12px; line-height:1.75; margin:0; }
.sdg-row { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:10px 0 8px; }
.sdg-item { display:flex; align-items:center; gap:11px; padding:12px 14px; background:#0c1421; border:1px solid #1f2e40; border-radius:10px; }
.sdg-item img { width:54px; height:54px; flex-shrink:0; border-radius:5px; }
.sdg-item b { display:block; color:#e4edf7; font-size:12px; line-height:1.4; margin-bottom:4px; }
.sdg-label { color:#ffffff; font-size:15px; font-weight:750; letter-spacing:.6px; margin-bottom:4px; }
.sdg-3 {border-left:3px solid #4c9f38;} .sdg-9 {border-left:3px solid #fd6925;} .sdg-11 {border-left:3px solid #fd9d24;} .sdg-13 {border-left:3px solid #3f7e44;}
.sdg-item span { color:#a0b1c6; font-size:11px; }
.stTabs [data-baseweb="tab-highlight"] { background:#4bb6dc!important; }
.stTabs [aria-selected="true"] { background:#162d42!important; color:#92dafa!important; }
@media(max-width:900px) { .intro-grid {grid-template-columns:1fr 1fr; gap:16px;} .st-key-about_hero {padding:22px;} .intro-title{font-size:34px!important;} .intro-copy{font-size:13px;} .workflow-card{padding:15px;} }
@media(max-width:640px) { .intro-grid{grid-template-columns:1fr;} .intro-art{max-width:450px;} .fact-row,.workflow-grid,.sdg-row{grid-template-columns:repeat(2,1fr);} .intro-title{font-size:36px!important;} .st-key-about_hero{padding:20px;} .fact{padding:12px;gap:9px;} }
</style>
"""

HERO = """
<div class="intro-grid">
  <div>
    <div class="intro-kicker">GEOSPATIAL INTELLIGENCE · CHEMBUR, MUMBAI</div>
    <h1 class="intro-title">When the shortest route<br>isn’t the <span>safest.</span></h1>
    <p class="intro-copy">Aegis CrisisNav connects incident reports with a road network to explore emergency routes around floods and infrastructure damage.</p>
  </div>
  <div class="intro-art">
    <svg viewBox="0 0 500 260" role="img" aria-label="Illustration of an alternative route around a flooded road to a hospital">
      <defs>
        <pattern id="routeGrid" width="30" height="30" patternUnits="userSpaceOnUse"><path d="M 30 0 L 0 0 0 30" fill="none" stroke="#1b3042" stroke-width=".6"/></pattern>
        <linearGradient id="routeLine" x1="0" x2="1"><stop stop-color="#32a9d4"/><stop offset="1" stop-color="#61d6b0"/></linearGradient>
      </defs>
      <rect width="500" height="260" fill="url(#routeGrid)"/>
      <path d="M25 110L120 80L230 105L332 45L475 70 M25 215L130 195L200 225L340 205L475 205 M120 80L130 195 M230 105L200 225 M332 45L340 205 M435 70L435 225" fill="none" stroke="#294052" stroke-width="5" stroke-linejoin="round"/>
      <path d="M75 190L230 105L435 80" fill="none" stroke="#f08d86" stroke-opacity=".7" stroke-width="3" stroke-dasharray="6 5"/>
      <ellipse cx="230" cy="105" rx="51" ry="35" fill="#12628a" fill-opacity=".23" stroke="#328aaf" stroke-dasharray="3 4"/>
      <path d="M209 105q7-9 14 0t14 0t14 0" fill="none" stroke="#62bdde" stroke-width="2"/>
      <path d="M75 190L130 195L200 225L340 205L435 80" fill="none" stroke="#37c9ac" stroke-opacity=".10" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M75 190L130 195L200 225L340 205L435 80" fill="none" stroke="url(#routeLine)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="75" cy="190" r="10" fill="#163f59" stroke="#79d4f5" stroke-width="2"/><circle cx="75" cy="190" r="3" fill="#c8f1ff"/>
      <circle cx="435" cy="80" r="16" fill="#123e35" stroke="#63dcb0" stroke-width="2"/>
      <path d="M435 72v16M427 80h16" stroke="#9bead0" stroke-width="3"/>
      <circle cx="200" cy="225" r="4" fill="#83dbe4"/><circle cx="340" cy="205" r="4" fill="#83dbe4"/>
      <text x="44" y="164" fill="#b7cde0" font-size="11" font-family="system-ui">Origin</text>
      <text x="392" y="48" fill="#b7cde0" font-size="11" font-family="system-ui">Hospital</text>
      <rect x="174" y="45" width="112" height="23" rx="5" fill="#142a3c" stroke="#2b5068"/><text x="230" y="60" text-anchor="middle" fill="#8ac9e2" font-size="10" font-family="system-ui">Flooded road</text>
      <text x="247" y="190" fill="#76cbb5" font-size="10" font-family="system-ui">Alternative route</text>
    </svg>
    <div class="intro-art-label">ILLUSTRATION · A route around a reported hazard</div>
  </div>
</div>
"""


def render_about(graph_store, reports, vehicle_profiles, project_root):
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    _render_sdgs(project_root)
    with st.container(key="about_hero"):
        st.markdown(HERO, unsafe_allow_html=True)
        action, explanation = st.columns([1, 2], vertical_alignment="center")
        with action:
            st.markdown('<a class="intro-action" href="?view=planner" target="_self">Explore route planner →</a>', unsafe_allow_html=True)
        with explanation:
            st.caption("Plan a route. Update an incident. See what changes.")

    roads = sum(1 for _, _, data in graph_store.graph.edges(data=True) if not data["id"].endswith("_REV"))
    facts = [(len(graph_store.graph), "connected locations"), (roads, "road links"),
             (len(reports), "incident reports"), (len(vehicle_profiles), "vehicle profiles")]
    st.markdown('<div class="fact-row">' + ''.join(
        f'<div class="fact"><strong>{value}</strong><span>{label}</span></div>' for value, label in facts
    ) + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">THE WORKFLOW</div><div class="section-title">From incident to explainable route.</div>', unsafe_allow_html=True)
    steps = [
        ("01 / DESCRIBE", "Set the destination", "Enter a route request and choose a vehicle with its configured flood clearance."),
        ("02 / RETRIEVE", "Find the evidence", "Search incident reports and related images stored in Qdrant."),
        ("03 / ROUTE", "Apply the restrictions", "Use the road graph and all active incidents to calculate an alternative path."),
        ("04 / EXPLAIN", "Inspect the decision", "Review the selected roads, supporting reports, and alternative routes."),
    ]
    st.markdown('<div class="workflow-grid">' + ''.join(
        f'<div class="workflow-card"><div class="workflow-step">{number}</div><h3>{title}</h3><p>{detail}</p></div>'
        for number, title, detail in steps
    ) + '</div>', unsafe_allow_html=True)

    with st.expander("Dataset & evaluation notes"):
        st.markdown("This is a **Chembur demonstration dataset**, not a live emergency service. Road restrictions, hospital capacity, and aerial alternatives are simulated or entered manually.")
        st.markdown("Route estimates include configured hazard penalties. Image search uses visual features and filenames; it does not diagnose damage or measure flood depth. Evidence counts are not safety probabilities.")
        st.markdown("Run the benchmark in **Evaluation** to calculate retrieval metrics. Grounding and judge scores include custom heuristics and are not operational safety validation.")


def _render_sdgs(project_root):
    st.markdown('<div class="section-kicker" style="margin-top:12px">CONNECTED TO THE SUSTAINABLE DEVELOPMENT GOALS</div>', unsafe_allow_html=True)
    sdgs = [(3, "Good health & well-being", "Emergency access"), (9, "Industry, innovation & infrastructure", "Road-network awareness"),
            (11, "Sustainable cities & communities", "Urban disaster response"), (13, "Climate action", "Flood resilience")]
    cards = []
    for number, title, subtitle in sdgs:
        path = Path(project_root) / "assets" / "sdg" / f"sdg_{number}.svg"
        icon = f'<img alt="SDG {number}" src="data:image/svg+xml;base64,{base64.b64encode(path.read_bytes()).decode()}"/>' if path.exists() else ''
        cards.append(f'<div class="sdg-item sdg-{number}">{icon}<div><div class="sdg-label">SDG {number}</div><b>{title}</b><span>{subtitle}</span></div></div>')
    st.markdown('<div class="sdg-row">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
