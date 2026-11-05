"""Incident editor kept separate from the mission dashboard layout."""
import uuid
import streamlit as st


def render_incident_manager(repository):
    if "incident_notice" in st.session_state:
        st.success(st.session_state.pop("incident_notice"))
    with st.expander("Incident management — add, update, or resolve reports"):
        st.caption("Save a report to update Qdrant and recalculate the current route. Resolved reports remain in the registry but stop imposing restrictions. Operator edits are saved locally.")
        reports, revision = repository.load()
        st.dataframe([{
            "ID": row["report_id"], "Incident": row["incident_type"],
            "Status": row.get("incident_status", "active"), "Severity": row.get("severity", ""),
            "Flood (cm)": row.get("flood_depth_cm", 0),
            "Updated (UTC)": row.get("timestamp", ""),
        } for row in reports], hide_index=True, use_container_width=True)
        by_id = {row["report_id"]: row for row in reports}
        selected = st.selectbox("Incident to edit", ["New incident"] + list(by_id),
                                format_func=lambda key: key if key == "New incident" else f"{key}: {by_id[key]['incident_type']}")
        report = by_id.get(selected, {})
        # Keep the revision associated with the displayed form on submission,
        # rather than silently overwriting another operator's intervening edit.
        revision_key = f"incident_revision_{selected}"
        st.session_state.setdefault(revision_key, revision)
        with st.form(f"incident_editor_{selected}"):
            title = st.text_input("Incident title", value=report.get("incident_type", ""))
            source = st.text_input("Report source", value=report.get("source", "Operator report"))
            roads = st.multiselect("Affected roads", list(repository.roads),
                default=[edge for edge in report.get("affected_edges", []) if edge in repository.roads],
                format_func=lambda edge: f"{repository.roads[edge]['name']} ({edge})")
            left, right = st.columns(2)
            with left:
                states = ["active", "resolved"]
                status = st.selectbox("Incident status", states, index=states.index(report.get("incident_status", "active")))
                flood = st.number_input("Flood depth (cm)", 0.0, 1000.0, float(report.get("flood_depth_cm", 0)), step=5.0)
            with right:
                severities = ["CLEAR", "MODERATE", "HIGH", "CRITICAL"]
                severity = st.selectbox("Incident severity", severities, index=severities.index(report.get("severity", "MODERATE")))
                damage = st.number_input("Structural damage (%)", 0.0, 100.0, float(report.get("damage_severity", 0)) * 100, step=1.0)
            closed = st.checkbox("Road closed to all vehicles", value=report.get("road_closed", False))
            text = st.text_area("Incident details", value=report.get("text", ""))
            action = st.text_area("Required action", value=report.get("action_required", ""))
            submitted = st.form_submit_button("Save incident", type="primary")
        if submitted:
            try:
                saved = repository.save(dict(report,
                    report_id=report.get("report_id") or f"INC-{uuid.uuid4().hex[:12].upper()}",
                    incident_type=title, source=source, affected_edges=roads,
                    incident_status=status, severity=severity, flood_depth_cm=flood,
                    damage_severity=damage / 100, road_closed=closed, text=text,
                    action_required=action), st.session_state[revision_key])
            except (ValueError, OSError) as error:
                st.error(str(error))
            else:
                for key in list(st.session_state):
                    if key.startswith("incident_revision_"):
                        del st.session_state[key]
                st.session_state.pop("current_state", None)
                st.session_state.pop("benchmark_results", None)
                st.session_state.incident_notice = f"Saved {saved['report_id']} as {status}. Route and Qdrant refreshed."
                st.rerun()
