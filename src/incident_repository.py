"""Persistent incident snapshots with optimistic edit checks and atomic writes."""
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import portalocker


class IncidentRepository:
    def __init__(self, path, seed_path, roads):
        self.path = Path(path)
        self.seed_path = Path(seed_path)
        self.roads = roads

    def load(self):
        source = self.path if self.path.exists() else self.seed_path
        raw = source.read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    def save(self, report, expected_revision):
        report = dict(report)
        for field in ("report_id", "incident_type", "text", "source"):
            if not isinstance(report.get(field), str) or not report[field].strip():
                raise ValueError(f"{field.replace('_', ' ').title()} is required")
            report[field] = report[field].strip()
        edges = report.get("affected_edges", [])
        if not isinstance(edges, list) or not edges or any(edge not in self.roads for edge in edges):
            raise ValueError("Select at least one valid affected road")
        if report.get("incident_status", "active") not in ("active", "resolved"):
            raise ValueError("Invalid incident status")
        if report.get("severity") not in ("CLEAR", "MODERATE", "HIGH", "CRITICAL"):
            raise ValueError("Invalid severity")
        for field, maximum in (("flood_depth_cm", 1000), ("damage_severity", 1)):
            value = float(report.get(field, 0))
            if not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{field} must be between 0 and {maximum}")
            report[field] = value
        report["affected_nodes"] = sorted({node for edge in edges for node in self.roads[edge]["nodes"]})
        report["location_ref"] = edges[0]
        report["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.path) + ".lock", timeout=5):
            reports, revision = self.load()
            if revision != expected_revision:
                raise ValueError("Incidents changed while you were editing. Reload the page and try again.")
            existing = next((i for i, row in enumerate(reports) if row["report_id"] == report["report_id"]), None)
            if existing is None:
                reports.append(report)
            else:
                previous = reports[existing]
                report["cleared_edges"] = sorted(
                    set(previous.get("cleared_edges", []))
                    | (set(previous.get("affected_edges", [])) - set(edges))
                )
                reports[existing] = report
            fd, filename = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(reports, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(filename, self.path)
            finally:
                if os.path.exists(filename):
                    os.unlink(filename)
        return report
