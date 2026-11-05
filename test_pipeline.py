import sys
import os
import json
from qdrant_client import QdrantClient

# Ensure paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph_store import CrisisSpatialGraph
from src.multimodal_retriever import MultimodalRetriever
from src.agent_workflow import CrisisResponseAgent
from src.explainability import ReasoningChainVisualizer
from src.evaluation import run_benchmark_suite

def main():
    print("=== TEST 1: SPATIAL GRAPH STORE ===")
    osm_path = os.path.join("data", "sample_osm_network.json")
    graph = CrisisSpatialGraph(osm_path)
    print(f"Nodes loaded: {len(graph.graph.nodes)}, Edges: {len(graph.graph.edges)}")
    assert len(graph.graph.nodes) > 0, "No nodes loaded"
    assert len(graph.graph.edges) > 0, "No edges loaded"

    print("\n=== TEST 2: MULTIMODAL RETRIEVER ===")
    reports_path = os.path.join("data", "disaster_reports.json")
    client = QdrantClient(":memory:")
    retriever = MultimodalRetriever(embedding_dim=128, qdrant_client=client, embedding_provider="hash")
    with open(reports_path, "r", encoding="utf-8") as f:
        reps = json.load(f)
    retriever.index_reports(reps)
    matched = retriever.retrieve_reports("monsoon flood inundation at Shell Colony", top_k=2)
    print(f"Top matched report: {matched[0]['report']['report_id']} - {matched[0]['report']['incident_type']}")
    assert len(matched) > 0, "No reports retrieved"

    print("\n=== TEST 3: AGENT WORKFLOW ===")
    agent = CrisisResponseAgent(graph, retriever)
    state = agent.run_pipeline(
        query="Find fastest safe ambulance route from Chembur Naka to nearest hospital avoiding flooded streets",
        vehicle_type="ambulance",
        avoid_hazards=True
    )
    primary_route = state.get("primary_route")
    print(f"Route Path: {' -> '.join(primary_route['path'])}")
    print(f"Distance: {primary_route['total_distance_km']} km | ETA: {primary_route['estimated_travel_time_min']} min")
    assert primary_route is not None, "Primary route failed"
    assert len(primary_route["path"]) >= 2, "Path too short"

    print("\n=== TEST 4: EXPLAINABILITY (MODULE 6.3) ===")
    vis = ReasoningChainVisualizer(graph)
    exp = vis.explain_route_decision(
        primary_route=primary_route,
        alternative_routes=state.get("alternative_routes", []),
        retrieved_incidents=state.get("retrieved_incidents", []),
        target_facility_info=state.get("target_facility_info")
    )
    print(f"Explanation summary: {exp['summary']}")
    print(f"Eliminated hazard edges count: {len(exp['eliminated_edges'])}")
    assert len(exp["reasoning_steps"]) > 0, "No reasoning steps extracted"

    print("\n=== TEST 5: EVALUATION BENCHMARK (MODULE 6.2) ===")
    bench = run_benchmark_suite(agent, retriever)
    print(f"Mean NDCG@3: {bench['mean_ndcg@3']}")
    print(f"MRR: {bench['mrr']}")
    print(f"Mean Precision@3: {bench['mean_precision@3']}")
    print(f"Mean Faithfulness: {bench['mean_faithfulness']}")
    print(f"Judge Score: {bench['mean_judge_score']} / 10")
    assert bench["mean_ndcg@3"] > 0, "NDCG@3 should be > 0"
    assert bench["mrr"] > 0, "MRR should be > 0"

    client.close()
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
