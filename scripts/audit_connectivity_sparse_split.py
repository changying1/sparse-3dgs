import argparse
import itertools
import json
import sqlite3
from pathlib import Path


COLMAP_PAIR_ID_BASE = 2147483647
TARGET_INDICES_9 = [0, 31, 62, 94, 125, 156, 188, 219, 250]
ROBUST_THRESHOLDS = [100, 50, 30, 15]


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic connectivity-aware sparse-view split audit."
    )
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", default="phase_12c_1b_connectivity_split_audit.json")
    parser.add_argument("--beam-width", type=int, default=10000)
    args = parser.parse_args()

    audit = run_audit(Path(args.database), beam_width=args.beam_width)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(summary_for_console(audit), indent=2))


def run_audit(database_path, beam_width):
    images, raw_matches, verified = load_database(database_path)
    edge_distribution = distribution([edge["verified_inliers"] for edge in verified.values() if edge["verified_inliers"] > 0])

    selected_window = None
    connected_candidates = []
    search_reports = []
    for window in (15, 25):
        candidates_by_slot = candidate_windows(len(images), TARGET_INDICES_9, window)
        connected_candidates, report = search_s9(
            images=images,
            verified=verified,
            candidates_by_slot=candidates_by_slot,
            beam_width=beam_width,
            window_half_width=window,
        )
        search_reports.append(report)
        selected_window = window
        if connected_candidates:
            break

    top_s9 = connected_candidates[:5]
    for item in top_s9:
        item["best_nested"] = best_nested_splits(item["indices"], images, verified)
        item["active_edges"] = active_edges(item["indices"], images, verified)
        item["mst_edges"] = maximum_spanning_tree(item["indices"], images, verified)["edges"]

    threshold_checks = {}
    for threshold in ROBUST_THRESHOLDS:
        hits = [item for item in connected_candidates if item["metrics"]["mst_bottleneck"] >= threshold]
        threshold_checks[str(threshold)] = {
            "exists": bool(hits),
            "best": compact_candidate(hits[0]) if hits else None,
        }

    recommendation = top_s9[0] if top_s9 else None
    return {
        "database": str(database_path),
        "image_count": len(images),
        "ordering": "COLMAP images.name sorted lexicographically; indices are 0-based in that order.",
        "edge_weight": "two_view_geometries.rows; missing or zero rows are treated as w_ij=0.",
        "raw_match_weight": "matches.rows, reported per pair when present.",
        "verified_edge_inlier_distribution": edge_distribution,
        "candidate_window": {
            "selected_half_width": selected_window,
            "attempted_half_widths": [report["window_half_width"] for report in search_reports],
            "targets": TARGET_INDICES_9,
            "windows": candidate_windows(len(images), TARGET_INDICES_9, selected_window),
        },
        "search_algorithm": {
            "name": "deterministic trajectory-window beam search plus exact nested subset enumeration",
            "beam_width": beam_width,
            "s9": (
                "One candidate is selected from each of the 9 target-position windows. "
                "Partial states are ranked deterministically by connected component count, "
                "partial span, MST bottleneck, MST total, total verified inliers, negative "
                "target deviation, then lexicographic indices. Final connected S9 candidates "
                "are ranked by coverage span, MST bottleneck, MST total, total verified "
                "inliers, target deviation, then indices."
            ),
            "nested": "For each retained S9, all C(9,6) S6 and all C(6,3) S3 subsets are enumerated exactly.",
        },
        "search_reports": search_reports,
        "threshold_checks": threshold_checks,
        "top5_s9": top_s9,
        "recommendation": recommendation,
        "constraints": {
            "modified_train_view_files": False,
            "ran_point_triangulator": False,
            "ran_matching": False,
            "used_full_points3D": False,
            "started_training": False,
        },
    }


def load_database(database_path):
    uri = "file:" + str(database_path) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    cur = con.cursor()
    images = [
        {"image_id": int(image_id), "name": name, "index": idx}
        for idx, (image_id, name) in enumerate(
            cur.execute("select image_id, name from images order by name").fetchall()
        )
    ]
    id_to_index = {image["image_id"]: image["index"] for image in images}
    raw_matches = {}
    for pair_id, rows in cur.execute("select pair_id, rows from matches"):
        a, b = image_ids_from_pair_id(pair_id)
        if a in id_to_index and b in id_to_index:
            key = tuple(sorted((id_to_index[a], id_to_index[b])))
            raw_matches[key] = int(rows)
    verified = {}
    for pair_id, rows in cur.execute("select pair_id, rows from two_view_geometries"):
        a, b = image_ids_from_pair_id(pair_id)
        if a in id_to_index and b in id_to_index:
            key = tuple(sorted((id_to_index[a], id_to_index[b])))
            verified[key] = {
                "verified_inliers": int(rows),
                "raw_matches": int(raw_matches.get(key, 0)),
            }
    con.close()
    return images, raw_matches, verified


def image_ids_from_pair_id(pair_id):
    image_id2 = int(pair_id % COLMAP_PAIR_ID_BASE)
    image_id1 = int((pair_id - image_id2) / COLMAP_PAIR_ID_BASE)
    return image_id1, image_id2


def candidate_windows(total, targets, half_width):
    windows = []
    for target in targets:
        start = max(0, target - half_width)
        end = min(total - 1, target + half_width)
        windows.append(list(range(start, end + 1)))
    return windows


def search_s9(images, verified, candidates_by_slot, beam_width, window_half_width):
    states = [()]
    metrics_cache = {}
    for slot, candidates in enumerate(candidates_by_slot):
        expanded = []
        for state in states:
            used = set(state)
            for candidate in candidates:
                if candidate not in used:
                    next_state = state + (candidate,)
                    expanded.append((partial_key(next_state, verified, metrics_cache), next_state))
        expanded.sort(key=lambda item: item[0], reverse=True)
        deduped = []
        seen = set()
        for _, state in expanded:
            sorted_state = tuple(sorted(state))
            if sorted_state in seen:
                continue
            seen.add(sorted_state)
            deduped.append(state)
            if len(deduped) >= beam_width:
                break
        states = deduped

    final = []
    for state in states:
        metrics = cached_graph_metrics(state, verified, metrics_cache)
        if metrics["connected"]:
            candidate = describe_split(state, images, verified)
            final.append(candidate)
    final.sort(key=s9_final_key, reverse=True)
    return final, {
        "window_half_width": window_half_width,
        "beam_width": beam_width,
        "final_beam_states": len(states),
        "connected_final_states": len(final),
        "cached_graph_states": len(metrics_cache),
    }


def partial_key(indices, verified, metrics_cache):
    metrics = cached_graph_metrics(indices, verified, metrics_cache)
    deviation = sum(abs(index - TARGET_INDICES_9[pos]) for pos, index in enumerate(indices))
    return (
        -metrics["component_count"],
        metrics["coverage"],
        metrics["mst_bottleneck"],
        metrics["mst_total_weight"],
        metrics["total_verified_inliers"],
        -deviation,
        tuple(-index for index in indices),
    )


def cached_graph_metrics(indices, verified, metrics_cache):
    key = tuple(sorted(indices))
    if key not in metrics_cache:
        metrics_cache[key] = graph_metrics(key, verified)
    return metrics_cache[key]


def s9_final_key(item):
    m = item["metrics"]
    return (
        m["coverage"],
        m["mst_bottleneck"],
        m["mst_total_weight"],
        m["total_verified_inliers"],
        -item["target_deviation"],
        tuple(-index for index in item["indices"]),
    )


def describe_split(indices, images, verified):
    indices = tuple(sorted(indices))
    metrics = graph_metrics(indices, verified)
    return {
        "indices": list(indices),
        "images": [images[index]["name"] for index in indices],
        "image_ids": [images[index]["image_id"] for index in indices],
        "target_deviation": sum(abs(index - target) for index, target in zip(indices, TARGET_INDICES_9[: len(indices)])),
        "metrics": metrics,
    }


def graph_metrics(indices, verified):
    indices = tuple(sorted(indices))
    degrees = {index: 0 for index in indices}
    total = 0
    active = 0
    for a, b in itertools.combinations(indices, 2):
        w = verified.get(tuple(sorted((a, b))), {}).get("verified_inliers", 0)
        if w > 0:
            active += 1
            total += w
            degrees[a] += 1
            degrees[b] += 1
    components = connected_components(indices, verified)
    mst = maximum_spanning_tree(indices, None, verified)
    span = max(indices) - min(indices) if indices else 0
    return {
        "connected": len(components) == 1 if indices else False,
        "component_count": len(components),
        "components": [list(component) for component in components],
        "coverage": span,
        "coverage_ratio": span / 250.0,
        "active_edges": active,
        "total_verified_inliers": total,
        "minimum_node_degree": min(degrees.values()) if degrees else 0,
        "maximum_node_degree": max(degrees.values()) if degrees else 0,
        "node_degrees": {str(index): degrees[index] for index in indices},
        "mst_bottleneck": mst["bottleneck"],
        "mst_total_weight": mst["total_weight"],
    }


def connected_components(indices, verified):
    remaining = set(indices)
    components = []
    while remaining:
        start = min(remaining)
        stack = [start]
        remaining.remove(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for other in list(remaining):
                if verified.get(tuple(sorted((node, other))), {}).get("verified_inliers", 0) > 0:
                    remaining.remove(other)
                    stack.append(other)
        components.append(tuple(sorted(component)))
    return sorted(components)


def maximum_spanning_tree(indices, images, verified):
    indices = tuple(sorted(indices))
    if not indices:
        return {"edges": [], "bottleneck": 0, "total_weight": 0}
    edges = []
    for a, b in itertools.combinations(indices, 2):
        w = verified.get(tuple(sorted((a, b))), {}).get("verified_inliers", 0)
        if w > 0:
            edges.append((w, a, b))
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    parent = {index: index for index in indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[rb] = ra
        record = {"index_a": a, "index_b": b, "verified_inliers": w}
        if images is not None:
            record.update(
                {
                    "image_a": images[a]["name"],
                    "image_b": images[b]["name"],
                    "image_id_a": images[a]["image_id"],
                    "image_id_b": images[b]["image_id"],
                }
            )
        tree.append(record)
        if len(tree) == len(indices) - 1:
            break
    connected = len(tree) == len(indices) - 1
    return {
        "edges": tree,
        "bottleneck": min((edge["verified_inliers"] for edge in tree), default=0) if connected else 0,
        "total_weight": sum(edge["verified_inliers"] for edge in tree) if connected else 0,
    }


def active_edges(indices, images, verified):
    rows = []
    for a, b in itertools.combinations(sorted(indices), 2):
        edge = verified.get(tuple(sorted((a, b))), {"verified_inliers": 0, "raw_matches": 0})
        if edge["verified_inliers"] > 0:
            rows.append(
                {
                    "index_a": a,
                    "index_b": b,
                    "image_a": images[a]["name"],
                    "image_b": images[b]["name"],
                    "image_id_a": images[a]["image_id"],
                    "image_id_b": images[b]["image_id"],
                    "raw_matches": edge["raw_matches"],
                    "verified_inliers": edge["verified_inliers"],
                }
            )
    rows.sort(key=lambda item: (-item["verified_inliers"], item["index_a"], item["index_b"]))
    return rows


def best_nested_splits(s9_indices, images, verified):
    s6_candidates = [describe_split(combo, images, verified) for combo in itertools.combinations(sorted(s9_indices), 6)]
    s6_candidates = [item for item in s6_candidates if item["metrics"]["connected"]]
    s6_candidates.sort(key=nested_key, reverse=True)
    best_s6 = s6_candidates[0] if s6_candidates else None
    if not best_s6:
        return {"s6": None, "s3": None}

    s3_candidates = [
        describe_split(combo, images, verified)
        for combo in itertools.combinations(best_s6["indices"], 3)
    ]
    s3_candidates = [item for item in s3_candidates if item["metrics"]["connected"]]
    s3_candidates.sort(key=nested_key, reverse=True)
    best_s3 = s3_candidates[0] if s3_candidates else None
    if best_s3:
        best_s3["all_pairs"] = all_pairs(best_s3["indices"], images, verified)
        best_s3["mst_edges"] = maximum_spanning_tree(best_s3["indices"], images, verified)["edges"]
    best_s6["active_edges"] = active_edges(best_s6["indices"], images, verified)
    best_s6["mst_edges"] = maximum_spanning_tree(best_s6["indices"], images, verified)["edges"]
    return {"s6": best_s6, "s3": best_s3}


def nested_key(item):
    m = item["metrics"]
    target_ratio = 0.85 if len(item["indices"]) == 6 else 0.70
    return (
        m["coverage_ratio"] >= target_ratio,
        m["mst_bottleneck"],
        m["coverage"],
        m["mst_total_weight"],
        m["total_verified_inliers"],
        -item["target_deviation"],
        tuple(-index for index in item["indices"]),
    )


def all_pairs(indices, images, verified):
    rows = []
    for a, b in itertools.combinations(sorted(indices), 2):
        edge = verified.get(tuple(sorted((a, b))), {"verified_inliers": 0, "raw_matches": 0})
        rows.append(
            {
                "index_a": a,
                "index_b": b,
                "image_a": images[a]["name"],
                "image_b": images[b]["name"],
                "image_id_a": images[a]["image_id"],
                "image_id_b": images[b]["image_id"],
                "raw_matches": edge["raw_matches"],
                "verified_inliers": edge["verified_inliers"],
            }
        )
    return rows


def compact_candidate(item):
    return {
        "indices": item["indices"],
        "images": item["images"],
        "coverage": item["metrics"]["coverage"],
        "coverage_ratio": item["metrics"]["coverage_ratio"],
        "mst_bottleneck": item["metrics"]["mst_bottleneck"],
        "mst_total_weight": item["metrics"]["mst_total_weight"],
        "total_verified_inliers": item["metrics"]["total_verified_inliers"],
    }


def distribution(values):
    values = sorted(values)
    if not values:
        return {}
    return {
        "count": len(values),
        "min": values[0],
        "q10": percentile(values, 10),
        "q25": percentile(values, 25),
        "median": percentile(values, 50),
        "q75": percentile(values, 75),
        "q90": percentile(values, 90),
        "max": values[-1],
    }


def percentile(values, pct):
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summary_for_console(audit):
    return {
        "image_count": audit["image_count"],
        "verified_edge_inlier_distribution": audit["verified_edge_inlier_distribution"],
        "candidate_window": audit["candidate_window"]["selected_half_width"],
        "search_reports": audit["search_reports"],
        "threshold_checks": audit["threshold_checks"],
        "top5_s9": [compact_candidate(item) for item in audit["top5_s9"]],
        "recommended": compact_candidate(audit["recommendation"]) if audit["recommendation"] else None,
        "constraints": audit["constraints"],
    }


if __name__ == "__main__":
    main()
