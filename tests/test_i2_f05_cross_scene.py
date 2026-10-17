import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("f05", ROOT / "tools" / "analyze_i2_f05_cross_scene.py")
F05 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(F05)


def test_quantiles_are_deterministic():
    result = F05.quantiles([4, 1, 3, 2])
    assert result == {"n": 4, "min": 1.0, "p25": 1.75, "median": 2.5, "p75": 3.25, "max": 4.0, "mean": 2.5}


def test_discovery_uses_metadata_and_finds_unique_formal_arms():
    for scene in F05.SCENES:
        cfg = F05.read_json(ROOT / "configs" / "i2_scenes" / f"{scene}.json")
        arms, _ = F05.discover_arms(ROOT, cfg)
        assert set(arms) == set(F05.ROLES)
        for path in arms.values():
            assert (path / "formal_metrics_5000.json").is_file()


def test_full_analysis_writes_new_complete_report(tmp_path):
    output, audit = F05.run(ROOT, tmp_path / "f05")
    assert audit["final_verdict"]["f05"] == "GO"
    assert audit["final_verdict"]["adaptive_k_readiness"] == "NOT_READY"
    assert audit["resource_matching"]["D"].startswith("UNKNOWN")
    assert audit["resource_matching"]["R"].startswith("UNKNOWN")
    for name in ("f05_audit.json", "f05_scene_comparison.csv", "f05_capacity_comparison.csv", "f05_summary.md"):
        assert (output / name).is_file()
    saved = json.loads((output / "f05_audit.json").read_text(encoding="utf-8"))
    for scene in F05.SCENES:
        data = saved["scenes"][scene]
        assert data["donor"]["actual_released_count"] == 20
        assert data["receiver"]["actual_added_count"] == 20
        assert data["integrity_checks"]["same_donor_set_release_receiver"]["status"] == "PASS"
        assert data["integrity_checks"]["final_checkpoint_audits"]["status"] == "PASS"
        assert all(len(row["trend"]) == 5 for row in data["quality"].values())
    with (output / "f05_capacity_comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["D"] for row in rows} == {"UNKNOWN"}
    assert {row["R"] for row in rows} == {"UNKNOWN"}


def test_existing_output_is_versioned_not_overwritten(tmp_path):
    base = tmp_path / "f05"
    base.mkdir()
    sentinel = base / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    chosen = F05.choose_output(ROOT, base)
    assert chosen != base
    assert not chosen.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
