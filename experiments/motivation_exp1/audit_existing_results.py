"""Verify an archived E1 run and optionally refresh analysis without rerunning GPU jobs.

The baseline is captured BEFORE edits, while every recorded source hash matches.
Never use this command to relabel changed model/data or profiling implementations.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from .utils.data_utils import ROOT, digest, initialize, mark, read_csv, write_csv, write_json


def audit(archive):
    baseline = json.loads((archive / "audit_baseline.json").read_text())
    old = json.loads((archive / "run_metadata.json").read_text())
    if not baseline["all_recorded_source_hashes_matched_before_changes"]:
        raise ValueError("Original source identity was not established")
    for name, expected in baseline["artifact_sha256"].items():
        if digest(archive / name) != expected:
            raise ValueError(f"Archived artifact changed: {name}")
    protected = [p for p in old["sources"] if p.startswith(("llava/", "data_tools/")) or p.endswith("utils/profiling_utils.py")]
    for name in protected:
        if digest(ROOT / name) != old["sources"][name]:
            raise ValueError(f"Original inference/instrumentation source changed: {name}")
    for name, expected_functions in baseline["protected_function_hashes"].items():
        tree = ast.parse((ROOT / name).read_text())
        actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n, ast.FunctionDef)}
        if any(actual.get(n) != h for n, h in expected_functions.items()):
            raise ValueError(f"Reused artifact producer changed: {name}")
    cfg = old["config"]
    folders = {"model_files": Path(cfg["senna"]["checkpoint_path"]),
               "dataset_tables": Path(cfg["nuscenes"]["dataroot"]) / cfg["nuscenes"]["version"]}
    folders["vision_files"] = Path(json.loads((folders["model_files"] / "config.json").read_text())["mm_vision_tower"])
    for kind, folder in folders.items():
        for name, expected in old[kind].items():
            stat = (folder / name).stat()
            if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
                raise ValueError(f"Input identity changed: {folder / name}")
    if digest(cfg["senna"]["eval_json_path"]) != old["eval_json_path_sha256"]:
        raise ValueError("Evaluation workload changed")
    from importlib.metadata import version
    package_changes = {p: {"original": v, "current": version(p)} for p, v in old["packages"].items() if version(p) != v}
    if set(package_changes) - {"nuscenes-devkit"}:
        raise ValueError(f"Numerical/profiling dependency changed: {package_changes}")
    frames = read_csv(archive / "camera_manifest.csv")
    if not frames or any(not Path(r["image_path"]).is_file() for r in frames):
        raise ValueError("Indexed real images are missing")
    profile = read_csv(archive / "profiling_samples.csv")
    timing = [r for r in profile if r["measurement_mode"] == "timing"]
    max_error = 0.
    for row in profile:
        if row["timing_decomposition_valid"].lower() != "true" or row["llm_split_valid"].lower() != "true":
            raise ValueError("Archived decomposition invalid")
        parts = ["image_file_loading_ms", "image_preprocessing_ms", "host_to_device_ms", "vision_tower_ms", "projector_ms", "img_adapter_ms", "multimodal_preparation_ms", "llm_prefill_ms", "llm_decode_ms", "llm_generation_overhead_ms", "other_ms"]
        error = abs(sum(float(row[k]) for k in parts) - float(row["end_to_end_ms"]))
        max_error = max(error, max_error)
        calls = json.loads(row["llm_forward_calls"])
        if len(calls) != int(row["output_tokens"]) or sum(c["phase"] == "prefill" for c in calls) != 1:
            raise ValueError("Forward/token classification mismatch")
        for phase in ("prefill", "decode"):
            if abs(sum(c["duration_ms"] for c in calls if c["phase"] == phase) - float(row[f"llm_{phase}_ms"])) > 1e-4:
                raise ValueError("Forward/event accounting mismatch")
    if not timing or max_error > 1e-4:
        raise ValueError("No valid original timing measurements")
    equivalence = json.loads((archive / "output_equivalence.json").read_text())
    if not equivalence or not all(r["passed"] and r["original_ids"] == r["profiled_ids"] for r in equivalence):
        raise ValueError("Original/instrumented equivalence not established")
    return {"status": "passed", "archive": str(archive), "original_fingerprint": old["fingerprint"],
            "original_metadata_sha256": digest(archive / "run_metadata.json"),
            "archived_artifacts_verified": len(baseline["artifact_sha256"]),
            "inference_and_instrumentation_source_hashes_unchanged": protected,
            "reused_producer_function_hashes_unchanged": baseline["protected_function_hashes"],
            "dataset_and_model_size_mtime_identity_unchanged": True,
            "identity_limit": "Original provenance records table/model size and mtime, not full weight/table content SHA256. No stronger identity claim is made.",
            "package_changes": package_changes,
            "reuse_policy": "Preserve original SDK provenance for camera/object tables. Model timing uses unchanged numerical/profiling packages, sources, prompts and weights. Recompute pair selection, all image metrics, aggregations and figures in the current environment.",
            "timing_records": len(timing), "unique_profile_samples": len({r["sample_token"] for r in timing}),
            "max_partition_error_ms": max_error, "equivalence_samples": len(equivalence),
            "archived_artifact_sha256": baseline["artifact_sha256"]}


def refresh(archive, out, evidence):
    from .analyze_temporal_redundancy import run as similarity
    from .aggregate_results import run as aggregate
    from .build_nuscenes_manifest import make_pairs, PAIR_FIELDS
    from .plot_results import run as plot
    from .generate_report import run as report
    old = json.loads((archive / "run_metadata.json").read_text())
    cfg = old["config"]
    cfg["nuscenes"]["dataroot"] = str(Path(cfg["nuscenes"]["dataroot"]).resolve())
    cfg["output"]["directory"] = str(out)
    # Refuse to delete any modified/unrecognized top-level artifact. The immutable
    # archive remains under previous_runs, including the complete original report.
    for p in archive.iterdir():
        target = out / p.name
        if p.is_file() and p.name != "audit_baseline.json" and target.exists():
            if digest(target) != digest(p):
                raise ValueError(f"Output changed since archive; refusing refresh: {target}")
    for p in archive.iterdir():
        target = out / p.name
        if p.is_file() and p.name != "audit_baseline.json" and target.exists():
            target.unlink()
    args = argparse.Namespace(existing="overwrite")
    initialize(cfg, args)
    reused = ["camera_manifest.csv", "object_persistence.csv", "object_issues.json", "manifest_issues.json", "profiling_samples.csv", "matched_profile_pairs.csv", "matched_profile_issues.json", "model_metadata.json", "output_equivalence.json", "profile_status.json", "objects_status.json"]
    for name in reused:
        shutil.copy2(archive / name, out / name)
    for p in (archive / "profiler_trace").glob("*"):
        (out / "profiler_trace").mkdir(exist_ok=True)
        shutil.copy2(p, out / "profiler_trace" / p.name)
        reused.append("profiler_trace/" + p.name)
    evidence["reused_artifacts"] = {n: {"sha256": digest(out / n), "origin": str(archive / n)} for n in reused}
    write_json(out / "reuse_provenance.json", evidence)
    metadata = json.loads((out / "run_metadata.json").read_text())
    metadata["reused_artifacts"] = evidence["reused_artifacts"]
    write_json(out / "run_metadata.json", metadata)
    frames = read_csv(out / "camera_manifest.csv")
    pairs, candidates = make_pairs(frames, cfg)
    write_csv(out / "temporal_pairs.csv", pairs, PAIR_FIELDS)
    manifest = json.loads((archive / "manifest_summary.json").read_text())
    manifest.update(candidate_pairs=candidates, selected_pairs=len(pairs), selection="seeded scene/camera/gap strata; 0.083s adjacent camera links within tolerance; 0.5s successive keyframes only; real timestamps")
    write_json(out / "manifest_summary.json", manifest)
    mark(out, "index", frames=len(frames), pairs=len(pairs), camera_manifest_origin="verified archived official SDK index")
    for name, fn in (("redundancy", similarity), ("aggregate", aggregate), ("plot", plot), ("report", report)):
        fn(cfg, args, out)
        if name != "redundancy":
            mark(out, name)
    write_json(out / "invocation_status.json", {"failed_stages": [], "mode": "verified_derived_refresh", "profiling_reused": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/motivation_exp1")
    parser.add_argument("--refresh-derived", action="store_true")
    args = parser.parse_args()
    evidence = audit(args.archive.resolve())
    if args.refresh_derived:
        refresh(args.archive.resolve(), args.output_dir.resolve(), evidence)
    write_json(args.output_dir / "implementation_audit.json", evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k not in ("archived_artifact_sha256", "reused_artifacts")}, indent=2))


if __name__ == "__main__":
    main()
