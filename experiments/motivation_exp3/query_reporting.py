"""Scene-level statistics, E1-style figures, and an evidence-bounded report."""
from collections import defaultdict
import json
import os

import numpy as np

from experiments.motivation_exp1.utils.data_utils import write_csv, write_json
from experiments.motivation_exp1.utils.stats_utils import distribution, scene_bootstrap

STABILITY = ("answer_stable", "dependency_stable", "dependency_overlap", "same_target",
             "position_delta_m", "speed_delta_mps", "position_label_changed", "answer_same_dependency_changed",
             "oracle_safe_reuse", "stale_answer", "dependency_change", "state_information_unavailable")
TIMINGS = ("end_to_end_ms", "llm_prefill_ms", "llm_decode_ms", "llm_generation_ms", "generated_token_count", "strict_correct")


def number(value):
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def aggregate(rows, base, sequences, transitions, pairs, measured, cfg, out):
    def stats(records, key):
        return {**distribution([r.get(key) for r in records]),
                **scene_bootstrap(records, key, cfg["experiment"]["seed"], cfg["experiment"]["bootstrap_iterations"])}
    # Repeats are averaged within requests before scene-level aggregation.
    repeats = defaultdict(list)
    for r in measured:
        repeats[r["record_id"]].append(r)
    request_means = {}
    for record_id, records in repeats.items():
        mean = {k: records[0][k] for k in ("record_id", "scene_token", "sequence_id", "query_type")}
        for key in TIMINGS:
            values = [number(r.get(key)) for r in records]
            values = [v for v in values if v is not None and np.isfinite(v)]
            mean[key] = float(np.mean(values)) if len(values) == cfg["senna"]["profile_repeats"] else None
        request_means[record_id] = mean
    by_type, flat = {}, []
    for kind in cfg["experiment"]["query_types"]:
        pp = [p for p in pairs if p["query_type"] == kind]
        timed = [r for r in request_means.values() if r["query_type"] == kind]
        summary = {"unique_queries": sum(q["query_type"] == kind for q in base), "valid_adjacent_pairs": len(pp),
                   "profiled_query_events": len(timed), "metrics": {k: stats(pp, k) for k in STABILITY},
                   "inference": {k: stats(timed, k) for k in TIMINGS}}
        complete = [r for r in timed if all(r[k] is not None for k in ("end_to_end_ms", "llm_prefill_ms", "llm_decode_ms"))]
        summary["latency_breakdown"] = {k: stats(complete, k) for k in ("end_to_end_ms", "llm_prefill_ms", "llm_decode_ms")}
        by_type[kind] = summary
        row = {"query_type": kind, **{k: summary[k] for k in ("unique_queries", "valid_adjacent_pairs", "profiled_query_events")}}
        for group in (summary["metrics"], summary["inference"]):
            for key, value in group.items():
                row.update({key + "_pooled_mean": value["mean"], key + "_scene_mean": value["scene_equal_weight_mean"],
                            key + "_ci95": json.dumps(value["ci95"]), key + "_n": value["n"]})
        flat.append(row)
    sequence_rows = []
    for sequence in sequences:
        queries = [q for q in rows if q["sequence_id"] == sequence["sequence_id"]]
        profiled = [request_means[q["record_id"]] for q in queries if q["record_id"] in request_means]
        result = {**sequence, "profiled_query_events": len(profiled),
                  "annotation_complete": sequence["valid_queries"] == sequence["sequence_length"],
                  "all_valid_queries_measured": bool(queries) and len(profiled) == len(queries)}
        for key in ("end_to_end_ms", "llm_prefill_ms", "llm_decode_ms", "llm_generation_ms"):
            values = [p[key] for p in profiled]
            result["total_" + key] = sum(values) if result["all_valid_queries_measured"] and all(v is not None for v in values) else None
        result["total_llm_forward_ms"] = (result["total_llm_prefill_ms"] + result["total_llm_decode_ms"]
                                          if result["total_llm_prefill_ms"] is not None and result["total_llm_decode_ms"] is not None else None)
        tt = [t for t in transitions if t["sequence_id"] == sequence["sequence_id"]]
        result.update({key + "_count": sum(t[key] for t in tt) for key in
                       ("oracle_safe_reuse", "stale_answer", "dependency_change", "state_information_unavailable")})
        sequence_rows.append(result)
    workload = {}
    for name in cfg["experiment"]["workloads"]:
        tt = [t for t in transitions if t["workload"] == name]
        adjacent = [t for t in tt if t["has_adjacent_history"]]
        workload[name] = {"query_events": len(tt), "adjacent_query_pairs": len(adjacent),
                          "no_adjacent_history_count": len(tt) - len(adjacent),
                          "adjacent_pair_metrics": {key: stats(adjacent, key) for key in STABILITY},
                          "safe_fraction_all_query_events": sum(t["oracle_safe_reuse"] for t in tt) / len(tt) if tt else None,
                          "unavailable_count_all_events": sum(t["state_information_unavailable"] for t in tt)}
    build = json.loads((out / "query_build_summary.json").read_text())
    summary = {"status": "profile_complete" if len(measured) == len(rows) * cfg["senna"]["profile_repeats"] else "offline_or_partial_profile",
               "scenes": len(build["scenes"]), "sequence_count": len(sequences), "unique_sample_queries": len(base),
               "valid_query_events": len(rows), "canonical_adjacent_pairs": len(pairs), "profile_measurements": len(measured),
               "expected_profile_measurements": len(rows) * cfg["senna"]["profile_repeats"],
               "excluded_by_reason": build["excluded_by_reason"], "by_query_type": by_type, "workloads": workload,
               "counts": {key: sum(p[key] for p in pairs) for key in ("oracle_safe_reuse", "stale_answer", "dependency_change", "state_information_unavailable")},
               "statistics": "Pooled pair means plus equal-weight scene means; percentile bootstrap resamples scene means, not frames. CI unavailable with <2 scenes. Canonical pairs deduplicate overlapping sequence prefixes.",
               "oracle_interpretation": "Offline GT upper bound conditional on the selected discrete signatures, not implemented runtime reuse or a measured speedup. Motion/planning require oracle access to past GT future-derived labels; these are not observable current world facts. Numerical changes are reported separately; existence requires exact scope and instance-set stability.",
               "baseline_interpretation": "Every valid query event is independently executed through original Senna; no answer cache, fact answering or historical visual/KV reuse. Standard within-request autoregressive KV cache remains original.",
               "navigation_caveat": "Original offline_future_route_6s navigation is an explicit route condition; future ego/object trajectories and prediction labels never enter model_sample."}
    write_csv(out / "query_type_summary.csv", flat)
    write_csv(out / "sequence_summary.csv", sequence_rows)
    write_json(out / "aggregate_summary.json", summary)
    return summary, sequence_rows


def plot(summary, pairs, sequence_rows, cfg, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    names = list(summary["by_query_type"])
    labels = [n.replace("_", "\n") for n in names]
    metadata = {}
    def save(fig, name):
        fig.savefig(directory / (name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        metadata[name] = {"png_dpi": 300, "pdf": "vector", "statistics": "scene-equal-weight means with scene-bootstrap CI where available"}
    def bars(metrics, name, title):
        fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
        width = .75 / len(metrics)
        for j, (key, label) in enumerate(metrics):
            for i, n in enumerate(names):
                stat = summary["by_query_type"][n]["metrics"][key]
                x = i + (j - (len(metrics) - 1) / 2) * width
                if stat["scene_equal_weight_mean"] is not None:
                    ax.bar(x, stat["scene_equal_weight_mean"], width=width, color=f"C{j}", label=label if i == 0 else None)
                    if stat["ci95"]:
                        ax.vlines(x, *stat["ci95"], color="black", linewidth=1)
                else:
                    ax.text(x, .02, "N/A", ha="center", fontsize=7)
        ax.set_xticks(range(len(names)), labels)
        ax.set(ylim=(0, 1.12), ylabel="Fraction (scene mean)", title=title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        save(fig, name)
    bars([("answer_stable", "Answer")], "answer_stability", "Ground-truth Answer Stability")
    bars([("dependency_stable", "Dependency")], "dependency_stability", "Ground-truth Dependency Stability")
    bars([("answer_stable", "Answer"), ("dependency_stable", "Dependency")], "answer_vs_dependency", "Answer and Dependency Stability Are Different")
    bars([("oracle_safe_reuse", "GT Oracle")], "oracle_safe_reuse", "Offline GT Oracle Opportunity; No Runtime Speedup Claimed")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for n in names:
        values = [p["dependency_overlap"] for p in pairs if p["query_type"] == n and p["dependency_overlap"] is not None]
        if values:
            ax.hist(values, bins=np.linspace(0, 1, 12), histtype="step", linewidth=1.5, label=n, weights=np.ones(len(values)) / len(values))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.set(xlabel="Instance-set Jaccard (empty existence sets = 1)", ylabel="Fraction of valid pairs", title="Dependency Overlap; Planning Has No Object Set")
    save(fig, "dependency_overlap")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    bottom = np.zeros(len(names))
    e2e = [summary["by_query_type"][n]["latency_breakdown"]["end_to_end_ms"]["scene_equal_weight_mean"] for n in names]
    prefill = [summary["by_query_type"][n]["latency_breakdown"]["llm_prefill_ms"]["scene_equal_weight_mean"] for n in names]
    decode = [summary["by_query_type"][n]["latency_breakdown"]["llm_decode_ms"]["scene_equal_weight_mean"] for n in names]
    available = all(v is not None for v in e2e + prefill + decode)
    if available:
        for values, label in ((prefill, "LLM Prefill"), (decode, "LLM Decode"),
                              ([e - p - d for e, p, d in zip(e2e, prefill, decode)], "Visual path, I/O and other")):
            ax.bar(range(len(names)), values, bottom=bottom, label=label)
            bottom += np.array(values)
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "Real GPU timings unavailable or incomplete", ha="center", transform=ax.transAxes)
    ax.set_xticks(range(len(names)), labels)
    ax.set(ylabel="Measured latency (ms)", title="Original Senna End-to-End Latency Breakdown")
    save(fig, "latency_breakdown")
    metadata["latency_breakdown"]["available"] = available
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for workload in cfg["experiment"]["workloads"]:
        for kind in names if workload == "fixed" else ["alternating"]:
            xs, ys = [], []
            for length in cfg["experiment"]["sequence_lengths"]:
                group = [r for r in sequence_rows if r["workload"] == workload and r["query_type"] == kind
                         and r["sequence_length"] == length and r["annotation_complete"] and r["total_llm_forward_ms"] is not None]
                stat = scene_bootstrap(group, "total_llm_forward_ms", cfg["experiment"]["seed"], cfg["experiment"]["bootstrap_iterations"])
                if stat["scene_equal_weight_mean"] is not None:
                    xs.append(length)
                    ys.append(stat["scene_equal_weight_mean"])
                    if stat["ci95"]:
                        ax.vlines(length, *stat["ci95"], alpha=.35)
            if xs:
                ax.plot(xs, ys, "o-", label=workload + ": " + kind)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    else:
        ax.text(.5, .5, "No fully measured, annotation-complete sequences", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Sequence length (real keyframes)", ylabel="Cumulative measured Prefill + Decode (ms)", title="Independent Original Senna Execution Cost")
    save(fig, "cumulative_llm_cost")
    write_json(out / "query_figure_metadata.json", metadata)


def report(summary, cfg, out, error=None):
    def fmt(value, percentage=False):
        return "未测得" if value is None else (f"{100 * value:.2f}%" if percentage else f"{value:.3f}")
    lines = ["# Experiment 3 — Temporal Redundancy in Semantic Query Execution", "",
             f"状态：`{summary['status']}`。采样 {summary['scenes']} 个官方 val 场景，{summary['sequence_count']} 条序列，"
             f"{summary['valid_query_events']} 次有效查询事件，{summary['unique_sample_queries']} 个不同 sample/模板，"
             f"{summary['canonical_adjacent_pairs']} 个去重后的有效相邻同模板查询对。",
             f"原始 Senna 正式测量：{summary['profile_measurements']}/{summary['expected_profile_measurements']}；warmup 与等价性请求不计入。", "",
             "## 构建与真值来源", "",
             "- Category/Position：官方 sample_annotation，以当前 LiDAR 坐标筛选前方扇区内最近车辆，按 instance token 打破距离平局；类别直接来自 category_name，位置由固定横向边界离散。每帧重新选择，明确记录目标切换。",
             "- Existence：沿用 VRU Identification 问题结构，以官方标注中的行人、自行车、摩托车及固定前方空间范围判定；空集是真实负例，缺失或损坏标注不能当成负例。",
             "- Motion Prediction：沿用原始 Motion Prediction 的 SPEED/PATH 任务与 get_obj_acc_or_dec、get_obj_turn_or_lane_change；要求同实例完整连续六步未来标注，保留未来位置、真实时间戳和标签。输入速度仅由过去标注计算，不使用原始构造代码中的未来位移速度。目标位置与速度是当前 GT 条件，非纯视觉检测结果。",
             "- Meta-action Planning：直接调用 planning_question/current_speed/navigation_command，使用原始三秒未来自车轨迹标签及 parse_plan。原始六秒未来路线仅作为明确标记的离线导航条件，不能据此宣称在线无未来信息的规划表现。",
             "- 全部对象、场景、sample、annotation token、六视角路径、输入协议、离散规则和连续属性可在 query_traces.csv/query_dependencies.csv/queries.json 中追溯。未来数据只存在离线标签字段，模型输入由 model_sample 的字段白名单构建。", "",
             "## 统计口径", "",
             "同场景 sample.prev/next 建立连续序列，取每场景第 1 个关键帧开始的固定前缀（跳过第 0 帧以支持过去速度）。固定模板与具体输入文本分别记录：运动提示的当前数值、规划的速度和导航会随帧变化。Fixed 指固定问题模板；Category/Existence/Position 的问题文本也完全相同。",
             "相邻答案和依赖统计基于 base_queries 去重，不重复计算不同长度的重叠前缀。过滤后不跨帧连线。图中为场景等权均值，CSV 同时给出 pooled 比例和 scene Bootstrap 95% CI；少于两个场景不给 CI。", "",
             "| 查询类型 | 有效相邻对 | Answer 稳定 | Dependency 稳定 | Oracle 安全机会 |", "|---|---:|---:|---:|---:|"]
    for kind, r in summary["by_query_type"].items():
        values = [fmt(r["metrics"][k]["scene_equal_weight_mean"], True) for k in ("answer_stable", "dependency_stable", "oracle_safe_reuse")]
        lines.append(f"| {kind} | {r['valid_adjacent_pairs']} | " + " | ".join(values) + " |")
    lines += ["", "## Oracle 的含义与边界", "",
              "Oracle 仅查看上一时刻查询暴露的事实，检查当前所需事实是否齐全、历史依赖仍有效且可得到相同答案。Existence 要求作用域及实例集合完全一致；Category/Position 同时验证最近目标身份。数值位移、速度差和离散属性变化另行保留。Mixed 只使用紧邻上一条查询的事实，不累积整段完整世界状态。",
              "Motion/Planning 的 Oracle 使用离线可知的历史预测/规划 GT 标签，因此是包含预测标签的乐观 GT 上界；这些标签不是可在线观测的当前环境事实。稳定的离散签名也不表示所有连续属性完全不变。不能把这类机会等同于可部署的事实复用。",
              "分类计数如下（不是互斥分类；依赖改变可能同时发生旧答案错误）：", "", f"```json\n{json.dumps(summary['counts'], ensure_ascii=False, indent=2)}\n```", "",
              "Oracle、原始 Senna 回答和 WorldStream 能力严格分开：本实验未实现 QCSE、事实直答、答案缓存、历史视觉 Token/KV 复用或运行时世界状态维护。所有正式请求仍执行原始 Senna；Oracle 不是实测加速收益。", "",
              "## 原始 Senna 实测", "",
              "| 查询类型 | 请求数 | Prefill ms | Decode ms | E2E ms | 输出 token | 严格正确率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for kind, r in summary["by_query_type"].items():
        # TIMINGS order begins E2E; present columns in header order.
        vals = [fmt(r["inference"][k]["scene_equal_weight_mean"], k == "strict_correct") for k in
                ("llm_prefill_ms", "llm_decode_ms", "end_to_end_ms", "generated_token_count", "strict_correct")]
        lines.append(f"| {kind} | {r['profiled_query_events']} | " + " | ".join(vals) + " |")
    lines += ["", "Prefill/Decode 复用 Experiment 1 CUDA forward 区间：Prefill 产生首 token，Decode 为后续带 KV 的单 token forward。CUDA 区间包含 dispatch/stream 空闲，不是纯 kernel 时间；generation overhead、图像编码、I/O 等包含在 E2E 的其他部分。每个真实序列分别执行，不从其他长度或负载借用计时。",
              "每个序列的成本只在所有有效请求均完成测量时求和；注释缺失造成的不完整序列明确标记，累计成本图仅展示 annotation_complete 的序列。真实预测、标签和输出 token 存于 query_profile_samples.csv。", "",
              "硬件负载、版本、模型配置与原始 generation 参数见 query_model_metadata.json；逐类型原始/插桩输入哈希、输出 token 和回答等价性见 query_output_equivalence.json。",
              f"本次测量条件：{cfg.get('measurement_context', '未执行 GPU 测量；以 metadata 为准')}。", "",
              "## 过滤与验证", "", f"```json\n{json.dumps(summary['excluded_by_reason'], ensure_ascii=False, indent=2)}\n```", "",
              "测试执行记录单独保存在 validation.json（如已执行）。Smoke 数据位于独立输出目录，不能替代正式结果。"]
    if error:
        lines += ["", f"尚未完成的运行步骤：`{error}`。未测得的时延保持缺失，不填入估计值。"]
    lines += ["", "## 直接运行", "", "```bash", "cd /data/qh/phdCode/work4/Senna",
              "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\",
              "/data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp3.run_queries \\",
              "  --config experiments/motivation_exp3/config.yaml --mode all --existing resume --wait-for-idle", "```", "",
              "采样参数见 resolved_query_config.yaml。完整交付包括八种要求的 CSV/JSON/报告，以及七组 PNG 300 DPI 和 PDF 图。"]
    (out / "experiment_report.md").write_text("\n".join(lines) + "\n")
