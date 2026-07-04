from __future__ import annotations

import argparse
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.coded_learning_exp.network_runtime import NetworkExperimentConfig, load_problem
from src.coded_learning_exp.multiprocess_runtime import _make_strategy_specs, _make_worker_states


GUARD_TO_CANDIDATE = {
    "online_counter_guard_rank_aware_sparse_flexible": "rank_aware_sparse_flexible",
    "online_counter_guard_deadline_aware_sparse_flexible": "deadline_aware_sparse_flexible",
}

GUARD_LABEL = {
    "online_counter_guard_rank_aware_sparse_flexible": "Guard-R",
    "online_counter_guard_deadline_aware_sparse_flexible": "Guard-D",
}


@dataclass(frozen=True)
class Thresholds:
    label: str
    theta_cv: float
    theta_g: float
    theta_k: float
    theta_a: float


DEFAULT_THRESHOLDS = Thresholds("default", 0.20, 0.01, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Algorithm-3 guard prediction and threshold-sensitivity diagnostics "
            "from paired direct-K3s worker-service logs."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("majorrev_k8s_diagnostics"))
    parser.add_argument("--out", type=Path, default=Path("guard_prediction_diagnostics"))
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=None,
        help="Optional paper figure directory for guard_threshold_sensitivity.{pdf,png}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_round = collect_per_round(args.root)
    if per_round.empty:
        raise SystemExit(f"No guard rounds found under {args.root}")

    prediction = summarize_prediction_accuracy(per_round)
    sensitivity = summarize_threshold_sensitivity(per_round)
    resources = summarize_resource_counters(args.root)
    live_resources, live_by_workers = summarize_live_resource_snapshots(args.root)

    per_round.to_csv(args.out / "guard_prediction_per_round.csv", index=False)
    prediction.to_csv(args.out / "guard_prediction_accuracy.csv", index=False)
    sensitivity.to_csv(args.out / "guard_threshold_sensitivity.csv", index=False)
    resources.to_csv(args.out / "k8s_resource_counters.csv", index=False)
    live_resources.to_csv(args.out / "k8s_live_resource_snapshots.csv", index=False)
    live_by_workers.to_csv(args.out / "k8s_live_resource_by_workers.csv", index=False)

    write_latex_tables(prediction, sensitivity, resources, args.out / "guard_prediction_tables.tex")
    write_threshold_sensitivity_figure(sensitivity, args.out, args.paper_figures_dir)
    write_report(
        prediction,
        sensitivity,
        resources,
        live_by_workers,
        args.out / "guard_prediction_report.md",
    )
    print(f"Wrote guard diagnostics to {args.out}")


def collect_per_round(root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metrics_path in sorted(root.glob("**/network_metrics.csv")):
        run_dir = metrics_path.parent
        metrics = pd.read_csv(metrics_path)
        if not set(GUARD_TO_CANDIDATE).intersection(set(metrics["strategy"].astype(str))):
            continue
        run_args = parse_run_args(run_dir)
        problem = load_problem(run_dir / "problem")
        config = NetworkExperimentConfig(
            n_samples=int(run_args.get("samples", metrics["n_samples"].iloc[0])),
            n_features=int(run_args.get("features", metrics["n_features"].iloc[0])),
            density=float(run_args.get("density", metrics["density"].iloc[0])),
            n_shards=int(run_args.get("shards", metrics["n_shards"].iloc[0])),
            n_workers=int(run_args.get("workers", metrics["n_workers"].iloc[0])),
            rounds=int(run_args.get("rounds", metrics["iteration"].max() + 1)),
            learning_rate=float(run_args.get("learning-rate", 0.25)),
            l2=float(run_args.get("l2", 1e-3)),
            scenario=str(run_args.get("scenario", metrics["scenario"].iloc[0])),
            drift_period=int(run_args.get("drift-period", 4)),
            straggler_fraction=float(run_args.get("straggler-fraction", 0.35)),
            straggler_slowdown=float(run_args.get("straggler-slowdown", 0.12)),
            burst_probability=float(run_args.get("burst-probability", 0.45)),
            seed=int(run_args.get("seed", parse_seed_from_path(run_dir))),
            sleep_scale=float(run_args.get("sleep-scale", 0.01)),
            cost_scale=float(run_args.get("cost-scale", 0.002)),
            cancel_poll_seconds=float(run_args.get("cancel-poll-seconds", 0.003)),
            network_rtt_seconds=float(run_args.get("network-rtt-ms", 0.0)) / 1000.0,
            network_bandwidth_mbps=float(run_args.get("network-bandwidth-mbps", 0.0)),
            alignment_mode=str(run_args.get("alignment-mode", metrics["alignment_mode"].iloc[0])),
        )
        worker_states = _make_worker_states(config)
        specs = _make_strategy_specs(problem, config)
        by_key = {
            (int(row.iteration), str(row.strategy)): row
            for row in metrics.itertuples(index=False)
        }
        workers = int(config.n_workers)
        seed = int(config.seed)
        for guard_strategy, candidate_strategy in GUARD_TO_CANDIDATE.items():
            guard_spec = specs[guard_strategy]
            for iteration in sorted(metrics.loc[metrics["strategy"] == guard_strategy, "iteration"].unique()):
                iteration = int(iteration)
                guard_row = by_key.get((iteration, guard_strategy))
                static_row = by_key.get((iteration, "sparse_flexible_static"))
                candidate_row = by_key.get((iteration, candidate_strategy))
                if guard_row is None or static_row is None or candidate_row is None:
                    continue
                worker_state = worker_states[iteration]
                static_payload = guard_spec.static_builder(worker_state)
                candidate_payload = guard_spec.candidate_builder(worker_state)
                pred_static_tau, pred_static_prefix = guard_spec._predict(static_payload, worker_state)
                pred_candidate_tau, pred_candidate_prefix = guard_spec._predict(
                    candidate_payload, worker_state
                )
                predicted_gain = (
                    (pred_static_tau - pred_candidate_tau) / max(pred_static_tau, 1e-12)
                    if np.isfinite(pred_static_tau)
                    else 0.0
                )
                realized_gain = (
                    (float(static_row.decode_latency) - float(candidate_row.decode_latency))
                    / max(float(static_row.decode_latency), 1e-12)
                )
                realized_barrier_gain = (
                    (float(static_row.barrier_latency) - float(candidate_row.barrier_latency))
                    / max(float(static_row.barrier_latency), 1e-12)
                )
                observed_action = "enable" if "online-guard-enable" in str(guard_row.config) else "fallback"
                speed_mean = max(float(worker_state.speeds.mean()), 1e-12)
                speed_cv = float(worker_state.speeds.std() / speed_mean)
                realized_prefix_delta = int(candidate_row.selected_rows) - int(static_row.selected_rows)
                beneficial = (
                    realized_gain >= DEFAULT_THRESHOLDS.theta_g
                    and realized_prefix_delta <= DEFAULT_THRESHOLDS.theta_k
                    and float(candidate_row.rows_after_decode) <= DEFAULT_THRESHOLDS.theta_a
                )
                records.append(
                    {
                        "run": run_dir.name,
                        "workers": workers,
                        "seed": seed,
                        "iteration": iteration,
                        "guard": guard_strategy,
                        "guard_label": GUARD_LABEL[guard_strategy],
                        "candidate": candidate_strategy,
                        "observed_action": observed_action,
                        "speed_cv": speed_cv,
                        "pred_static_tau_ms": 1000.0 * pred_static_tau,
                        "pred_candidate_tau_ms": 1000.0 * pred_candidate_tau,
                        "pred_gain_pct": 100.0 * predicted_gain,
                        "pred_prefix_delta": int(pred_candidate_prefix) - int(pred_static_prefix),
                        "static_decode_ms": 1000.0 * float(static_row.decode_latency),
                        "candidate_decode_ms": 1000.0 * float(candidate_row.decode_latency),
                        "candidate_barrier_ms": 1000.0 * float(candidate_row.barrier_latency),
                        "static_barrier_ms": 1000.0 * float(static_row.barrier_latency),
                        "guard_decode_ms": 1000.0 * float(guard_row.decode_latency),
                        "guard_barrier_ms": 1000.0 * float(guard_row.barrier_latency),
                        "tau_error_ms": 1000.0
                        * abs(pred_candidate_tau - float(candidate_row.decode_latency)),
                        "prefix_error_rows": abs(
                            (int(pred_candidate_prefix) - int(pred_static_prefix))
                            - realized_prefix_delta
                        ),
                        "realized_gain_pct": 100.0 * realized_gain,
                        "realized_barrier_gain_pct": 100.0 * realized_barrier_gain,
                        "realized_prefix_delta": realized_prefix_delta,
                        "candidate_rows_after_decode": float(candidate_row.rows_after_decode),
                        "beneficial": bool(beneficial),
                        "false_enable": bool(observed_action == "enable" and not beneficial),
                        "false_disable": bool(observed_action == "fallback" and beneficial),
                    }
                )
    return pd.DataFrame.from_records(records)


def parse_run_args(run_dir: Path) -> dict[str, object]:
    manifest = run_dir / "k8s_master_job.yaml"
    if not manifest.exists():
        return {}
    text = manifest.read_text(encoding="utf-8", errors="replace")
    marker = "python -m src.coded_learning_exp.direct_docker_master"
    start = text.find(marker)
    if start < 0:
        return {}
    line = text[start:].splitlines()[0].strip()
    tokens = shlex.split(line)
    parsed: dict[str, object] = {}
    flags_with_values = {
        "problem-dir",
        "out",
        "workers",
        "worker-hosts",
        "worker-port",
        "samples",
        "features",
        "density",
        "shards",
        "rounds",
        "learning-rate",
        "l2",
        "scenario",
        "drift-period",
        "straggler-fraction",
        "straggler-slowdown",
        "burst-probability",
        "sleep-scale",
        "cost-scale",
        "cancel-poll-seconds",
        "network-rtt-ms",
        "network-bandwidth-mbps",
        "seed",
        "alignment-mode",
    }
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--"):
            key = token[2:]
            if key in flags_with_values and idx + 1 < len(tokens):
                parsed[key] = tokens[idx + 1]
                idx += 2
                continue
            parsed[key] = True
        idx += 1
    return parsed


def parse_seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        return 17
    return int(match.group(1))


def summarize_prediction_accuracy(per_round: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_round.groupby(["guard_label", "workers"], as_index=False)
        .agg(
            rounds=("tau_error_ms", "size"),
            tau_error_ms=("tau_error_ms", "mean"),
            p95_tau_error_ms=("tau_error_ms", lambda x: float(np.percentile(x, 95))),
            prefix_error_rows=("prefix_error_rows", "mean"),
            false_enable_rate=("false_enable", "mean"),
            false_disable_rate=("false_disable", "mean"),
            enable_rate=("observed_action", lambda x: float((x == "enable").mean())),
        )
        .sort_values(["guard_label", "workers"])
    )
    overall = (
        per_round.groupby(["guard_label"], as_index=False)
        .agg(
            workers=("workers", lambda _: "all"),
            rounds=("tau_error_ms", "size"),
            tau_error_ms=("tau_error_ms", "mean"),
            p95_tau_error_ms=("tau_error_ms", lambda x: float(np.percentile(x, 95))),
            prefix_error_rows=("prefix_error_rows", "mean"),
            false_enable_rate=("false_enable", "mean"),
            false_disable_rate=("false_disable", "mean"),
            enable_rate=("observed_action", lambda x: float((x == "enable").mean())),
        )
    )
    return pd.concat([grouped, overall], ignore_index=True)


def summarize_threshold_sensitivity(per_round: pd.DataFrame) -> pd.DataFrame:
    thresholds = [
        Thresholds("low $\\theta_{cv}$", 0.10, 0.01, 0.0, 1.0),
        DEFAULT_THRESHOLDS,
        Thresholds("high $\\theta_{cv}$", 0.35, 0.01, 0.0, 1.0),
        Thresholds("zero $\\theta_g$", 0.20, 0.00, 0.0, 1.0),
        Thresholds("high $\\theta_g$", 0.20, 0.03, 0.0, 1.0),
        Thresholds("strict $\\theta_K$", 0.20, 0.01, -1.0, 1.0),
        Thresholds("loose $\\theta_K$", 0.20, 0.01, 1.0, 1.0),
        Thresholds("strict $\\theta_a$", 0.20, 0.01, 0.0, 0.0),
        Thresholds("loose $\\theta_a$", 0.20, 0.01, 0.0, 2.0),
    ]
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        for guard_label in sorted(per_round["guard_label"].unique()):
            sub = per_round[per_round["guard_label"] == guard_label].copy()
            replay = replay_threshold_policy(sub, threshold)
            rows.append(
                {
                    "setting": threshold.label,
                    "guard_label": guard_label,
                    "theta_cv": threshold.theta_cv,
                    "theta_g": threshold.theta_g,
                    "theta_K": threshold.theta_k,
                    "theta_a": threshold.theta_a,
                    "enable_rate": replay["enable"].mean(),
                    "false_enable_rate": replay["false_enable"].mean(),
                    "false_disable_rate": replay["false_disable"].mean(),
                    "mean_barrier_gain_pct": replay["barrier_gain_pct"].mean(),
                    "p95_barrier_gain_pct": replay["p95_barrier_gain_pct"].iloc[0],
                }
            )
    return pd.DataFrame.from_records(rows)


def replay_threshold_policy(sub: pd.DataFrame, threshold: Thresholds) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = sub.sort_values(["workers", "seed", "iteration"]).copy()
    for _, group in ordered.groupby(["workers", "seed"], sort=False):
        ema_prefix_delta = 0.0
        ema_rows_after_decode = 0.0
        for row in group.itertuples(index=False):
            history_ok = (
                ema_prefix_delta <= threshold.theta_k
                and ema_rows_after_decode <= threshold.theta_a
            )
            enable = (
                float(row.speed_cv) >= threshold.theta_cv
                and float(row.pred_gain_pct) / 100.0 >= threshold.theta_g
                and float(row.pred_prefix_delta) <= threshold.theta_k
                and history_ok
            )
            beneficial = (
                float(row.realized_gain_pct) / 100.0 >= threshold.theta_g
                and float(row.realized_prefix_delta) <= threshold.theta_k
                and float(row.candidate_rows_after_decode) <= threshold.theta_a
            )
            chosen_barrier = float(row.candidate_barrier_ms) if enable else float(row.static_barrier_ms)
            static_barrier = max(float(row.static_barrier_ms), 1e-12)
            barrier_gain = 100.0 * (static_barrier - chosen_barrier) / static_barrier
            if enable:
                ema_prefix_delta = (
                    0.70 * ema_prefix_delta
                    + 0.30 * (float(row.realized_prefix_delta))
                )
                ema_rows_after_decode = (
                    0.70 * ema_rows_after_decode
                    + 0.30 * float(row.candidate_rows_after_decode)
                )
            else:
                ema_prefix_delta *= 0.80
                ema_rows_after_decode *= 0.80
            records.append(
                {
                    "enable": bool(enable),
                    "false_enable": bool(enable and not beneficial),
                    "false_disable": bool((not enable) and beneficial),
                    "barrier_gain_pct": barrier_gain,
                }
            )
    out = pd.DataFrame.from_records(records)
    out["p95_barrier_gain_pct"] = float(np.percentile(out["barrier_gain_pct"], 5))
    return out


def summarize_resource_counters(root: Path) -> pd.DataFrame:
    frames = []
    for summary_path in sorted(root.glob("**/network_summary.csv")):
        run_dir = summary_path.parent
        match = re.search(r"_w(\d+)_seed(\d+)", run_dir.name)
        df = pd.read_csv(summary_path)
        df["workers"] = int(match.group(1)) if match else int(df.get("n_workers", 0).iloc[0])
        df["seed"] = int(match.group(2)) if match else parse_seed_from_path(run_dir)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_summary = pd.concat(frames, ignore_index=True)
    keep = all_summary[
        all_summary["strategy"].isin(
            [
                "sparse_flexible_static",
                "rank_aware_sparse_flexible",
                "deadline_aware_sparse_flexible",
                "online_counter_guard_deadline_aware_sparse_flexible",
            ]
        )
    ].copy()
    labels = {
        "sparse_flexible_static": "Static",
        "rank_aware_sparse_flexible": "Rank",
        "deadline_aware_sparse_flexible": "Deadline",
        "online_counter_guard_deadline_aware_sparse_flexible": "Guard-D",
    }
    keep["policy"] = keep["strategy"].map(labels)
    out = (
        keep.groupby("policy", as_index=False)
        .agg(
            dispatch_ms=("mean_dispatch_seconds", lambda x: 1000.0 * float(x.mean())),
            cancel_ms=("mean_cancel_seconds", lambda x: 1000.0 * float(x.mean())),
            response_kb=("mean_network_response_mb", lambda x: 1000.0 * float(x.mean())),
            worker_errors=("mean_worker_errors", "mean"),
        )
        .sort_values("policy")
    )
    return out


def summarize_live_resource_snapshots(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted(root.glob("**/k8s_resource_counters.csv")):
        run_dir = path.parent
        match = re.search(r"_w(\d+)_seed(\d+)", run_dir.name)
        frame = pd.read_csv(path)
        frame["run"] = run_dir.name
        frame["workers"] = int(match.group(1)) if match else np.nan
        frame["seed"] = int(match.group(2)) if match else parse_seed_from_path(run_dir)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    snapshots = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "pod_count",
        "worker_pods",
        "master_pods",
        "running_pods",
        "succeeded_pods",
        "failed_pods",
        "restart_count",
        "pod_node_count",
        "top_pod_cpu_mean_mcores",
        "top_pod_cpu_p95_mcores",
        "top_pod_mem_mean_mib",
        "top_pod_mem_p95_mib",
        "top_node_cpu_mean_mcores",
        "top_node_cpu_p95_mcores",
        "top_node_mem_mean_mib",
        "top_node_mem_p95_mib",
        "stats_pod_cpu_mean_mcores",
        "stats_pod_cpu_p95_mcores",
        "stats_pod_mem_mean_mib",
        "stats_pod_mem_p95_mib",
        "network_worker_errors_sum",
        "network_dispatch_ms_mean",
        "network_cancel_ms_mean",
        "network_response_kb_mean",
        "network_decode_success_rate",
    ]
    for col in numeric_cols:
        if col in snapshots.columns:
            snapshots[col] = pd.to_numeric(snapshots[col], errors="coerce")
    available = [col for col in numeric_cols if col in snapshots.columns]
    by_workers = (
        snapshots.groupby("workers", as_index=False)
        .agg({col: "mean" for col in available})
        .sort_values("workers")
    )
    by_workers["n_resource_snapshots"] = (
        snapshots.groupby("workers")["run"].nunique().reindex(by_workers["workers"]).to_numpy()
    )
    return snapshots, by_workers


def write_latex_tables(
    prediction: pd.DataFrame,
    sensitivity: pd.DataFrame,
    resources: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "% Auto-generated by analyze_guard_prediction.py; paste selected rows into paper/sections/evaluation.tex.",
        latex_prediction_table(prediction),
        "",
        latex_sensitivity_table(sensitivity),
        "",
        latex_resource_table(resources),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_prediction_table(prediction: pd.DataFrame) -> str:
    rows = []
    sub = prediction[prediction["workers"].astype(str) == "all"].copy()
    for row in sub.sort_values("guard_label").itertuples(index=False):
        rows.append(
            "        "
            + " & ".join(
                [
                    str(row.guard_label),
                    str(int(row.rounds)),
                    f"{row.tau_error_ms:.2f}",
                    f"{row.p95_tau_error_ms:.2f}",
                    f"{row.prefix_error_rows:.2f}",
                    f"{100.0 * row.false_enable_rate:.1f}\\%",
                    f"{100.0 * row.false_disable_rate:.1f}\\%",
                    f"{100.0 * row.enable_rate:.1f}\\%",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"    \centering",
            r"    \caption{Algorithm~3 prediction accuracy on paired K3s guard rounds. Tau error is absolute error between the predicted candidate first-decodable time and the paired observed candidate decode time; prefix error compares predicted and observed candidate-minus-static prefix length.}",
            r"    \label{tab:guard-prediction}",
            r"    \scriptsize",
            r"    \begin{tabular}{lrrrrrrr}",
            r"        \toprule",
            r"        Guard & Rounds & Mean $\tau$ err. & P95 $\tau$ err. & Prefix err. & False en. & False dis. & Enable \\",
            r"              &        & (ms) & (ms) & (rows) & & & \\",
            r"        \midrule",
            *rows,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )


def latex_sensitivity_table(sensitivity: pd.DataFrame) -> str:
    rows = []
    sub = sensitivity[sensitivity["guard_label"] == "Guard-D"].copy()
    order = [
        "low $\\theta_{cv}$",
        "default",
        "high $\\theta_{cv}$",
        "zero $\\theta_g$",
        "high $\\theta_g$",
        "strict $\\theta_K$",
        "loose $\\theta_K$",
        "strict $\\theta_a$",
        "loose $\\theta_a$",
    ]
    sub["order"] = sub["setting"].apply(lambda x: order.index(x) if x in order else 99)
    for row in sub.sort_values("order").itertuples(index=False):
        rows.append(
            "        "
            + " & ".join(
                [
                    str(row.setting),
                    f"{row.theta_cv:.2f}",
                    f"{row.theta_g:.2f}",
                    f"{row.theta_K:.0f}",
                    f"{row.theta_a:.0f}",
                    f"{100.0 * row.enable_rate:.1f}\\%",
                    f"{100.0 * row.false_enable_rate:.1f}\\%",
                    f"{100.0 * row.false_disable_rate:.1f}\\%",
                    f"{row.mean_barrier_gain_pct:.1f}\\%",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"    \centering",
            r"    \caption{Threshold sensitivity for Guard-D under offline replay of paired K3s rounds. Each non-default row changes one guard threshold while holding the other thresholds at the deployed values.}",
            r"    \label{tab:guard-thresholds}",
            r"    \scriptsize",
            r"    \begin{tabular}{lrrrrrrrr}",
            r"        \toprule",
            r"        Setting & $\theta_{cv}$ & $\theta_g$ & $\theta_K$ & $\theta_a$ & Enable & False en. & False dis. & Barrier gain \\",
            r"        \midrule",
            *rows,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )


def latex_resource_table(resources: pd.DataFrame) -> str:
    rows = []
    for row in resources.sort_values("policy").itertuples(index=False):
        rows.append(
            "        "
            + " & ".join(
                [
                    str(row.policy),
                    f"{row.dispatch_ms:.2f}",
                    f"{row.cancel_ms:.2f}",
                    f"{row.response_kb:.1f}",
                    f"{row.worker_errors:.2f}",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"    \centering",
            r"    \caption{K3s worker-service resource counters averaged over all worker counts and seeds.}",
            r"    \label{tab:k8s-resources}",
            r"    \scriptsize",
            r"    \begin{tabular}{lrrrr}",
            r"        \toprule",
            r"        Policy & Dispatch & Cancel & Response & Worker errors \\",
            r"               & (ms) & (ms) & (KB/round) & (/round) \\",
            r"        \midrule",
            *rows,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )


def write_threshold_sensitivity_figure(
    sensitivity: pd.DataFrame,
    out_dir: Path,
    paper_figures_dir: Path | None,
) -> None:
    if sensitivity.empty:
        return
    order = [
        "low $\\theta_{cv}$",
        "default",
        "high $\\theta_{cv}$",
        "zero $\\theta_g$",
        "high $\\theta_g$",
        "strict $\\theta_K$",
        "loose $\\theta_K$",
        "strict $\\theta_a$",
        "loose $\\theta_a$",
    ]
    sub = sensitivity[sensitivity["guard_label"] == "Guard-D"].copy()
    if sub.empty:
        return
    sub["order"] = sub["setting"].apply(lambda x: order.index(x) if x in order else 99)
    sub = sub.sort_values("order")
    labels = [
        str(label)
        .replace(" $\\theta_{cv}$", "\n$\\theta_{cv}$")
        .replace(" $\\theta_g$", "\n$\\theta_g$")
        .replace(" $\\theta_K$", "\n$\\theta_K$")
        .replace(" $\\theta_a$", "\n$\\theta_a$")
        for label in sub["setting"]
    ]
    x = np.arange(len(sub))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.bar(
        x - width / 2,
        100.0 * sub["enable_rate"],
        width=width,
        label="Enable",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.25,
    )
    ax.bar(
        x + width / 2,
        100.0 * sub["false_enable_rate"],
        width=width,
        label="False enable",
        color="#F58518",
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_ylabel("Decision rate (%)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=6.8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        sub["mean_barrier_gain_pct"],
        color="#222222",
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        label="Barrier gain",
    )
    ax2.set_ylabel("Barrier gain (%)")
    ax2.tick_params(axis="y", labelsize=7.5)
    ax2.axhline(0.0, color="#777777", linewidth=0.5, linestyle="--")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.2,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.25)

    destinations = [out_dir]
    if paper_figures_dir is not None:
        destinations.append(paper_figures_dir)
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / "guard_threshold_sensitivity.pdf", bbox_inches="tight")
        fig.savefig(destination / "guard_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    prediction: pd.DataFrame,
    sensitivity: pd.DataFrame,
    resources: pd.DataFrame,
    live_resources: pd.DataFrame,
    path: Path,
) -> None:
    parts = [
        "# Guard prediction diagnostics",
        "",
        "## Prediction accuracy",
        "",
        frame_to_markdown(prediction),
        "",
        "## Threshold sensitivity",
        "",
        frame_to_markdown(sensitivity),
        "",
        "## Resource counters",
        "",
        frame_to_markdown(resources),
        "",
        "## Live Kubernetes resource snapshots",
        "",
        (
            "No per-run `k8s_resource_counters.csv` snapshots were found."
            if live_resources.empty
            else frame_to_markdown(live_resources)
        ),
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


if __name__ == "__main__":
    main()
