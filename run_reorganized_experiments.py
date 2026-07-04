from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.coded_learning_exp.experiment import ExperimentConfig, run_experiment


ASSIGNMENT_STRATEGIES = (
    "uncoded_sync",
    "replication",
    "static_sparse_code",
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
    "balanced_sparse_flexible",
    "balanced_rank_aware_sparse_flexible",
    "balanced_deadline_aware_sparse_flexible",
    "flexible_robust_static",
)

SPARSITY_STRATEGIES = (
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
    "balanced_rank_aware_sparse_flexible",
    "balanced_deadline_aware_sparse_flexible",
    "flexible_thin_static",
    "flexible_robust_static",
    "flexible_dense_static",
)

ADAPTATION_STRATEGIES = (
    "flexible_thin_static",
    "sparse_flexible_static",
    "flexible_robust_static",
    "flexible_dense_static",
    "adaptive_sparse_flexible",
    "ucb_sparse_flexible",
    "window_sparse_flexible",
    "worker_aware_ucb_sparse_flexible",
    "rank_aware_ucb_sparse_flexible",
    "balanced_rank_aware_ucb_sparse_flexible",
    "contextual_ucb_sparse_flexible",
)

ABLATION_STRATEGIES = (
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "adaptive_sparse_flexible",
    "adaptive_latency_only",
    "window_sparse_flexible",
    "worker_aware_adaptive_sparse_flexible",
    "worker_aware_ucb_sparse_flexible",
    "rank_aware_adaptive_sparse_flexible",
    "rank_aware_ucb_sparse_flexible",
    "balanced_rank_aware_ucb_sparse_flexible",
    "contextual_ucb_sparse_flexible",
)

STATIC_FLEXIBLE = {
    "flexible_thin_static",
    "sparse_flexible_static",
    "flexible_robust_static",
    "flexible_dense_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
    "balanced_sparse_flexible",
    "balanced_rank_aware_sparse_flexible",
    "balanced_deadline_aware_sparse_flexible",
}

ADAPTIVE = {
    "adaptive_sparse_flexible",
    "ucb_sparse_flexible",
    "window_sparse_flexible",
    "worker_aware_ucb_sparse_flexible",
    "contextual_ucb_sparse_flexible",
    "worker_aware_adaptive_sparse_flexible",
    "rank_aware_adaptive_sparse_flexible",
    "rank_aware_ucb_sparse_flexible",
    "balanced_rank_aware_ucb_sparse_flexible",
}


@dataclass(frozen=True)
class ResearchCase:
    suite: str
    label: str
    scenario: str
    density: float
    straggler_fraction: float
    straggler_slowdown: float
    drift_period: int
    seed: int
    strategies: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hypothesis-organized coded learning experiments."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Smoke is fast; full uses more seeds and parameter points.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=("assignment", "sparsity", "adaptation", "ablation"),
        default=["assignment", "sparsity", "adaptation", "ablation"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--samples", type=int, default=2600)
    parser.add_argument("--features", type=int, default=420)
    parser.add_argument("--shards", type=int, default=14)
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("organized_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds or ((5,) if args.mode == "smoke" else (3, 5, 11)))
    rounds = args.rounds or (50 if args.mode == "smoke" else 120)
    cases = build_cases(args.mode, args.suites, seeds)

    summaries: list[pd.DataFrame] = []
    for case in cases:
        run_dir = args.out / case.suite / case.label / f"seed{case.seed}"
        config = ExperimentConfig(
            n_samples=args.samples,
            n_features=args.features,
            density=case.density,
            n_shards=args.shards,
            n_workers=args.workers,
            rounds=rounds,
            scenario=case.scenario,
            drift_period=case.drift_period,
            straggler_fraction=case.straggler_fraction,
            straggler_slowdown=case.straggler_slowdown,
            seed=case.seed,
            output_dir=run_dir,
            strategy_names=case.strategies,
        )
        _, summary = run_experiment(config)
        summary.insert(0, "suite", case.suite)
        summary.insert(1, "case", case.label)
        summary.insert(2, "scenario", case.scenario)
        summary.insert(3, "density", case.density)
        summary.insert(4, "straggler_fraction", case.straggler_fraction)
        summary.insert(5, "straggler_slowdown", case.straggler_slowdown)
        summary.insert(6, "drift_period", case.drift_period)
        summary.insert(7, "seed", case.seed)
        summaries.append(summary)
        best = summary.loc[summary["mean_latency"].idxmin()]
        print(f"{case.suite}/{case.label}/seed{case.seed}: best={best['strategy']}")

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.out / "organized_summary.csv", index=False)

    aggregate = aggregate_results(combined)
    aggregate.to_csv(args.out / "organized_aggregate.csv", index=False)

    report = build_hypothesis_report(aggregate)
    report.to_csv(args.out / "hypothesis_report.csv", index=False)

    write_markdown_report(report, aggregate, args.out / "research_report.md")
    write_plots(aggregate, report, args.out)

    print(f"Wrote organized summary to {args.out / 'organized_summary.csv'}")
    print(f"Wrote hypothesis report to {args.out / 'hypothesis_report.csv'}")
    print(report.to_string(index=False))


def build_cases(
    mode: str, requested_suites: list[str], seeds: tuple[int, ...]
) -> list[ResearchCase]:
    cases: list[ResearchCase] = []
    suite_set = set(requested_suites)
    densities = (0.005, 0.02) if mode == "smoke" else (0.003, 0.01, 0.03, 0.06)

    if "assignment" in suite_set:
        fractions = (0.15, 0.35) if mode == "smoke" else (0.10, 0.20, 0.35, 0.45)
        slowdowns = (0.22, 0.12) if mode == "smoke" else (0.30, 0.20, 0.12, 0.08)
        for fraction in fractions:
            for slowdown in slowdowns:
                for seed in seeds:
                    label = f"phase_f{fraction:g}_s{slowdown:g}"
                    cases.append(
                        ResearchCase(
                            suite="assignment",
                            label=label,
                            scenario="phase",
                            density=0.01,
                            straggler_fraction=fraction,
                            straggler_slowdown=slowdown,
                            drift_period=20,
                            seed=seed,
                            strategies=ASSIGNMENT_STRATEGIES,
                        )
                    )

    if "sparsity" in suite_set:
        for density in densities:
            for seed in seeds:
                cases.append(
                    ResearchCase(
                        suite="sparsity",
                        label=f"density{density:g}",
                        scenario="burst",
                        density=density,
                        straggler_fraction=0.35,
                        straggler_slowdown=0.12,
                        drift_period=20,
                        seed=seed,
                        strategies=SPARSITY_STRATEGIES,
                    )
                )

    if "adaptation" in suite_set:
        scenarios = ("burst", "phase")
        for scenario in scenarios:
            for seed in seeds:
                cases.append(
                    ResearchCase(
                        suite="adaptation",
                        label=f"{scenario}_dynamic",
                        scenario=scenario,
                        density=0.01,
                        straggler_fraction=0.35,
                        straggler_slowdown=0.12,
                        drift_period=20,
                        seed=seed,
                        strategies=ADAPTATION_STRATEGIES,
                    )
                )

    if "ablation" in suite_set:
        scenarios = ("burst",) if mode == "smoke" else ("burst", "phase")
        for scenario in scenarios:
            for seed in seeds:
                cases.append(
                    ResearchCase(
                        suite="ablation",
                        label=f"{scenario}_high_hetero",
                        scenario=scenario,
                        density=0.01,
                        straggler_fraction=0.35,
                        straggler_slowdown=0.12,
                        drift_period=20,
                        seed=seed,
                        strategies=ABLATION_STRATEGIES,
                    )
                )
    return cases


def aggregate_results(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby(
            [
                "suite",
                "case",
                "scenario",
                "density",
                "straggler_fraction",
                "straggler_slowdown",
                "strategy",
            ],
            sort=False,
        )
        .agg(
            mean_latency=("mean_latency", "mean"),
            p95_latency=("p95_latency", "mean"),
            steady_mean_latency=("steady_mean_latency", "mean"),
            steady_p95_latency=("steady_p95_latency", "mean"),
            total_wall_clock=("total_wall_clock", "mean"),
            final_loss=("final_loss", "mean"),
            decode_success_rate=("decode_success_rate", "mean"),
            mean_extra_compute=("mean_extra_compute", "mean"),
            mean_selected_rows=("mean_selected_rows", "mean"),
            second_layer_rate=("second_layer_rate", "mean"),
            latency_gap_to_oracle_static_flexible=(
                "latency_gap_to_oracle_static_flexible",
                "mean",
            ),
        )
        .reset_index()
    )


def build_hypothesis_report(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    rows.extend(
        compare_by_case(
            aggregate,
            "assignment",
            "H1a cost-aware assignment",
            "sparse_flexible_static",
            "worker_aware_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "assignment",
            "H1b decode-aware assignment",
            "sparse_flexible_static",
            "rank_aware_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "assignment",
            "H1c deadline-aware assignment",
            "sparse_flexible_static",
            "deadline_aware_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "assignment",
            "H1d decode-balanced code",
            "sparse_flexible_static",
            "balanced_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "assignment",
            "H1e balanced decode-aware assignment",
            "rank_aware_sparse_flexible",
            "balanced_rank_aware_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "sparsity",
            "H2a cost-aware under sparse inputs",
            "sparse_flexible_static",
            "worker_aware_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "sparsity",
            "H2b decode-aware under sparse inputs",
            "sparse_flexible_static",
            "rank_aware_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "sparsity",
            "H2c balanced decode-aware under sparse inputs",
            "rank_aware_sparse_flexible",
            "balanced_rank_aware_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "adaptation",
            "H3 window adaptation",
            "adaptive_sparse_flexible",
            "window_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "adaptation",
            "H4 UCB exploration",
            "adaptive_sparse_flexible",
            "ucb_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "adaptation",
            "H5 contextual UCB",
            "window_sparse_flexible",
            "contextual_ucb_sparse_flexible",
        )
    )
    rows.extend(
        compare_by_case(
            aggregate,
            "ablation",
            "H6 compute-aware reward",
            "adaptive_latency_only",
            "adaptive_sparse_flexible",
        )
    )
    rows.extend(best_adaptive_vs_best_fixed(aggregate))
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return (
        report.sort_values(["hypothesis", "scope"])
        .reset_index(drop=True)
    )


def compare_by_case(
    aggregate: pd.DataFrame,
    suite: str,
    hypothesis: str,
    reference: str,
    candidate: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    suite_frame = aggregate[aggregate["suite"] == suite]
    for case, frame in suite_frame.groupby("case", sort=False):
        if reference not in set(frame["strategy"]) or candidate not in set(frame["strategy"]):
            continue
        ref = frame[frame["strategy"] == reference].iloc[0]
        cand = frame[frame["strategy"] == candidate].iloc[0]
        rows.append(make_report_row(hypothesis, case, reference, candidate, ref, cand))
    if rows:
        rows.append(aggregate_report_rows(hypothesis, rows))
    return rows


def best_adaptive_vs_best_fixed(aggregate: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    frame = aggregate[aggregate["suite"] == "adaptation"]
    for case, case_frame in frame.groupby("case", sort=False):
        fixed = case_frame[case_frame["strategy"].isin(STATIC_FLEXIBLE)]
        adaptive = case_frame[case_frame["strategy"].isin(ADAPTIVE)]
        if fixed.empty or adaptive.empty:
            continue
        ref = fixed.loc[fixed["mean_latency"].idxmin()]
        cand = adaptive.loc[adaptive["mean_latency"].idxmin()]
        rows.append(
            make_report_row(
                "H7 best online policy vs best fixed",
                case,
                str(ref["strategy"]),
                str(cand["strategy"]),
                ref,
                cand,
            )
        )
    if rows:
        rows.append(aggregate_report_rows("H7 best online policy vs best fixed", rows))
    return rows


def make_report_row(
    hypothesis: str,
    scope: str,
    reference: str,
    candidate: str,
    ref: pd.Series,
    cand: pd.Series,
) -> dict[str, float | str]:
    mean_improvement = improvement(float(ref["mean_latency"]), float(cand["mean_latency"]))
    p95_improvement = improvement(float(ref["p95_latency"]), float(cand["p95_latency"]))
    steady_improvement = improvement(
        float(ref["steady_mean_latency"]), float(cand["steady_mean_latency"])
    )
    extra_compute_delta = float(cand["mean_extra_compute"] - ref["mean_extra_compute"])
    return {
        "hypothesis": hypothesis,
        "scope": scope,
        "reference": reference,
        "candidate": candidate,
        "mean_latency_improvement": mean_improvement,
        "p95_latency_improvement": p95_improvement,
        "steady_latency_improvement": steady_improvement,
        "extra_compute_delta": extra_compute_delta,
        "candidate_decode_success": float(cand["decode_success_rate"]),
        "verdict": verdict(mean_improvement, p95_improvement, steady_improvement),
    }


def aggregate_report_rows(
    hypothesis: str, rows: list[dict[str, float | str]]
) -> dict[str, float | str]:
    numeric = [
        "mean_latency_improvement",
        "p95_latency_improvement",
        "steady_latency_improvement",
        "extra_compute_delta",
        "candidate_decode_success",
    ]
    aggregate_row: dict[str, float | str] = {
        "hypothesis": hypothesis,
        "scope": "ALL",
        "reference": "case average",
        "candidate": "case average",
    }
    for column in numeric:
        aggregate_row[column] = float(pd.Series([row[column] for row in rows]).mean())
    aggregate_row["verdict"] = verdict(
        float(aggregate_row["mean_latency_improvement"]),
        float(aggregate_row["p95_latency_improvement"]),
        float(aggregate_row["steady_latency_improvement"]),
    )
    return aggregate_row


def verdict(mean_improvement: float, p95_improvement: float, steady_improvement: float) -> str:
    if mean_improvement >= 0.05 and steady_improvement >= 0.03 and p95_improvement > -0.05:
        return "promising"
    if mean_improvement >= 0.01 or steady_improvement >= 0.03:
        return "mixed"
    return "weak_or_negative"


def improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def write_markdown_report(report: pd.DataFrame, aggregate: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Reorganized Experiment Report",
        "",
        "The experiments are organized by hypothesis instead of by strategy list.",
        "",
        "## Hypothesis Summary",
        "",
    ]
    all_rows = report[report["scope"] == "ALL"]
    for _, row in all_rows.iterrows():
        lines.append(
            "- "
            + str(row["hypothesis"])
            + ": "
            + f"mean {100 * row['mean_latency_improvement']:.1f}%, "
            + f"P95 {100 * row['p95_latency_improvement']:.1f}%, "
            + f"steady {100 * row['steady_latency_improvement']:.1f}%, "
            + f"verdict `{row['verdict']}`."
        )
    lines.extend(
        [
            "",
            "## Paper-Facing Interpretation",
            "",
            "- Treat decode-aware worker assignment as the main positive result; cost-only assignment is a diagnostic baseline and can be harmful.",
            "- Keep online code-density adaptation as an ablation: useful in bursty cases, but not yet robust enough to be the paper's central claim.",
            "- Use contextual policies as a negative/diagnostic result unless richer features or warm-start training are added.",
            "- Report best fixed flexible and per-round oracle gaps so reviewers can see whether online adaptation has real headroom.",
            "",
            "## Files",
            "",
            "- `organized_summary.csv`: per-run summaries.",
            "- `organized_aggregate.csv`: seed-averaged metrics by case and strategy.",
            "- `hypothesis_report.csv`: direct hypothesis comparisons.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(aggregate: pd.DataFrame, report: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    all_rows = report[report["scope"] == "ALL"].copy()
    if not all_rows.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        labels = all_rows["hypothesis"].str.replace(r"^H\d+\s+", "", regex=True)
        ax.bar(labels, 100.0 * all_rows["mean_latency_improvement"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Mean latency improvement (%)")
        ax.set_title("Hypothesis-level results")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / "hypothesis_improvements.png", dpi=180)
        plt.close(fig)

    assignment = aggregate[
        (aggregate["suite"] == "assignment")
        & aggregate["strategy"].isin(
            ["sparse_flexible_static", "worker_aware_sparse_flexible"]
            + [
                "rank_aware_sparse_flexible",
                "deadline_aware_sparse_flexible",
                "balanced_rank_aware_sparse_flexible",
            ]
        )
    ]
    if not assignment.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for strategy, frame in assignment.groupby("strategy", sort=False):
            ax.plot(frame["case"], frame["mean_latency"], marker="o", label=strategy)
        ax.set_ylabel("Mean iteration latency")
        ax.set_title("Worker-aware assignment under heterogeneity")
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "assignment_latency.png", dpi=180)
        plt.close(fig)

    sparsity = report[
        (report["hypothesis"] == "H2b decode-aware under sparse inputs")
        & (report["scope"] != "ALL")
    ].copy()
    if not sparsity.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(
            sparsity["scope"],
            100.0 * sparsity["mean_latency_improvement"],
            marker="o",
            label="latency improvement",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Improvement (%)")
        ax.set_title("Worker-aware benefit across input sparsity")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / "sparsity_worker_aware.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
