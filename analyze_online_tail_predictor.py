"""Online high-tail predictor diagnostic for the K3s guard logs.

The diagnostic asks whether the high-static-tail regime used in the paper's
24-worker split is visible from features available before a round starts.  It
does not drive the deployed guard.  The target label is the paired static-coded
barrier tail for the same round; the predictors use only worker-state CV,
predicted first-decode times, predicted prefix changes, and EMA counters from
previous rounds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictorSpec:
    name: str
    score_col: str
    higher_is_tail: bool = True


PREDICTORS = [
    PredictorSpec("Speed CV", "speed_cv"),
    PredictorSpec("Pred. static tau", "pred_static_tau_ms"),
    PredictorSpec("Pred. gain", "pred_gain_pct"),
    PredictorSpec("Prefix saving", "pred_prefix_saving"),
    PredictorSpec("Mismatch score", "mismatch_score"),
    PredictorSpec("Algorithm 3 score", "algorithm3_score"),
    PredictorSpec("EMA risk", "ema_risk"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-round",
        type=Path,
        default=Path("guard_prediction_diagnostics") / "guard_prediction_per_round.csv",
    )
    parser.add_argument("--out", type=Path, default=Path("tail_predictor_diagnostics"))
    parser.add_argument("--guard-label", default="Guard-D")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--tail-threshold-ms",
        type=float,
        default=50.0,
        help="Static-coded barrier threshold used as the high-tail label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rounds = pd.read_csv(args.per_round)
    table_rounds = build_round_features(
        rounds,
        guard_label=args.guard_label,
        workers=args.workers,
        tail_threshold_ms=args.tail_threshold_ms,
    )
    summary = summarize_predictors(table_rounds)
    table_rounds.to_csv(args.out / "online_tail_predictor_rounds.csv", index=False)
    summary.to_csv(args.out / "online_tail_predictor_summary.csv", index=False)
    write_latex_table(summary, args.out / "online_tail_predictor_table.tex")
    write_report(table_rounds, summary, args.out / "online_tail_predictor_report.md")
    print(summary.to_string(index=False))
    print(f"Wrote online tail predictor diagnostics to {args.out}")


def build_round_features(
    rounds: pd.DataFrame,
    *,
    guard_label: str,
    workers: int,
    tail_threshold_ms: float,
) -> pd.DataFrame:
    frame = rounds[
        (rounds["guard_label"].astype(str) == guard_label)
        & (rounds["workers"].astype(int) == workers)
    ].copy()
    if frame.empty:
        raise SystemExit(f"No {guard_label} W{workers} rounds found.")

    frame = frame.sort_values(["seed", "iteration"]).reset_index(drop=True)
    frame["pred_prefix_saving"] = -frame["pred_prefix_delta"].astype(float)
    frame["mismatch_score"] = (
        frame["speed_cv"].astype(float)
        * np.maximum(frame["pred_gain_pct"].astype(float), 0.0)
        * np.maximum(frame["pred_prefix_saving"].astype(float), 0.0)
    )
    frame = add_ema_counters(frame)
    frame["ema_risk"] = np.maximum(frame["ema_prefix_delta_before"], 0.0) + np.maximum(
        frame["ema_rows_after_decode_before"] - 1.0, 0.0
    )
    frame["algorithm3_score"] = (
        25.0 * frame["speed_cv"].astype(float)
        + frame["pred_gain_pct"].astype(float)
        + 5.0 * frame["pred_prefix_saving"].astype(float)
        - 10.0 * np.maximum(frame["ema_prefix_delta_before"], 0.0)
        - 4.0 * np.maximum(frame["ema_rows_after_decode_before"] - 1.0, 0.0)
    )
    frame["high_static_tail"] = frame["static_barrier_ms"].astype(float) >= tail_threshold_ms
    return frame


def add_ema_counters(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ema_prefix_delta_before"] = 0.0
    out["ema_rows_after_decode_before"] = 0.0
    for _, index in out.groupby("seed", sort=False).groups.items():
        ema_prefix = 0.0
        ema_after = 0.0
        for row_index in index:
            out.at[row_index, "ema_prefix_delta_before"] = ema_prefix
            out.at[row_index, "ema_rows_after_decode_before"] = ema_after
            row = out.loc[row_index]
            if str(row["observed_action"]) == "enable":
                ema_prefix = 0.70 * ema_prefix + 0.30 * float(row["realized_prefix_delta"])
                ema_after = 0.70 * ema_after + 0.30 * float(row["candidate_rows_after_decode"])
            else:
                ema_prefix *= 0.80
                ema_after *= 0.80
    return out


def summarize_predictors(rounds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in PREDICTORS:
        scores = rounds[spec.score_col].astype(float).to_numpy()
        if not spec.higher_is_tail:
            scores = -scores
        labels = rounds["high_static_tail"].astype(bool).to_numpy()
        predictions = leave_one_seed_out_predictions(rounds, scores, labels)
        tp = int(np.sum(predictions & labels))
        fp = int(np.sum(predictions & ~labels))
        tn = int(np.sum(~predictions & ~labels))
        fn = int(np.sum(~predictions & labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "predictor": spec.name,
                "rounds": int(labels.size),
                "positives": int(labels.sum()),
                "auc": auc(scores, labels),
                "precision": precision,
                "recall": recall,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "true_negatives": tn,
            }
        )
    return pd.DataFrame.from_records(rows)


def leave_one_seed_out_predictions(
    rounds: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros(labels.size, dtype=bool)
    seeds = rounds["seed"].astype(int).to_numpy()
    for seed in sorted(set(seeds)):
        train = seeds != seed
        test = seeds == seed
        threshold = calibrate_threshold(scores[train], labels[train])
        predictions[test] = scores[test] >= threshold
    return predictions


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0:
        return float("inf")
    candidates = np.unique(scores)
    best_threshold = float(candidates[-1])
    best_score = (-1.0, -1.0)
    for threshold in candidates:
        pred = scores >= threshold
        tp = float(np.sum(pred & labels))
        fp = float(np.sum(pred & ~labels))
        fn = float(np.sum(~pred & labels))
        tn = float(np.sum(~pred & ~labels))
        tpr = tp / max(tp + fn, 1.0)
        fpr = fp / max(fp + tn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tpr
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        youden = tpr - fpr
        score = (f1, youden)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = 0.0
    total = float(pos.size * neg.size)
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return wins / total


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Online high-static-tail prediction on W24 K3s Guard-D rounds. The label is static-coded barrier latency above 50 ms. Scores use only pre-iteration features and EMA counters from previous rounds; precision/recall use leave-one-seed-out threshold calibration.}",
        r"  \label{tab:online-tail-predictor}",
        r"  \begin{tabular}{@{}lrrrrr@{}}",
        r"    \toprule",
        r"    Predictor & AUC & Precision & Recall & FP & FN \\",
        r"    \midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "    "
            + " & ".join(
                [
                    str(row.predictor),
                    f"{row.auc:.2f}",
                    f"{100.0 * row.precision:.1f}\\%",
                    f"{100.0 * row.recall:.1f}\\%",
                    str(int(row.false_positives)),
                    str(int(row.false_negatives)),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(rounds: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    positives = int(rounds["high_static_tail"].sum())
    seeds = ",".join(str(int(seed)) for seed in sorted(rounds["seed"].unique()))
    lines = [
        "# Online High-Tail Predictor Diagnostic",
        "",
        f"Rounds: {len(rounds)}; positives: {positives}; seeds: {seeds}.",
        "",
        "The label is diagnostic: paired static-coded barrier latency above 50 ms.",
        "The features are available before scheduling the current guard round, except that EMA counters use only previous rounds.",
        "",
        frame_to_markdown(summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def frame_to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


if __name__ == "__main__":
    main()
