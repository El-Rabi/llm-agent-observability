"""Generate traces, engineer features, and compare detection timing."""

from pathlib import Path

from features import extract_session_features
from generate_traces import save_traces
from train import run_comparison


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "agent_events.csv"
    results_dir = project_root / "results"

    events = save_traces(data_path, n_sessions=1500, seed=42)
    full_features = extract_session_features(events, fraction=1.0)
    early_features = extract_session_features(events, fraction=0.40)
    results_dir.mkdir(parents=True, exist_ok=True)
    full_features.to_csv(results_dir / "full_trace_features.csv", index=False)
    early_features.to_csv(results_dir / "early_trace_features.csv", index=False)
    metrics = run_comparison(full_features, early_features, results_dir, seed=42)

    print(f"Generated {full_features['session_id'].nunique():,} synthetic agent sessions")
    print(f"Early-trace ROC-AUC: {metrics['early']['roc_auc']:.3f}")
    print(f"Full-trace ROC-AUC: {metrics['full']['roc_auc']:.3f}")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
