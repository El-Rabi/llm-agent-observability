"""Generate content-free synthetic multi-agent runtime traces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AGENTS = ["planner", "researcher", "analyst", "writer"]
BENIGN_ACTIONS = ["handoff", "retrieve", "tool_call", "summarize", "respond"]
BENIGN_TOOLS = ["none", "vector_db", "internal_search", "calculator", "internal_api"]
LABELS = ["benign", "prompt_hijacking", "retrieval_poisoning", "tool_redirection"]


def generate_trace_data(n_sessions: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Return synthetic event rows for multi-agent sessions."""
    if n_sessions < 100:
        raise ValueError("n_sessions must be at least 100")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for index in range(n_sessions):
        session_id = f"TRACE-{index + 1:05d}"
        label = str(rng.choice(LABELS, p=[0.65, 0.12, 0.12, 0.11]))
        total_steps = int(rng.integers(9, 19))

        events = []
        for step in range(1, total_steps + 1):
            action = str(rng.choice(BENIGN_ACTIONS, p=[0.20, 0.24, 0.25, 0.18, 0.13]))
            tool = "none" if action != "tool_call" else str(rng.choice(BENIGN_TOOLS[1:]))
            events.append(
                {
                    "session_id": session_id,
                    "step": step,
                    "total_steps": total_steps,
                    "agent": str(rng.choice(AGENTS)),
                    "action": action,
                    "tool": tool,
                    "latency_ms": int(np.clip(rng.normal(420, 150), 80, 1600)),
                    "data_volume_kb": float(np.clip(rng.gamma(2.0, 16.0), 0.2, 220)),
                    "permission_level": int(rng.choice([0, 1, 2], p=[0.48, 0.45, 0.07])),
                    "external_destination": int(rng.random() < 0.025),
                    "error_flag": int(rng.random() < 0.045),
                    "label": label,
                }
            )

        early_step = max(2, int(np.ceil(total_steps * 0.28))) - 1
        later_step = max(early_step + 1, int(np.ceil(total_steps * 0.62))) - 1

        # Compromises are not always observable in metadata, and benign
        # maintenance occasionally resembles suspicious tool use. The early
        # signal is intentionally less consistent than the later signal.
        detectable_compromise = label != "benign" and rng.random() >= 0.10
        early_signal_visible = detectable_compromise and rng.random() < 0.68

        if label == "prompt_hijacking" and detectable_compromise:
            if early_signal_visible:
                events[early_step].update(
                    action="instruction_override", tool="none", permission_level=2, latency_ms=780
                )
            events[later_step].update(
                action="tool_call", tool="external_webhook", permission_level=3,
                external_destination=1, data_volume_kb=310.0
            )
        elif label == "retrieval_poisoning" and detectable_compromise:
            if early_signal_visible:
                events[early_step].update(
                    action="untrusted_retrieval", tool="external_search", external_destination=1,
                    data_volume_kb=245.0, latency_ms=980
                )
            events[later_step].update(
                action="retrieve", tool="vector_db", data_volume_kb=285.0, error_flag=1
            )
        elif label == "tool_redirection" and detectable_compromise:
            if early_signal_visible:
                events[early_step].update(
                    action="tool_call", tool="shell", permission_level=3, latency_ms=690
                )
            events[later_step].update(
                action="tool_call", tool="external_webhook", permission_level=3,
                external_destination=1, data_volume_kb=360.0
            )
        elif label == "benign" and rng.random() < 0.045:
            # Legitimate administrator automation creates hard benign cases.
            events[early_step].update(
                action="tool_call", tool="shell", permission_level=2, latency_ms=650
            )

        rows.extend(events)

    return pd.DataFrame(rows)


def save_traces(output_path: Path, n_sessions: int = 1500, seed: int = 42) -> pd.DataFrame:
    data = generate_trace_data(n_sessions=n_sessions, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data
