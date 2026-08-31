"""Aggregate content-free agent events into session-level features."""

from __future__ import annotations

import math

import pandas as pd


SUSPICIOUS_TOOLS = {"shell", "external_webhook", "external_search"}


def extract_session_features(events: pd.DataFrame, fraction: float = 1.0) -> pd.DataFrame:
    """Create one feature row per session using the requested trace fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")

    rows: list[dict[str, object]] = []
    for session_id, group in events.groupby("session_id", sort=True):
        group = group.sort_values("step")
        observed_steps = max(1, math.ceil(int(group["total_steps"].iloc[0]) * fraction))
        observed = group[group["step"] <= observed_steps]
        rows.append(
            {
                "session_id": session_id,
                "event_count": len(observed),
                "agent_count": observed["agent"].nunique(),
                "tool_call_count": int((observed["action"] == "tool_call").sum()),
                "unique_tool_count": int(observed.loc[observed["tool"] != "none", "tool"].nunique()),
                "external_call_count": int(observed["external_destination"].sum()),
                "privileged_action_count": int((observed["permission_level"] >= 2).sum()),
                "max_permission_level": int(observed["permission_level"].max()),
                "error_rate": float(observed["error_flag"].mean()),
                "mean_latency_ms": float(observed["latency_ms"].mean()),
                "max_latency_ms": int(observed["latency_ms"].max()),
                "total_data_volume_kb": float(observed["data_volume_kb"].sum()),
                "handoff_count": int((observed["action"] == "handoff").sum()),
                "retrieval_count": int(observed["action"].isin(["retrieve", "untrusted_retrieval"]).sum()),
                "suspicious_tool_count": int(observed["tool"].isin(SUSPICIOUS_TOOLS).sum()),
                "instruction_override_count": int((observed["action"] == "instruction_override").sum()),
                "untrusted_retrieval_count": int((observed["action"] == "untrusted_retrieval").sum()),
                "label": group["label"].iloc[0],
            }
        )
    return pd.DataFrame(rows)
