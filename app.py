"""ThreatLens: network anomaly triage dashboard (Streamlit).

Run with:  streamlit run app.py

Flow: load data -> validate -> score with the saved Isolation Forest -> rank alerts ->
explain the selected alert -> analyst labels it -> label is saved in SQLite -> metrics update.
Streamlit re-runs this whole script on every interaction, so anything slow is cached
and every label is written to SQLite immediately.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import db
from src.data_loader import LABEL_COLUMN, ValidationError, is_attack, load_connections
from src.metrics import (CONFIRMED, FALSE_POSITIVE, benchmark_metrics, labeled_precision,
                         suggest_threshold, threshold_sweep)
from src.model import DEFAULT_MODEL_PATH, explain_records, load_bundle, score_records, summarize_reasons

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_nslkdd.csv"
MAX_TABLE_ROWS = 500
DEFAULT_THRESHOLD = 95.0
LABEL_NAMES = {CONFIRMED: "confirmed threat", FALSE_POSITIVE: "false positive"}

st.set_page_config(page_title="ThreatLens", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- cached work
@st.cache_resource(show_spinner="Loading model...")
def get_bundle():
    return load_bundle(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner="Validating and scoring...")
def load_and_score(raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse + validate + score an uploaded file. Cached on the file's bytes."""
    df = load_connections(raw)
    return df, score_records(get_bundle(), df)


# --------------------------------------------------------------------------- DB callbacks
def _label_callback(run_id: int, record_index: int, score: float, flagged: bool,
                    top_features: str, record: dict, label: str | None) -> None:
    """Runs *before* the next rerun, so the table and metrics already reflect the change."""
    conn = db.connect()
    try:
        if label is None:
            db.clear_label(conn, run_id, record_index)
        else:
            db.save_label(conn, run_id, record_index, score, flagged, top_features, record, label)
    finally:
        conn.close()


def _apply_threshold(value: float) -> None:
    st.session_state["threshold"] = float(round(value, 1))


def _demo_review(run_id: int, df: pd.DataFrame, scores: pd.DataFrame, threshold: float, n: int) -> None:
    """Demo shortcut: label a random sample of flagged alerts using the file's ground truth."""
    bundle = get_bundle()
    flagged_idx = scores.index[scores["score"] >= threshold].to_numpy()
    pick = np.random.RandomState(0).permutation(flagged_idx)[:n]
    truth = is_attack(df.loc[pick, LABEL_COLUMN])
    explanations = explain_records(bundle, df.loc[pick])
    conn = db.connect()
    try:
        for idx, threat, reasons in zip(pick, truth, explanations):
            db.save_label(conn, run_id, int(idx), float(scores.at[idx, "score"]), True,
                          summarize_reasons(reasons), df.loc[idx].to_dict(),
                          CONFIRMED if threat else FALSE_POSITIVE)
    finally:
        conn.close()


# --------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[bytes | None, str]:
    st.sidebar.title("🔎 ThreatLens")
    st.sidebar.caption("Network anomaly triage")
    source = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True)
    raw, name = None, ""
    if source == "Sample data":
        raw, name = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
        st.sidebar.caption("3,000 unseen connections from NSL-KDD KDDTest+, with ground-truth labels.")
    else:
        uploaded = st.sidebar.file_uploader("Connection records (CSV)", type=["csv", "txt"])
        if uploaded is not None:
            raw, name = uploaded.getvalue(), uploaded.name

    st.sidebar.divider()
    st.sidebar.slider(
        "Sensitivity threshold", 0.0, 100.0, step=0.5, key="threshold",
        help="A connection is flagged when its score is at or above this value. The score is the "
             "percentage of normal training traffic that looks LESS unusual than this connection.")
    t = st.session_state["threshold"]
    st.sidebar.caption(f"Flag connections more unusual than **{t:.1f}%** of normal traffic "
                       f"(about {100 - t:.1f}% of normal traffic would be flagged).")
    return raw, name


# --------------------------------------------------------------------------- main sections
def metric_row(df, scores, threshold, labels, has_truth) -> None:
    flagged = int((scores["score"] >= threshold).sum())
    rev = labeled_precision(pd.Series(list(labels.values()), dtype=object))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records scored", f"{len(df):,}")
    c2.metric("Flagged", f"{flagged:,}", f"{flagged / len(df):.1%} of records", delta_color="off")
    c3.metric("Alerts reviewed", f"{rev['reviewed']:,}",
              f"{rev['confirmed']} confirmed / {rev['false_positive']} false positive", delta_color="off")
    c4.metric("Precision (my labels)", "n/a" if rev["precision"] is None else f"{rev['precision']:.2f}",
              help="Share of reviewed alerts that you confirmed as real threats.")
    if has_truth:
        m = benchmark_metrics(scores["score"].to_numpy(), is_attack(df[LABEL_COLUMN]), threshold)
        st.caption("**Benchmark** (uses this file's ground-truth `label` column, which real traffic would not have)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Precision", f"{m['precision']:.3f}")
        b2.metric("Recall", f"{m['recall']:.3f}")
        b3.metric("F1", f"{m['f1']:.3f}")
        b4.metric("False-alarm rate", f"{m['false_alarm_rate']:.3f}")


def build_alert_table(df, scores, threshold, labels, has_truth) -> pd.DataFrame:
    flagged = scores[scores["score"] >= threshold].sort_values("raw_score", ascending=False)
    view = pd.DataFrame({
        "rank": np.arange(1, len(flagged) + 1),
        "record_index": flagged.index,
        "score": flagged["score"].round(1),
        "protocol": df.loc[flagged.index, "protocol_type"],
        "service": df.loc[flagged.index, "service"],
        "flag": df.loc[flagged.index, "flag"],
        "src_bytes": df.loc[flagged.index, "src_bytes"],
        "dst_bytes": df.loc[flagged.index, "dst_bytes"],
        "count": df.loc[flagged.index, "count"],
        "my label": [LABEL_NAMES.get(labels.get(int(i)), "") for i in flagged.index],
    })
    if has_truth:
        view["ground truth"] = df.loc[flagged.index, LABEL_COLUMN].to_numpy()
    return view.reset_index(drop=True)


def alerts_tab(conn, run_id, df, scores, threshold, labels, has_truth) -> None:
    view = build_alert_table(df, scores, threshold, labels, has_truth)
    if view.empty:
        st.info("No connections reach this threshold. Lower the sensitivity threshold in the sidebar.")
        return
    shown = view.head(MAX_TABLE_ROWS)
    st.subheader(f"Ranked alerts ({len(view):,})")
    st.caption("Ranked by anomaly score, most unusual first. Select a row to see why it was flagged."
               + (f" Showing the top {MAX_TABLE_ROWS} of {len(view):,}." if len(view) > MAX_TABLE_ROWS else ""))
    event = st.dataframe(shown, hide_index=True, width="stretch", height=300,
                         on_select="rerun", selection_mode="single-row", key=f"alerts_{run_id}")

    # Which alert is selected? Prefer the click; otherwise keep the previous one; otherwise the top alert.
    chosen = shown["record_index"].iloc[0]
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        chosen = int(shown["record_index"].iloc[selected_rows[0]])
        st.session_state["selected_record"] = chosen
    elif st.session_state.get("selected_record") in set(shown["record_index"]):
        chosen = st.session_state["selected_record"]
    chosen = int(chosen)

    row = view[view["record_index"] == chosen].iloc[0]
    record = df.loc[chosen]
    reasons = explain_records(get_bundle(), df.loc[[chosen]])[0]
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Why was #{int(row['rank'])} flagged?")
        st.caption(f"Record {chosen} · {record['protocol_type']}/{record['service']} · "
                   f"score {row['score']:.1f}/100")
        numeric = [r for r in reasons if r["kind"] == "numeric"]
        if numeric:
            numeric = numeric[::-1]  # plotly draws the first bar at the bottom
            fig = go.Figure(go.Bar(
                x=[r["z"] for r in numeric], y=[r["feature"] for r in numeric], orientation="h",
                marker_color=["#c0504d" if r["z"] > 0 else "#4a7ebb" for r in numeric],
                text=[f"{r['z']:+.1f}σ  (value {r['value']:g}, typical {r['typical']:g})" for r in numeric],
                textposition="auto"))
            fig.update_layout(height=40 * len(numeric) + 80, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="standard deviations from normal traffic (log scale)")
            st.plotly_chart(fig, width="stretch")
        for r in reasons:
            if r["kind"] == "category":
                st.warning(f"**{r['feature']} = `{r['value']}`** appears in only {r['typical']:.2%} of normal traffic.")
        if not reasons:
            st.info("No single feature stands out. The connection is unusual because of a combination of features.")
        st.caption("Explanations are an approximation: each feature is compared with normal traffic "
                   "on its own. The forest itself scores combinations of features.")
        with st.expander("Full record"):
            # str: one column mixes text (protocol) and numbers, which the table renderer rejects
            st.dataframe(record.astype(str).to_frame("value"), width="stretch")
    with right:
        st.subheader("Analyst feedback")
        current = labels.get(chosen)
        st.markdown(f"Current label: **{LABEL_NAMES.get(current, 'not reviewed')}**")
        args = (run_id, chosen, float(row["score"]), True, summarize_reasons(reasons), record.to_dict())
        st.button("✅ Confirmed threat", type="primary", width="stretch", key=f"c{chosen}",
                  on_click=_label_callback, args=(*args, CONFIRMED))
        st.button("❌ False positive", width="stretch", key=f"f{chosen}",
                  on_click=_label_callback, args=(*args, FALSE_POSITIVE))
        st.button("Clear label", width="stretch", key=f"x{chosen}",
                  on_click=_label_callback, args=(*args, None), disabled=current is None)
        st.caption("Saved to SQLite immediately and restored the next time you open this file.")


def distribution_tab(df, scores, threshold, has_truth) -> None:
    st.subheader("Score distribution")
    fig = go.Figure()
    if has_truth:
        attack = is_attack(df[LABEL_COLUMN])
        fig.add_histogram(x=scores["score"][~attack], name="normal", xbins=dict(start=0, end=100, size=2),
                          marker_color="#4a7ebb", opacity=0.75)
        fig.add_histogram(x=scores["score"][attack], name="attack", xbins=dict(start=0, end=100, size=2),
                          marker_color="#c0504d", opacity=0.75)
        fig.update_layout(barmode="overlay")
    else:
        fig.add_histogram(x=scores["score"], name="records", xbins=dict(start=0, end=100, size=2),
                          marker_color="#7f7f7f")
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"threshold {threshold:.1f}")
    fig.update_layout(xaxis_title="anomaly score (0-100)", yaxis_title="connections",
                      margin=dict(l=0, r=0, t=10, b=0), height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption("The tall bar at 100 is every connection more unusual than ALL held-out normal traffic. "
               "Within that bar, alerts are still ranked by the forest's underlying raw score.")
    if has_truth:
        sweep = threshold_sweep(scores["score"].to_numpy(), is_attack(df[LABEL_COLUMN]))
        fig2 = go.Figure()
        for col, colour in [("precision", "#4a7ebb"), ("recall", "#c0504d"), ("f1", "#2e8b57")]:
            fig2.add_scatter(x=sweep["threshold"], y=sweep[col], name=col, line_color=colour)
        fig2.add_vline(x=threshold, line_dash="dash")
        fig2.update_layout(xaxis_title="threshold", yaxis_title="metric", height=320,
                           margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[0, 1.02])
        st.subheader("Precision / recall trade-off")
        st.plotly_chart(fig2, width="stretch")


def feedback_tab(conn, run_id, df, scores, threshold, labels, has_truth) -> None:
    st.subheader("Feedback loop")
    rev = labeled_precision(pd.Series(list(labels.values()), dtype=object))
    reviewed_idx = np.array(list(labels.keys()), dtype=int)
    suggestion = None
    if len(reviewed_idx):
        suggestion = suggest_threshold(scores["score"].to_numpy()[reviewed_idx],
                                       np.array([labels[i] for i in reviewed_idx]))
    if suggestion:
        st.success(f"Suggested threshold: **{suggestion['threshold']:.1f}**. On your {rev['reviewed']} reviewed alerts it "
                   f"gives precision {suggestion['precision']:.2f} and recall {suggestion['recall']:.2f} "
                   f"(recall is relative to the threats you confirmed).")
        st.button("Apply suggested threshold", on_click=_apply_threshold, args=(suggestion["threshold"],))
    else:
        st.info(f"Review at least 5 alerts, including both confirmed threats and false positives, "
                f"to get a threshold suggestion. Reviewed so far: {rev['reviewed']}.")

    if has_truth:
        with st.expander("Demo shortcut: simulate an analyst"):
            st.caption("Labels a random sample of currently flagged alerts using this file's ground-truth column, "
                       "so you can see the feedback loop without clicking through dozens of alerts. "
                       "A real analyst would not have ground truth.")
            n = st.number_input("Number of alerts to label", 5, 200, 40, step=5)
            st.button("Simulate review", on_click=_demo_review, args=(run_id, df, scores, threshold, int(n)))

    st.markdown("#### Alert history")
    all_runs = st.checkbox("Include all uploaded files", value=False)
    history = db.get_labeled_alerts(conn, None if all_runs else run_id)
    if history.empty:
        st.caption("No labeled alerts yet.")
    else:
        st.dataframe(history.drop(columns=["run_id"]), hide_index=True, width="stretch", height=260)
        st.download_button("Export alert history (CSV)", history.to_csv(index=False).encode("utf-8-sig"),
                           file_name="threatlens_alert_history.csv", mime="text/csv")


def about_tab() -> None:
    st.markdown(ABOUT_TEXT)


ABOUT_TEXT = """
### How ThreatLens works

**1. The model: Isolation Forest.** It is *unsupervised*: it learns what normal traffic looks like and
never sees attack labels. It builds many random decision trees. Each tree repeatedly picks a random feature
and a random split value. An unusual connection sits alone in a sparse region, so it gets separated from the
rest after only a few splits; a typical connection is surrounded by similar ones and takes many splits. The
**average number of splits needed to isolate a connection** is its anomaly score (fewer splits = more unusual).

**2. The 0-100 score.** The raw forest score has no natural unit, so ThreatLens converts it to a percentile.
A score of 97 means the connection looks more unusual than 97% of normal traffic the forest never saw during
training. That is why the slider doubles as a false-alarm control: threshold 95 flags about 5% of normal traffic.

**3. The explanations.** Isolation Forest cannot explain a single prediction, so ThreatLens adds a simple
approximation: for each feature it measures how many standard deviations the value is from the normal-traffic
average (on a log scale, because byte counts are extremely skewed), and it flags category values (protocol,
service, flag) that are rare in normal traffic. It is a guide for the analyst, not a statement of what the forest used.

**4. The feedback loop.** Each label is stored in SQLite. From your labeled alerts ThreatLens computes the
precision of your reviewed alerts and suggests a threshold that balances precision and recall for *your* data.

**Limits.** The model was trained on 1990s-era NSL-KDD traffic; treat results as a demonstration of the
workflow, not as production protection. In real networks attacks are rare, so precision will be lower than on this
benchmark, where more than half of the test records are attacks.
"""


# --------------------------------------------------------------------------- page
def main() -> None:
    st.session_state.setdefault("threshold", DEFAULT_THRESHOLD)
    raw, name = sidebar()
    st.title("ThreatLens: Network Anomaly Triage")

    if raw is None:
        st.info("Upload a CSV of connection records in the sidebar, or switch to **Sample data**.")
        about_tab()
        return
    try:
        df, scores = load_and_score(raw)
    except ValidationError as exc:
        st.error(f"**This file cannot be scored.** {exc}")
        st.caption("Tip: the sample file `data/sample_nslkdd.csv` shows the expected columns.")
        return
    except RuntimeError as exc:           # e.g. scikit-learn version mismatch
        st.error(str(exc))
        return

    threshold = float(st.session_state["threshold"])
    has_truth = LABEL_COLUMN in df.columns
    conn = db.connect()
    try:
        run_id = db.get_or_create_run(conn, name, hashlib.sha256(raw).hexdigest(), len(df), threshold)
        db.update_threshold(conn, run_id, threshold)
        labels = db.get_labels(conn, run_id)

        metric_row(df, scores, threshold, labels, has_truth)
        tab_alerts, tab_dist, tab_feedback, tab_about = st.tabs(
            ["Alerts", "Score distribution", "Metrics & feedback", "How it works"])
        with tab_alerts:
            alerts_tab(conn, run_id, df, scores, threshold, labels, has_truth)
        with tab_dist:
            distribution_tab(df, scores, threshold, has_truth)
        with tab_feedback:
            feedback_tab(conn, run_id, df, scores, threshold, labels, has_truth)
        with tab_about:
            about_tab()
    finally:
        conn.close()


main()
