"""
NIDS Dashboard — Streamlit client for the FastAPI-based detection API.
Run: streamlit run dashboard/app.py
The dashboard will automatically start the FastAPI backend if it isn't already running.
"""
# ── sys.path fix ──────────────────────────────────────────────────────────────
# Streamlit sets sys.path[0] to the script's own directory (dashboard/), so
# sibling packages like `data/` are not importable without this fix.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

API_URL = os.environ.get("NIDS_API_URL", "http://localhost:8000")

# ── Backend auto-start ────────────────────────────────────────────────────────

def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if something is already listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_backend_running() -> None:
    """
    Spawn the FastAPI backend automatically if it isn't already running.

    - Only auto-spawns when the API_URL points to localhost (skips if NIDS_API_URL
      is set to a remote address — Docker, Render, etc.).
    - Idempotent: port-check gates the spawn, so repeated Streamlit reruns never
      create duplicate uvicorn processes.
    - Shows a spinner while waiting for /health to respond (up to 15 s).
    - Shows a clear st.error with stderr tail if it never comes up.
    """
    parsed = urlparse(API_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000

    # Don't auto-spawn for remote URLs (Docker, cloud deployments).
    if host not in ("localhost", "127.0.0.1", "0.0.0.0"):
        return

    # If port is already bound (API already running), nothing to do.
    if _is_port_open(host, port):
        return

    # Spawn uvicorn as a background process that survives Streamlit reruns.
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", str(port),
        ],
        cwd=str(PROJECT_ROOT),
        start_new_session=True,          # detach from Streamlit's process group
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Poll /health until it responds 200 or we time out (~15 s).
    deadline = time.time() + 15
    started = False
    with st.spinner("🚀 Starting NIDS backend..."):
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                r = requests.get(f"{API_URL}/health", timeout=2)
                if r.status_code == 200:
                    started = True
                    break
            except Exception:
                pass

    if started:
        st.toast("✅ Backend started automatically.", icon="🛡️")
    else:
        # Try to surface stderr for a clear error message.
        try:
            _, stderr_bytes = proc.communicate(timeout=1)
            err_tail = stderr_bytes.decode(errors="replace")[-800:]
        except Exception:
            err_tail = "(could not read stderr)"
        st.error(
            "❌ **Could not start the NIDS backend** within 15 seconds.\n\n"
            "Make sure you have run `python train_model.py` first, then try:\n"
            "```\nuvicorn api.main:app --port 8000\n```\n\n"
            f"**Backend stderr (tail):**\n```\n{err_tail}\n```"
        )


# st.set_page_config() MUST be the very first Streamlit command executed in the
# script (Streamlit enforces this at runtime). ensure_backend_running() calls
# st.spinner/st.toast/st.error internally, so set_page_config has to run first —
# calling it after caused a StreamlitSetPageConfigMustBeFirstCommandError crash
# on first-ever launch (the exact scenario auto-start exists for).
st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

# Run once per page load — idempotent due to port check inside.
ensure_backend_running()

# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "Critical": "#d32f2f", "High": "#f57c00", "Medium": "#fbc02d",
    "Low": "#388e3c", "Info": "#1976d2",
}


# If NIDS_API_KEY is set (same var the API's verify_api_key dependency checks),
# attach it automatically so enabling API auth doesn't lock the dashboard itself out.
_DASHBOARD_API_KEY = os.environ.get("NIDS_API_KEY", "").strip() or None


def _with_auth_headers(kwargs: dict) -> dict:
    if _DASHBOARD_API_KEY:
        headers = kwargs.pop("headers", {}) or {}
        headers["X-API-Key"] = _DASHBOARD_API_KEY
        kwargs["headers"] = headers
    return kwargs


def api_get(path, **kwargs):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10, **_with_auth_headers(kwargs))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


def api_post(path, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=30, **_with_auth_headers(kwargs))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


st.title("🛡️ Network Intrusion Detection System")
st.caption(f"Connected to API: `{API_URL}`")

health = api_get("/health")
if health:
    st.success("API is reachable", icon="✅")

tab1, tab2, tab3 = st.tabs(["📡 Live Monitor", "📁 Batch Analysis", "📊 Analytics & Model Report"])

# ─────────────────────────────────────────────────────────────
# TAB 1: LIVE MONITOR (simulated stream from test dataset)
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Simulated Live Traffic Monitor")
    st.caption(
        "Replays real NSL-KDD test flows through the detection API to mimic a live feed. "
        "This avoids needing raw packet capture privileges on your laptop."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        speed = st.slider("Flows per tick", 1, 20, 5)
    with col2:
        delay = st.slider("Delay per tick (sec)", 0.1, 3.0, 0.5)
    with col3:
        n_flows = st.number_input("Total flows to replay", 10, 2000, 100)
    with col4:
        refresh_interval = st.selectbox(
            "Auto-refresh interval",
            options=["Off", "5s", "10s", "30s", "60s"],
            index=0,
            help="Auto-refresh the monitor at this interval (separate from stream tick).",
        )

    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "stream_idx" not in st.session_state:
        st.session_state.stream_idx = 0

    start_col, stop_col = st.columns(2)
    if start_col.button("▶️ Start Stream", use_container_width=True):
        st.session_state.streaming = True
        st.session_state.stream_idx = 0
    if stop_col.button("⏹️ Stop Stream", use_container_width=True):
        st.session_state.streaming = False

    live_table = st.empty()
    live_metrics = st.empty()

    if st.session_state.streaming:
        try:
            sample_df = pd.read_csv(
                PROJECT_ROOT / "data" / "raw" / "KDDTest.txt", header=None, nrows=2000
            )
            from data.columns import COLUMNS
            sample_df.columns = COLUMNS[: sample_df.shape[1]]
        except Exception as e:
            st.error(f"Could not load sample dataset: {e}")
            sample_df = None

        if sample_df is not None:
            recent_rows = []
            total_normal, total_attack = 0, 0

            for _ in range(int(n_flows // speed)):
                if not st.session_state.streaming:
                    break
                batch = sample_df.iloc[
                    st.session_state.stream_idx: st.session_state.stream_idx + speed
                ]
                if batch.empty:
                    st.session_state.stream_idx = 0
                    continue

                for _, row in batch.iterrows():
                    record = row.drop(
                        labels=[c for c in ["label", "difficulty"] if c in row.index]
                    ).to_dict()
                    record["src_ip"] = f"10.0.{row.name % 255}.{(row.name * 7) % 255}"
                    record["dst_ip"] = "192.168.1.100"
                    result = api_post("/predict", json=record)
                    if result:
                        recent_rows.insert(0, {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "src_ip": record["src_ip"],
                            "protocol": record["protocol_type"],
                            "category": result["predicted_category"],
                            "confidence": result["confidence"],
                            "severity": result["severity"],
                        })
                        if result["predicted_category"] == "Normal":
                            total_normal += 1
                        else:
                            total_attack += 1

                st.session_state.stream_idx += speed
                recent_rows = recent_rows[:30]

                with live_metrics.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Normal Flows", total_normal)
                    m2.metric("Attack Flows", total_attack)
                    m3.metric(
                        "Attack Rate",
                        f"{(total_attack / max(1, total_normal + total_attack)) * 100:.1f}%",
                    )

                with live_table.container():
                    df_display = pd.DataFrame(recent_rows)
                    if not df_display.empty:
                        st.dataframe(df_display, use_container_width=True, height=400)

                time.sleep(delay)

            st.session_state.streaming = False
            st.info("Stream finished. Adjust settings and start again.")

    # Auto-refresh logic (outside the streaming block)
    if refresh_interval != "Off" and not st.session_state.streaming:
        interval_map = {"5s": 5, "10s": 10, "30s": 30, "60s": 60}
        secs = interval_map.get(refresh_interval, 0)
        if secs:
            time.sleep(secs)
            st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 2: BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Dataset Analysis")
    st.caption("Upload a CSV of network flows (NSL-KDD schema) for bulk classification.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv", "txt"])
    if uploaded is not None:
        with st.spinner("Running detection pipeline..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            result = api_post("/predict/batch", files=files)

        if result:
            st.success(f"Processed {result['total_rows']} flows")

            c1, c2 = st.columns(2)
            with c1:
                pred_df = pd.DataFrame(
                    list(result["predictions"].items()), columns=["Category", "Count"]
                )
                fig = px.pie(pred_df, names="Category", values="Count",
                              title="Predicted Attack Category Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                sev_df = pd.DataFrame(
                    list(result["severity_breakdown"].items()), columns=["Severity", "Count"]
                )
                fig2 = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                               color_discrete_map=SEVERITY_COLORS,
                               title="Severity Breakdown")
                st.plotly_chart(fig2, use_container_width=True)

            st.metric("Alerts Generated (non-Normal)", result["alerts_generated"])
            st.subheader("Sample Results (first 50 rows)")
            st.dataframe(pd.DataFrame(result["results_preview"]), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: ANALYTICS & MODEL REPORT
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Performance Report")
    metadata = api_get("/metrics")

    if metadata:
        st.markdown(f"**Best Model Selected:** `{metadata['best_model']}`")

        # Show training metadata if available
        meta_cols = st.columns(3)
        if metadata.get("training_date"):
            meta_cols[0].metric("Training Date", metadata["training_date"][:10])
        if metadata.get("library_versions", {}).get("scikit-learn"):
            meta_cols[1].metric("scikit-learn", metadata["library_versions"]["scikit-learn"])
        if metadata.get("dataset_hash"):
            meta_cols[2].metric("Dataset Hash", metadata["dataset_hash"][:8] + "…")

        results_df = pd.DataFrame(metadata["all_results"]).T.reset_index()
        results_df.rename(columns={"index": "model"}, inplace=True)
        st.dataframe(results_df, use_container_width=True)

        # Cross-validation results
        if metadata.get("cv_scores"):
            st.subheader("Cross-Validation Results (5-fold)")
            cv_df = pd.DataFrame(metadata["cv_scores"]).T.reset_index()
            cv_df.rename(columns={"index": "model"}, inplace=True)
            st.dataframe(cv_df, use_container_width=True)

        fig = px.bar(
            results_df, x="model", y=["accuracy", "precision", "recall", "f1_score"],
            barmode="group", title="Model Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        if metadata.get("feature_importance"):
            fi_df = pd.DataFrame(
                list(metadata["feature_importance"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                              title="Top 15 Feature Importances (Best Model)")
            st.plotly_chart(fig_fi, use_container_width=True)

        if metadata.get("confusion_matrix"):
            cm = metadata["confusion_matrix"]
            classes = metadata["class_names"]
            fig_cm = px.imshow(
                cm, x=classes, y=classes, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                title="Confusion Matrix (Test Set)", color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    st.subheader("Alert History (from live SQLite log)")
    stats = api_get("/alerts/stats")
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Alerts Logged", stats["total_alerts"])
        c2.metric("Distinct Categories", len(stats["by_category"]))
        c3.metric("Top Attack Sources", len(stats["top_sources"]))

        if stats["by_category"]:
            cat_df = pd.DataFrame(
                list(stats["by_category"].items()), columns=["Category", "Count"]
            )
            st.plotly_chart(
                px.bar(cat_df, x="Category", y="Count", title="Alerts by Category"),
                use_container_width=True,
            )

        if stats["top_sources"]:
            st.markdown("**Top Attacking Source IPs**")
            st.dataframe(pd.DataFrame(stats["top_sources"]), use_container_width=True)

    # ── Alert filtering ──────────────────────────────────────────────────────
    recent_all = api_get("/alerts?limit=500")
    if recent_all:
        st.markdown("**Recent Alerts (filterable)**")
        st.caption(
            "Click **Explain** on any alert to get an AI-generated explanation "
            "of why the traffic looks malicious (powered by Groq LLM)."
        )

        # Build filter widgets
        all_severities = sorted({r.get("severity", "") for r in recent_all if r.get("severity")})
        all_categories = sorted({r.get("predicted_category", "") for r in recent_all if r.get("predicted_category")})

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            filter_severity = st.multiselect(
                "Filter by Severity", options=all_severities, default=all_severities,
                key="filter_severity"
            )
        with fcol2:
            filter_category = st.multiselect(
                "Filter by Category", options=all_categories, default=all_categories,
                key="filter_category"
            )
        with fcol3:
            filter_ip = st.text_input("Filter by Source IP (partial match)", key="filter_ip")

        # Apply filters
        recent = [
            r for r in recent_all
            if r.get("severity") in filter_severity
            and r.get("predicted_category") in filter_category
            and (not filter_ip or filter_ip.lower() in (r.get("src_ip") or "").lower())
        ]

        st.caption(f"Showing {len(recent)} of {len(recent_all)} alerts")

        # CSV export of filtered results
        if recent:
            export_df = pd.DataFrame(recent)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download filtered alerts (CSV)",
                data=csv_bytes,
                file_name=f"nids_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_alerts",
            )

        for row in recent[:50]:  # display cap to avoid UI freeze
            alert_id = row.get("id", "?")
            category = row.get("predicted_category", "Unknown")
            severity = row.get("severity", "")
            src_ip = row.get("src_ip", "")
            ts = row.get("timestamp", "")[:19].replace("T", " ")
            sev_icon = {
                "Critical": "🔴", "High": "🟠", "Medium": "🟡",
                "Low": "🟢", "Info": "🔵",
            }.get(severity, "⚪")
            label = (
                f"{sev_icon} **#{alert_id}** — `{category}` — "
                f"src: `{src_ip}` — {ts}"
            )
            with st.expander(label, expanded=False):
                col_details, col_btn = st.columns([4, 1])
                with col_details:
                    st.json({
                        k: row[k] for k in
                        ["id", "timestamp", "src_ip", "dst_ip",
                         "predicted_category", "confidence", "severity"]
                        if k in row
                    })
                with col_btn:
                    btn_key = f"explain_{alert_id}"
                    if st.button("🤖 Explain", key=btn_key, use_container_width=True):
                        with st.spinner("Asking Groq LLM..."):
                            resp = api_get(f"/alerts/{alert_id}/explain")
                        if resp:
                            cached_label = " *(cached)*" if resp.get("cached") else ""
                            st.info(
                                f"**AI Explanation**{cached_label}\n\n"
                                + resp.get("explanation", "No explanation returned."),
                                icon="🛡️",
                            )

    if st.button("🗑️ Clear Alert Log"):
        try:
            requests.delete(f"{API_URL}/alerts", **_with_auth_headers({}), timeout=10)
        except Exception as e:
            st.error(f"API error (/alerts DELETE): {e}")
        st.rerun()
