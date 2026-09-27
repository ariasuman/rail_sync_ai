"""
app.py
------
RailSync AI — Streamlit Dashboard
Run with: streamlit run app.py
"""

import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

from generate_data import save_data
from preprocess import load_trains, load_maintenance
from traffic_prediction import train_model
from optimizer import find_best_windows, explain_recommendation
from conflict_detection import detect_conflicts, minutes_to_hhmm
from what_if import simulate_window

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RailSync AI",
    page_icon="🚆",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic data...")
def load_data():
    """Generate CSVs (once) and return cleaned DataFrames."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    trains_path = os.path.join(data_dir, "trains.csv")
    maint_path  = os.path.join(data_dir, "maintenance.csv")

    if not os.path.exists(trains_path) or not os.path.exists(maint_path):
        save_data()

    trains = load_trains(trains_path)
    maint  = load_maintenance(maint_path)
    return trains, maint


@st.cache_resource(show_spinner="Training traffic model...")
def get_model(section_id, duration_hours):
    trains, _ = load_data()
    model, mae, _ = train_model(trains, section_id, duration_hours * 60)
    return model, mae


def hhmm_to_min(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚆 RailSync AI — Smart Maintenance Block Planner")
st.caption("AI-powered decision support for Indian Railways maintenance scheduling")
st.divider()

# ── Sidebar — Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Maintenance Parameters")

    trains_df, maint_df = load_data()

    sections = sorted(trains_df["section_id"].unique().tolist())
    section_id = st.selectbox("🛤️ Railway Section", sections)

    assets_in_section = maint_df[maint_df["section_id"] == section_id]
    if assets_in_section.empty:
        assets_in_section = maint_df  # fallback

    asset_options = assets_in_section["asset_id"].tolist()
    asset_id = st.selectbox("🔧 Asset ID", asset_options)

    selected_asset = maint_df[maint_df["asset_id"] == asset_id].iloc[0] \
        if asset_id in maint_df["asset_id"].values else maint_df.iloc[0]

    maint_type = st.selectbox(
        "🛠️ Maintenance Type",
        ["Routine Inspection", "Track Tamping", "Rail Replacement",
         "Signal Calibration", "Bridge Inspection", "OHE Maintenance",
         "Points Lubrication", "Ballast Cleaning"],
        index=0,
    )

    duration_hours = st.slider("⏱️ Duration (hours)", 1, 8,
                                int(selected_asset.get("maintenance_duration_hours", 2)))

    priority = st.selectbox(
        "🚨 Maintenance Priority",
        ["Low", "Medium", "High", "Critical"],
        index=["Low", "Medium", "High", "Critical"].index(
            selected_asset.get("maintenance_priority", "Medium")
        ),
    )

    deadline = st.date_input(
        "📅 Maintenance Deadline",
        value=pd.to_datetime(selected_asset.get("maintenance_deadline",
                                                  "2025-06-30")).date()
        if pd.notna(selected_asset.get("maintenance_deadline")) else date(2025, 6, 30),
    )

    worker_available  = st.checkbox("👷 Worker Available",
                                     value=bool(selected_asset.get("worker_available", True)))
    machine_available = st.checkbox("🏗️ Machine Available",
                                     value=bool(selected_asset.get("machine_available", True)))

    st.divider()
    find_btn = st.button("🔍 Find Best Maintenance Block", use_container_width=True,
                          type="primary")

# ── Main Panel ────────────────────────────────────────────────────────────────
if find_btn:
    with st.spinner("Optimizing maintenance windows..."):
        model, mae = get_model(section_id, duration_hours)

        results = find_best_windows(
            trains_df=trains_df,
            section_id=section_id,
            duration_hours=duration_hours,
            maintenance_priority=priority,
            deadline=str(deadline),
            worker_available=worker_available,
            machine_available=machine_available,
            traffic_model=model,
            top_n=3,
        )

    if results.empty:
        st.error("No feasible maintenance windows found. Try adjusting parameters.")
        st.stop()

    best = results.iloc[0]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.subheader("📊 Best Recommended Window")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("⏰ Window", f"{best['start_time']} – {best['end_time']}")
    k2.metric("🚆 Affected Trains", int(best["affected_trains"]))
    k3.metric("⏳ Est. Delay", f"{int(best['total_delay_min'])} min")
    k4.metric("🎯 Opt. Score", f"{best['score']:.4f}")

    # ── Explanation ───────────────────────────────────────────────────────────
    explanation = explain_recommendation(best, trains_df, section_id)
    st.info(f"💡 **Why this window?**\n\n{explanation}")

    st.divider()

    # ── Top 3 Alternative Windows ─────────────────────────────────────────────
    st.subheader("🏆 Top 3 Recommended Windows")
    cols = st.columns(3)
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i]:
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"### {rank_emoji} Option {i+1}")
            st.write(f"**Window:** {row['start_time']} – {row['end_time']}")
            st.write(f"**Affected Trains:** {int(row['affected_trains'])}")
            st.write(f"**Est. Delay:** {int(row['total_delay_min'])} min")
            st.write(f"**Worker:** {'✅' if row['worker_available'] else '❌'}")
            st.write(f"**Machine:** {'✅' if row['machine_available'] else '❌'}")
            st.write(f"**Priority:** {row['maintenance_priority']}")
            st.write(f"**Score:** `{row['score']:.4f}`")

    st.divider()

    # ── Bar Chart: Comparing Windows ──────────────────────────────────────────
    st.subheader("📈 Window Comparison")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_bar = px.bar(
            results,
            x=[f"Option {i+1}\n{r['start_time']}–{r['end_time']}"
               for i, r in results.iterrows()],
            y="affected_trains",
            color="total_delay_min",
            color_continuous_scale="RdYlGn_r",
            labels={"x": "Window", "affected_trains": "Affected Trains",
                    "total_delay_min": "Delay (min)"},
            title="Affected Trains per Window",
        )
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        fig_score = px.bar(
            results,
            x=[f"Option {i+1}" for i in range(len(results))],
            y="score",
            color="score",
            color_continuous_scale="RdYlGn_r",
            title="Optimization Score (lower = better)",
            labels={"x": "Window", "score": "Score"},
        )
        fig_score.update_layout(height=350)
        st.plotly_chart(fig_score, use_container_width=True)

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.subheader("🗓️ Train & Maintenance Timeline")

    section_trains = trains_df[trains_df["section_id"] == section_id].copy()
    today = "2025-06-01"

    timeline_data = []
    for _, t in section_trains.iterrows():
        arr = int(t["arr_min"]) % 1440
        dep = int(t["dep_min"]) % 1440
        timeline_data.append(dict(
            Task=t["train_id"],
            Start=f"{today} {minutes_to_hhmm(arr)}",
            Finish=f"{today} {minutes_to_hhmm(dep)}",
            Type="Train",
            Priority=t["train_priority"],
        ))

    # Add best maintenance window
    timeline_data.append(dict(
        Task="🔧 Maintenance",
        Start=f"{today} {best['start_time']}",
        Finish=f"{today} {best['end_time']}",
        Type="Maintenance",
        Priority=priority,
    ))

    tl_df = pd.DataFrame(timeline_data)
    color_map = {"Train": "#4C9BE8", "Maintenance": "#FF6B35"}

    fig_tl = px.timeline(
        tl_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Type",
        color_discrete_map=color_map,
        title=f"Train Schedule & Maintenance Block — Section {section_id}",
        hover_data=["Priority"],
    )
    fig_tl.update_yaxes(autorange="reversed")
    fig_tl.update_layout(height=max(300, len(timeline_data) * 18))
    st.plotly_chart(fig_tl, use_container_width=True)

    # ── Affected Trains Table ─────────────────────────────────────────────────
    st.subheader("🚆 Affected Trains Detail")
    affected, _ = detect_conflicts(
        trains_df, section_id,
        hhmm_to_min(best["start_time"]),
        hhmm_to_min(best["end_time"]),
    )
    if affected.empty:
        st.success("✅ No trains are affected during this window!")
    else:
        st.dataframe(
            affected[["train_id", "train_name", "section_id",
                       "arrival_time", "departure_time",
                       "train_priority", "estimated_delay_min"]],
            use_container_width=True,
        )

    st.divider()

    # ── What-If Simulator ─────────────────────────────────────────────────────
    st.subheader("🔬 What-If Simulator")
    st.caption("Test a custom maintenance window to see its impact.")

    wi_col1, wi_col2 = st.columns(2)
    with wi_col1:
        wi_start = st.text_input("Custom Start Time (HH:MM)", value="02:00")
    with wi_col2:
        wi_end = st.text_input("Custom End Time (HH:MM)", value="04:00")

    if st.button("▶️ Simulate This Window"):
        wi_result = simulate_window(
            trains_df, section_id, wi_start, wi_end,
            priority, str(deadline),
            worker_available, machine_available, model,
        )
        if "error" in wi_result:
            st.error(wi_result["error"])
        else:
            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric("Affected Trains", wi_result["affected_trains"])
            wc2.metric("Est. Delay", f"{wi_result['total_delay_min']} min")
            wc3.metric("Score", f"{wi_result['score']:.4f}")
            wc4.metric("Feasible", "✅ Yes" if wi_result["feasible"] else "❌ No")
            st.info(wi_result["explanation"])

            if wi_result["affected_train_list"]:
                st.dataframe(pd.DataFrame(wi_result["affected_train_list"]),
                             use_container_width=True)

else:
    # Landing state
    st.markdown("""
    ### Welcome to RailSync AI 🚆

    This system helps Indian Railways maintenance planners find the **optimal time window**
    to perform track and asset maintenance with **minimum disruption** to train operations.

    **How to use:**
    1. Select a **Railway Section** and **Asset** from the sidebar
    2. Set the **maintenance type**, **duration**, **priority**, and **deadline**
    3. Confirm **worker** and **machine** availability
    4. Click **"Find Best Maintenance Block"**

    The AI will analyze train traffic, detect conflicts, and recommend the **top 3 optimal windows**.

    ---
    > ⚠️ *This is a prototype decision-support system. It does NOT connect to real railway systems
    > and must NOT be used to directly control signals or train operations.*
    """)

    trains_df, maint_df = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("🚆 Train Records", len(trains_df))
    c2.metric("🔧 Maintenance Tasks", len(maint_df))
    c3.metric("🛤️ Railway Sections", trains_df["section_id"].nunique())
