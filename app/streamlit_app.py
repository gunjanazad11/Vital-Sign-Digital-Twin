import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

RESULTS_PATH = os.path.join("data", "results", "final_digital_twin_results.csv")


def reliability_status(value: float) -> str:
    if value >= 0.85:
        return "Reliable"
    if value >= 0.70:
        return "Watch"
    return "Unreliable"


def signal_action(status: str) -> str:
    if status == "Reliable":
        return "No immediate sensor action"
    if status == "Watch":
        return "Monitor trend / consider repeat reading"
    return "Recheck sensor placement or repeat measurement"


def overall_alert(hri_category: str, confidence: str) -> str:
    if hri_category == "Critical":
        return "Escalate clinical review"
    if hri_category == "Warning":
        return "Continue observation and reassess"
    if confidence == "Low Confidence":
        return "Stable risk but low confidence; verify inputs"
    return "Stable monitoring status"


st.set_page_config(page_title="Clinical Vital Validation Dashboard", layout="wide")

st.title("Clinical Vital Sign Reliability Monitor")
st.caption("Hospital-oriented dashboard for validating readings, correcting suspicious values, and estimating patient risk.")

if not os.path.exists(RESULTS_PATH):
    st.error("Results file not found. Run `python src/main.py` first.")
    st.stop()

df = pd.read_csv(RESULTS_PATH)

if "HRI_confidence" not in df.columns:
    df["HRI_confidence"] = "Moderate Confidence"
if "HRI_uncertainty" not in df.columns:
    df["HRI_uncertainty"] = 0.0

st.sidebar.header("Monitoring Controls")
n_rows = st.sidebar.slider("Time steps to display", min_value=100, max_value=2000, value=500, step=100)
record_idx = st.sidebar.slider("Patient record index", min_value=0, max_value=len(df) - 1, value=0, step=1)

selected = df.iloc[record_idx]

hr_status = reliability_status(selected["HR_rel"])
rr_status = reliability_status(selected["RR_rel"])
spo2_status = reliability_status(selected["SpO2_rel"])
alert_text = overall_alert(selected["HRI_category"], selected["HRI_confidence"])

st.markdown("## Patient Snapshot")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Health Risk Index", f"{selected['HRI']:.3f}", selected["HRI_category"])
col2.metric("Risk Confidence", selected["HRI_confidence"], f"Unc. {selected['HRI_uncertainty']:.3f}")
col3.metric("Heart Rate Status", hr_status, f"Rel. {selected['HR_rel']:.2f}")
col4.metric("Resp/Oxygen Status", f"{rr_status} / {spo2_status}", "RR / SpO2")

if selected["HRI_category"] == "Critical":
    st.error(f"Clinical Alert: {alert_text}")
elif selected["HRI_category"] == "Warning":
    st.warning(f"Clinical Alert: {alert_text}")
else:
    st.success(f"Clinical Alert: {alert_text}")

st.markdown("## Signal Validation Summary")
summary_df = pd.DataFrame(
    {
        "Signal": ["Heart Rate", "Respiratory Rate", "SpO2"],
        "Observed": [selected["HR_observed"], selected["RR_observed"], selected["SpO2_observed"]],
        "Corrected": [selected["HR_corrected"], selected["RR_corrected"], selected["SpO2_corrected"]],
        "Reliability": [selected["HR_rel"], selected["RR_rel"], selected["SpO2_rel"]],
        "Status": [hr_status, rr_status, spo2_status],
        "Suggested Action": [signal_action(hr_status), signal_action(rr_status), signal_action(spo2_status)],
    }
)
st.dataframe(summary_df, width="stretch")

st.markdown("## Risk Distribution")
dist_col1, dist_col2 = st.columns(2)
with dist_col1:
    st.bar_chart(df["HRI_category"].value_counts())
with dist_col2:
    st.bar_chart(df["HRI_confidence"].value_counts())

st.markdown("## Health Risk Trend")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df["HRI"][:n_rows], color="red", label="HRI")
ax1.set_title("Health Risk Index Over Time")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("HRI")
ax1.legend()
st.pyplot(fig1)

st.markdown("## Observed vs Corrected Vital Signs")
fig2, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(df["HR_observed"][:n_rows], label="Observed HR", alpha=0.7)
axes[0].plot(df["HR_corrected"][:n_rows], label="Corrected HR")
axes[0].set_title("Heart Rate Validation")
axes[0].legend()

axes[1].plot(df["RR_observed"][:n_rows], label="Observed RR", alpha=0.7, color="orange")
axes[1].plot(df["RR_corrected"][:n_rows], label="Corrected RR", color="brown")
axes[1].set_title("Respiratory Rate Validation")
axes[1].legend()

axes[2].plot(df["SpO2_observed"][:n_rows], label="Observed SpO2", alpha=0.7, color="green")
axes[2].plot(df["SpO2_corrected"][:n_rows], label="Corrected SpO2", color="darkgreen")
axes[2].set_title("Oxygen Saturation Validation")
axes[2].legend()

plt.tight_layout()
st.pyplot(fig2)

st.markdown("## Signal Reliability")
fig3, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(df["HR_rel"][:n_rows], label="HR Reliability")
axes[0].axhline(0.7, linestyle="--", color="red", linewidth=1)
axes[0].set_title("Heart Rate Reliability")
axes[0].legend()

axes[1].plot(df["RR_rel"][:n_rows], label="RR Reliability", color="orange")
axes[1].axhline(0.7, linestyle="--", color="red", linewidth=1)
axes[1].set_title("Respiratory Rate Reliability")
axes[1].legend()

axes[2].plot(df["SpO2_rel"][:n_rows], label="SpO2 Reliability", color="green")
axes[2].axhline(0.7, linestyle="--", color="red", linewidth=1)
axes[2].set_title("Oxygen Saturation Reliability")
axes[2].legend()

plt.tight_layout()
st.pyplot(fig3)

st.markdown("## Detailed Monitoring Records")
display_cols = [
    "HR_observed",
    "HR_corrected",
    "HR_rel",
    "RR_observed",
    "RR_corrected",
    "RR_rel",
    "SpO2_observed",
    "SpO2_corrected",
    "SpO2_rel",
    "HRI",
    "HRI_category",
    "HRI_confidence",
]
st.dataframe(df[display_cols].head(20), width="stretch")

