from pathlib import Path
import math
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_risk_pipeline.joblib"
TEMPLATE_PATH = BASE_DIR / "cohort_template.csv"
SAMPLE_PATH = BASE_DIR / "sample_cohort.csv"

st.set_page_config(page_title="Student Risk Advisor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_csv_bytes(path):
    return path.read_bytes()

artifact = load_artifact()
model = artifact["pipeline"]
features = artifact["features"]

st.title("🎓 Student Risk Advisor")
st.subheader("Cohort-level academic-risk prioritization")
st.write(
    "Upload a cohort CSV to estimate at-risk probabilities, rank students, and select a review list based on a fixed capacity."
)
st.warning(
    "Research prototype only. Predictions are not diagnoses or verified outcomes. Use results only to support a human-reviewed, supportive conversation—not for punishment, grading, admission, or other high-stakes decisions."
)

with st.expander("How to prepare your CSV", expanded=True):
    st.markdown(
        "- Include the 30 predictor columns shown in the template. Column names must match exactly.\n"
        "- An optional `student_id` column can be included; otherwise rows receive anonymous IDs.\n"
        "- `G1`, `G2`, and `G3` are never used as model inputs and are removed from the exported results.\n"
        "- Categorical values should use the dataset's original labels (for example, `GP`, `F`, `U`, `R`)."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download blank CSV template",
            data=load_csv_bytes(TEMPLATE_PATH),
            file_name="cohort_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download example cohort",
            data=load_csv_bytes(SAMPLE_PATH),
            file_name="sample_cohort.csv",
            mime="text/csv",
            use_container_width=True,
        )

uploaded_file = st.file_uploader("Upload cohort data (.csv)", type=["csv"])
budget = st.select_slider(
    "Review capacity — flag the highest-risk share of the cohort",
    options=[10, 20, 30],
    value=20,
    format_func=lambda value: f"Top {value}%",
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception:
        st.error("The file could not be read as a CSV. Please download the template and try again.")
        st.stop()

    if data.empty:
        st.error("The uploaded CSV has no rows.")
        st.stop()
    if len(data) > 10000:
        st.error("This research prototype is limited to 10,000 rows per upload.")
        st.stop()

    missing = [column for column in features if column not in data.columns]
    if missing:
        st.error("Required predictor columns are missing: " + ", ".join(missing))
        st.info("Download the CSV template above to check the exact column names.")
        st.stop()

    if "student_id" in data.columns:
        identifiers = data["student_id"].fillna("").astype(str).str.strip()
        identifiers = identifiers.where(identifiers.ne(""), [f"Record {i + 1}" for i in range(len(data))])
    else:
        identifiers = pd.Series([f"Record {i + 1}" for i in range(len(data))])

    model_input = data[features].copy()
    try:
        probabilities = model.predict_proba(model_input)[:, 1]
    except Exception as exc:
        st.error("Prediction failed. Check that numeric fields contain numbers and category labels match the dataset format.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        st.stop()

    ranked = data.drop(columns=["G1", "G2", "G3"], errors="ignore").copy()
    ranked.insert(0, "record_id", identifiers.to_numpy())
    ranked["predicted_at_risk_probability"] = probabilities.astype(float)
    ranked = ranked.sort_values("predicted_at_risk_probability", ascending=False, kind="stable").reset_index(drop=True)
    ranked.insert(1, "priority_rank", range(1, len(ranked) + 1))
    slots = max(1, math.ceil(len(ranked) * budget / 100))
    ranked["flag_for_human_review"] = ranked["priority_rank"] <= slots
    ranked["priority_status"] = ranked["flag_for_human_review"].map(
        {True: "Prioritize for human review", False: "Outside current review budget"}
    )

    st.divider()
    st.markdown("### Cohort summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records scored", f"{len(ranked):,}")
    m2.metric("Review slots", f"{slots:,}")
    m3.metric("Selected share", f"{slots / len(ranked):.1%}")
    m4.metric("Highest estimated risk", f"{ranked['predicted_at_risk_probability'].max():.1%}")

    st.markdown("### Prioritized review list")
    st.caption(
        f"The top {budget}% by model score are selected (ceil rule: {slots} of {len(ranked)} records). "
        "A score is an estimate, not a certainty that a student will experience academic difficulty."
    )
    st.dataframe(
        ranked[["record_id", "priority_rank", "predicted_at_risk_probability", "priority_status"]].assign(
            predicted_at_risk_probability=lambda frame: frame["predicted_at_risk_probability"].map(lambda p: f"{p:.1%}")
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View all exported columns"):
        st.dataframe(ranked, use_container_width=True, hide_index=True)

    output = ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranked cohort CSV",
        data=output,
        file_name="student_risk_prioritization.csv",
        mime="text/csv",
        type="primary",
    )
    st.info(
        f"{slots} records are in the review list. No messages are sent and no action is taken automatically. "
        "A qualified person should review context before offering support."
    )
else:
    st.info("Upload a CSV or download the example cohort to try the workflow.")

st.caption(
    "Model: Random Forest research pipeline trained on the UCI Student Performance Portuguese-subject dataset. "
    "The included model artifact is refitted on the full public dataset for demonstration inference; it is not the held-out evaluation model."
)
