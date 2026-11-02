import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).parent.parent
CHARTS = BASE / "charts"
METRICS = BASE / "model" / "decision_tree_metrics.json"
CLEANED = BASE / "data" / "student_cleaned.csv"

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="centered")
st.title("📊 Data Analysis and Decision Tree")
st.write("Exploratory analysis of the (synthetic) student dataset, and a Decision Tree classifier "
         "that predicts **On Track** or **Needs Support**.")

tab_data, tab_charts, tab_tree = st.tabs(["🧹 Data and Cleaning", "📈 Charts", "🌳 Decision Tree"])

with tab_data:
    st.subheader("Cleaned dataset")
    if CLEANED.exists():
        df = pd.read_csv(CLEANED)
        st.write(f"{df.shape[0]} rows and {df.shape[1]} columns after cleaning.")
        st.dataframe(df.head(20))
        st.dataframe(df.describe().round(2))
    else:
        st.warning("data/student_cleaned.csv not found. Run the cleaning cell (Cell E) in the notebook.")
    st.markdown("**Cleaning steps performed:** removed duplicate rows, fixed inconsistent text "
                "(spaces and capital letters), marked impossible values as missing, filled missing "
                "numbers with the median and missing categories with the most common value.")
    st.markdown("**Encoding used:** Label Encoding for the target (On Track / Needs Support), "
                "Ordinal Encoding for motivation level and year (they have a natural order), and "
                "One-Hot Encoding for study mode (no natural order).")

with tab_charts:
    charts = [
        ("1_histogram.png", "Histogram: how performance scores are spread"),
        ("2_kde.png", "KDE plot: smooth distribution of scores for each category"),
        ("3_boxplot.png", "Box plot: score by motivation level"),
        ("4_bar.png", "Bar plot: average score by study mode"),
        ("5_pie.png", "Pie chart: share of On Track and Needs Support records"),
        ("6_scatter.png", "Scatter plot: previous marks vs performance score"),
        ("7_heatmap.png", "Correlation heatmap"),
    ]
    found = False
    for file, caption in charts:
        if (CHARTS / file).exists():
            found = True
            st.image(str(CHARTS / file), caption=caption)
    if not found:
        st.warning("No charts found. Run Cells F, G and H in the notebook first.")

with tab_tree:
    if METRICS.exists():
        m = json.loads(METRICS.read_text())
        st.write("**Algorithm:** Decision Tree Classifier (Scikit-learn), max depth 5.")
        st.write(f"**Rows:** {m['n_train']} for training, {m['n_test']} for testing (split by student).")
        st.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
        st.markdown("**Classification report**")
        rows = []
        for c in m["classes"]:
            r = m["report"][c]
            rows.append({"Class": c, "Precision": round(r["precision"], 3), "Recall": round(r["recall"], 3),
                         "F1-score": round(r["f1-score"], 3), "Support": int(r["support"])})
        st.table(pd.DataFrame(rows))
        st.markdown("**Confusion matrix**")
        if (CHARTS / "8_confusion_matrix.png").exists():
            st.image(str(CHARTS / "8_confusion_matrix.png"))
        if (CHARTS / "9_decision_tree.png").exists():
            st.markdown("**The tree (top 3 levels)**")
            st.image(str(CHARTS / "9_decision_tree.png"))
        st.markdown("**Feature importance**")
        st.bar_chart(pd.Series(m["feature_importance"]).sort_values())
        st.caption("Trained on synthetic data. High scores show the tree learned the data's rule, "
                   "not that it would work on real students.")
    else:
        st.warning("model/decision_tree_metrics.json not found. Run Cell J in the notebook first.")
