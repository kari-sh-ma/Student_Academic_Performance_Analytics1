import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Academic Performance Analytics", page_icon="📊", layout="wide")
st.title("📊 Student Academic Performance Analytics")
st.caption("Analytics dashboard with demo ML estimates, early-warning flags, and study recommendations.")

DEFAULT_DATA = Path(__file__).parent / "data" / "student_performance.csv"
uploaded_file = st.sidebar.file_uploader("Upload student CSV dataset", type=["csv"])

@st.cache_data
def load_data(source):
    return pd.read_csv(source)

try:
    df = load_data(uploaded_file if uploaded_file is not None else DEFAULT_DATA).copy()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
df = df.drop_duplicates()
required = ["Gender", "Attendance", "Study_Hours", "Assignment_Score",
            "Internal_Marks", "Previous_Marks", "Final_Marks", "Extracurricular"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.info("Required columns: " + ", ".join(required))
    st.stop()

numeric_cols = ["Age", "Attendance", "Study_Hours", "Assignment_Score",
                "Internal_Marks", "Previous_Marks", "Final_Marks"]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Final_Marks"]).copy()
if df.empty:
    st.error("No valid rows found in the dataset.")
    st.stop()

st.sidebar.header("Dashboard Filters")
genders = ["All"] + sorted(df["Gender"].dropna().astype(str).unique().tolist())
extras = ["All"] + sorted(df["Extracurricular"].dropna().astype(str).unique().tolist())
g_filter = st.sidebar.selectbox("Gender", genders)
e_filter = st.sidebar.selectbox("Extracurricular", extras)
filtered = df.copy()
if g_filter != "All":
    filtered = filtered[filtered["Gender"].astype(str) == g_filter]
if e_filter != "All":
    filtered = filtered[filtered["Extracurricular"].astype(str) == e_filter]
if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

def category(mark):
    if pd.isna(mark): return "Unknown"
    if mark >= 80: return "High"
    if mark >= 60: return "Medium"
    return "Low"

filtered["Performance_Category"] = filtered["Final_Marks"].apply(category)

# Dashboard KPIs
a, b, c, d = st.columns(4)
a.metric("Students", len(filtered))
b.metric("Average Final Marks", f"{filtered['Final_Marks'].mean():.1f}")
c.metric("Average Attendance", f"{filtered['Attendance'].mean():.1f}%")
d.metric("Average Study Hours", f"{filtered['Study_Hours'].mean():.1f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🤖 ML Estimate & Early Warning",
    "🎯 Study Recommendations", "📋 Student Records & Export"
])

with tab1:
    st.subheader("Academic Performance Visualizations")
    left, right = st.columns(2)
    with left:
        counts = filtered["Performance_Category"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_xlabel("Performance Category"); ax.set_ylabel("Students")
        ax.set_title("Performance Category Distribution")
        st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots()
        ax.scatter(filtered["Attendance"], filtered["Final_Marks"])
        ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Final Marks")
        ax.set_title("Attendance vs Final Marks")
        st.pyplot(fig); plt.close(fig)
    left2, right2 = st.columns(2)
    with left2:
        fig, ax = plt.subplots()
        ax.scatter(filtered["Study_Hours"], filtered["Final_Marks"])
        ax.set_xlabel("Study Hours per Day"); ax.set_ylabel("Final Marks")
        ax.set_title("Study Hours vs Final Marks")
        st.pyplot(fig); plt.close(fig)
    with right2:
        score_cols = ["Assignment_Score", "Internal_Marks", "Previous_Marks", "Final_Marks"]
        avg = filtered[score_cols].mean()
        fig, ax = plt.subplots()
        ax.bar(["Assignment", "Internal", "Previous", "Final"], avg.values)
        ax.set_ylabel("Average score"); ax.set_title("Average Academic Scores")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Key Insights")
    st.write(f"- High performers: **{(filtered['Performance_Category'] == 'High').sum()}**")
    st.write(f"- Medium performers: **{(filtered['Performance_Category'] == 'Medium').sum()}**")
    st.write(f"- Low performers: **{(filtered['Performance_Category'] == 'Low').sum()}**")
    corr = filtered["Attendance"].corr(filtered["Final_Marks"])
    if pd.notna(corr):
        st.write(f"- Attendance/final-marks correlation: **{corr:.2f}**")
        st.caption("Correlation shows association, not proof of causation.")

# ML estimate: train a demonstration model using historical rows
with tab2:
    st.subheader("Machine Learning Final-Mark Estimate")
    st.warning("Demonstration only: estimates are not official grades or guaranteed predictions. Use representative, validated historical data before real decisions.")
    feature_cols = ["Gender", "Attendance", "Study_Hours", "Assignment_Score",
                    "Internal_Marks", "Previous_Marks", "Extracurricular"]
    model_df = df.dropna(subset=["Final_Marks"]).copy()
    X = model_df[feature_cols]
    y = model_df["Final_Marks"]
    categorical = ["Gender", "Extracurricular"]
    numeric = ["Attendance", "Study_Hours", "Assignment_Score", "Internal_Marks", "Previous_Marks"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=120, random_state=42))
    ])
    if len(model_df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        m1, m2 = st.columns(2)
        m1.metric("Test MAE", f"{mean_absolute_error(y_test, pred_test):.2f} marks")
        try:
            m2.metric("Test R²", f"{r2_score(y_test, pred_test):.2f}")
        except Exception:
            m2.metric("Test R²", "Unavailable")
        model.fit(X, y)
        st.caption("Metrics are based on the included sample dataset. They do not establish real-world accuracy.")
        st.markdown("#### Enter student details for a demo estimate")
        p1, p2, p3 = st.columns(3)
        with p1:
            p_gender = st.selectbox("Gender for estimate", sorted(df["Gender"].dropna().astype(str).unique()), key="pg")
            p_att = st.slider("Attendance (%)", 0, 100, 75)
        with p2:
            p_hours = st.slider("Study hours per day", 0.0, 12.0, 2.5, 0.5)
            p_assign = st.slider("Assignment score", 0, 100, 70)
        with p3:
            p_internal = st.slider("Internal marks", 0, 100, 68)
            p_previous = st.slider("Previous marks", 0, 100, 65)
            p_extra = st.selectbox("Extracurricular", ["Yes", "No"], key="pe")
        if st.button("Estimate final marks"):
            one = pd.DataFrame([{
                "Gender": p_gender, "Attendance": p_att, "Study_Hours": p_hours,
                "Assignment_Score": p_assign, "Internal_Marks": p_internal,
                "Previous_Marks": p_previous, "Extracurricular": p_extra
            }])
            estimated = float(model.predict(one)[0])
            st.success(f"Estimated final marks: **{estimated:.1f}/100**")
            if estimated < 60 or p_att < 75:
                st.error("Early-warning flag: consider checking in with this student.")
            else:
                st.info("No warning triggered by the demo thresholds. Continue monitoring progress.")
    else:
        st.info("At least 10 rows with final marks are needed for the demonstration model.")

with tab3:
    st.subheader("Personalized Study Recommendations")
    st.write("These simple rule-based suggestions are intended as guidance, not as a judgement of a student.")
    rec_df = filtered.copy()
    def recommend(row):
        tips = []
        if pd.notna(row["Attendance"]) and row["Attendance"] < 75:
            tips.append("Improve attendance and discuss barriers with a teacher.")
        if pd.notna(row["Study_Hours"]) and row["Study_Hours"] < 2:
            tips.append("Try a regular daily study routine, starting with 30–45 minute sessions.")
        if pd.notna(row["Assignment_Score"]) and row["Assignment_Score"] < 60:
            tips.append("Review assignment feedback and practise similar questions.")
        if pd.notna(row["Internal_Marks"]) and row["Internal_Marks"] < 60:
            tips.append("Revise internal-assessment topics and ask for help on difficult concepts.")
        if pd.notna(row["Final_Marks"]) and row["Final_Marks"] < 60:
            tips.append("Create a weekly improvement plan with teacher support.")
        return " ".join(tips) if tips else "Maintain current habits and continue reviewing progress."
    rec_df["Recommendations"] = rec_df.apply(recommend, axis=1)
    show_cols = [c for c in ["Student_ID", "Gender", "Attendance", "Study_Hours", "Final_Marks", "Recommendations"] if c in rec_df.columns]
    st.dataframe(rec_df[show_cols], use_container_width=True)
    st.caption("Recommendations are generated from simple thresholds, not a trained recommendation engine.")

with tab4:
    st.subheader("Student Records")
    st.dataframe(filtered, use_container_width=True)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered student data (CSV)", csv_bytes,
                       "filtered_student_performance.csv", "text/csv")
    # Download rule-based recommendations
    rec_export = filtered.copy()
    rec_export["Recommendations"] = rec_export.apply(
        lambda row: (
            ("Improve attendance. " if pd.notna(row["Attendance"]) and row["Attendance"] < 75 else "") +
            ("Build a regular study routine. " if pd.notna(row["Study_Hours"]) and row["Study_Hours"] < 2 else "") +
            ("Review assignment feedback. " if pd.notna(row["Assignment_Score"]) and row["Assignment_Score"] < 60 else "") +
            ("Revise internal-assessment topics. " if pd.notna(row["Internal_Marks"]) and row["Internal_Marks"] < 60 else "") +
            ("Create a weekly improvement plan. " if pd.notna(row["Final_Marks"]) and row["Final_Marks"] < 60 else "")
        ).strip() or "Maintain current habits and monitor progress.", axis=1
    )
    st.download_button("Download recommendations (CSV)", rec_export.to_csv(index=False).encode("utf-8"),
                       "student_recommendations.csv", "text/csv")

st.divider()
st.caption("Performance categories: High ≥ 80, Medium ≥ 60 and < 80, Low < 60. This academic demo is not intended for high-stakes decisions.")
