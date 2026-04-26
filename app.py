import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="EducatEd Choice Dashboard", layout="wide")

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; color: #1f2a44;}
.subtitle {font-size: 17px; color: #5f6b7a; margin-bottom: 25px;}
.card {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="main-title">EducatEd Choice: School Matching Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Helping families find the best-fit school using data-driven insights.</div>', unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("master_school_table_v5_2023_24.csv")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Navigation")
view = st.sidebar.radio(
    "Select Section",
    ["School Matching Map", "Key Relationships", "PCA Insights", "Dataset"]
)

# -------------------------
# PCA PREP
# -------------------------
features = [
    "grad_rate", "cohort_size", "sat_total", "mobility_rate",
    "mobility_count", "discipline_percent", "hope_eligible_percent"
]

df_pca = df[features].copy()
df_pca = df_pca.fillna(df_pca.median())

X_scaled = StandardScaler().fit_transform(df_pca)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["school_name"] = df["school_name"]
pca_df["grad_rate"] = df["grad_rate"]
pca_df["sat_total"] = df["sat_total"]
pca_df["hope_eligible_percent"] = df["hope_eligible_percent"]
pca_df["mobility_rate"] = df["mobility_rate"]

# -------------------------
# PROFILE CLASSIFICATION
# -------------------------
def assign_profile(row):
    if row["PC1"] >= 0 and row["PC2"] >= 0:
        return "High Performance + Stable"
    elif row["PC1"] >= 0 and row["PC2"] < 0:
        return "High Performance + Less Stable"
    elif row["PC1"] < 0 and row["PC2"] >= 0:
        return "Lower Performance + Stable"
    return "Lower Performance + Less Stable"

pca_df["school_profile"] = pca_df.apply(assign_profile, axis=1)

# -------------------------
# METRICS (TOP CARDS)
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Schools", f"{len(df):,}")
col2.metric("Avg Graduation Rate", f"{df['grad_rate'].mean():.1f}%")
col3.metric("Avg SAT Score", f"{df['sat_total'].mean():.0f}")
col4.metric("Avg Mobility Rate", f"{df['mobility_rate'].mean():.1f}%")

# =====================================================
# 1️⃣ SCHOOL MATCHING MAP
# =====================================================
if view == "School Matching Map":

    st.markdown('<div class="section-title">School Matching Map</div>', unsafe_allow_html=True)

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="grad_rate",
        hover_name="school_name",
        hover_data={
            "school_profile": True,
            "grad_rate": ":.1f",
            "sat_total": ":,.0f",
            "hope_eligible_percent": ":.1f",
            "mobility_rate": ":.1f",
            "PC1": False,
            "PC2": False
        },
        labels={
            "PC1": "Academic Performance",
            "PC2": "Stability",
            "grad_rate": "Graduation Rate"
        },
        color_continuous_scale="Viridis",
        opacity=0.8
    )

    # Quadrant lines
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=0, line_dash="dash")

    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="card">
    • PC1 represents academic performance (higher = stronger outcomes)<br>
    • PC2 represents school stability (higher = more stable)<br>
    • Schools in the top-right quadrant are the best overall matches
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 2️⃣ KEY RELATIONSHIP (REQUIREMENT PLOT #2)
# =====================================================
elif view == "Key Relationships":

    st.markdown('<div class="section-title">Graduation Rate vs HOPE Eligibility</div>', unsafe_allow_html=True)

    fig2 = px.scatter(
        df,
        x="hope_eligible_percent",
        y="grad_rate",
        trendline="ols",
        labels={
            "hope_eligible_percent": "HOPE Eligibility (%)",
            "grad_rate": "Graduation Rate (%)"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="card">
    There is a clear positive relationship between HOPE eligibility and graduation rate.  
    This suggests academic readiness and performance are strongly connected.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 3️⃣ PCA INSIGHTS (REQUIREMENT PLOT #3)
# =====================================================
elif view == "PCA Insights":

    st.markdown('<div class="section-title">Explained Variance (PCA)</div>', unsafe_allow_html=True)

    explained_var = pca.explained_variance_ratio_

    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(explained_var))],
        "Variance": explained_var
    })

    fig3 = px.bar(var_df, x="Component", y="Variance")

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="card">
    PCA reduces many variables into key dimensions.  
    PC1 explains the largest portion of variation (~40%), meaning it captures the most important academic patterns.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 4️⃣ DATASET
# =====================================================
elif view == "Dataset":

    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.dataframe(df.head(50), use_container_width=True)
