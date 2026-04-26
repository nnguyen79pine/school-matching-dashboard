import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="School Matching Dashboard", layout="wide")

st.title("EducatEd Choice: School Matching Dashboard")

st.write("""
This dashboard visualizes schools based on academic performance and stability
to support best-fit school matching.
""")

# Load dataset (make sure this file is in your repo)
df = pd.read_csv("master_school_table_v5_2023_24.csv")

# Features for PCA
features = [
    "grad_rate",
    "cohort_size",
    "sat_total",
    "mobility_rate",
    "mobility_count",
    "discipline_percent",
    "hope_eligible_percent"
]

# Clean data
df_pca = df[features].copy()
df_pca = df_pca.fillna(df_pca.median())

# Standardize
X_scaled = StandardScaler().fit_transform(df_pca)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create PCA DataFrame
pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["school_name"] = df["school_name"]
pca_df["grad_rate"] = df["grad_rate"]
pca_df["sat_total"] = df["sat_total"]
pca_df["hope_eligible_percent"] = df["hope_eligible_percent"]
pca_df["mobility_rate"] = df["mobility_rate"]

# Assign quadrant profile
def assign_profile(row):
    if row["PC1"] >= 0 and row["PC2"] >= 0:
        return "High Performance + Stable"
    elif row["PC1"] >= 0 and row["PC2"] < 0:
        return "High Performance + Less Stable"
    elif row["PC1"] < 0 and row["PC2"] >= 0:
        return "Lower Performance + Stable"
    else:
        return "Lower Performance + Less Stable"

pca_df["school_profile"] = pca_df.apply(assign_profile, axis=1)

# Create plot
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
    title="School Matching Map: Performance vs Stability",
    labels={
        "PC1": "Academic Performance",
        "PC2": "Stability / Structure",
        "grad_rate": "Graduation Rate",
        "school_profile": "Profile",
        "sat_total": "SAT Total",
        "hope_eligible_percent": "HOPE Eligible (%)",
        "mobility_rate": "Mobility Rate (%)"
    },
    color_continuous_scale="Viridis",
    opacity=0.8
)

# Add quadrant lines
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.add_vline(x=0, line_dash="dash", line_color="gray")

# Show chart
st.plotly_chart(fig, use_container_width=True)

# Insights
st.subheader("Key Insights")
st.write("""
- PC1 represents academic performance (higher = stronger performance)
- PC2 represents school stability (higher = more stable)
- Schools in the top-right quadrant are the best matches
""")
