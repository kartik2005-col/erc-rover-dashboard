import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_loader import load_data


def show():
    st.title(" Historical Analysis")
    st.markdown("Explore how rover specs relate to selection outcomes across years.")
    st.markdown("---")

    df = load_data()

    # filters
    col1, col2 = st.columns(2)
    with col1:
        years = sorted(df["year"].unique())
        selected_years = st.multiselect("Filter by Year", years, default=years)
    with col2:
        mob_types = df["mobility_type"].unique().tolist()
        selected_mob = st.multiselect("Mobility Type", mob_types, default=mob_types)

    filtered = df[df["year"].isin(selected_years) & df["mobility_type"].isin(selected_mob)]

    if filtered.empty:
        st.warning("No data with current filters.")
        return

    st.markdown(f"Showing **{len(filtered)}** teams after filtering.")
    # note: 2021 data looks a bit off — ERC was remote that year and scoring criteria shifted
    # treat those numbers with some scepticism
    st.markdown("---")

    # score distributions
    st.subheader("Score Distributions: Selected vs Not Selected")

    score_cols = ["terrain_score", "science_score", "presentation_score"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i, col in enumerate(score_cols):
        for label, color in [(0, "#d9534f"), (1, "#5cb85c")]:
            subset = filtered[filtered["selected"] == label][col]
            axes[i].hist(subset, bins=15, alpha=0.6, label=("Not Selected" if label == 0 else "Selected"), color=color)
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Score")
        axes[i].legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # weight vs terrain scatter
    st.subheader("Weight vs Terrain Score")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = filtered["selected"].map({0: "#d9534f", 1: "#5cb85c"})
    ax2.scatter(filtered["weight_kg"], filtered["terrain_score"],
                c=colors, alpha=0.5, edgecolors="none", s=40)
    ax2.set_xlabel("Weight (kg)")
    ax2.set_ylabel("Terrain Score")
    # manual legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#d9534f", label="Not Selected"),
                  Patch(facecolor="#5cb85c", label="Selected")]
    ax2.legend(handles=legend_els)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # correlation heatmap
    st.subheader("Feature Correlation Matrix")
    num_cols = ["weight_kg", "sensor_payload_score", "autonomy_score",
                "terrain_score", "science_score", "presentation_score", "selected"]
    corr = filtered[num_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3, linewidths=0.5)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
