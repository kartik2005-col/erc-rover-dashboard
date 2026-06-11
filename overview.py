import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_loader import load_data


def show():
    st.title("🤖 ERC Rover Intelligence Dashboard")
    st.markdown("**European Rover Challenge — Historical Selection Analysis**")
    st.markdown("---")

    df = load_data()

    # top-level stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Teams (Historical)", len(df))
    col2.metric("Selected Teams", df["selected"].sum())
    col3.metric("Selection Rate", f"{df['selected'].mean()*100:.1f}%")
    col4.metric("Avg Terrain Score", f"{df['terrain_score'].mean():.1f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Selection Rate by Year")
        year_sel = df.groupby("year")["selected"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(year_sel.index, year_sel.values, color="#4C72B0")
        ax.set_xlabel("Year")
        ax.set_ylabel("Selection Rate (%)")
        ax.set_ylim(0, 100)
        for i, v in zip(year_sel.index, year_sel.values):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Mobility Type Distribution")
        mob_counts = df["mobility_type"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.barh(mob_counts.index, mob_counts.values, color="#55A868")
        ax2.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)
