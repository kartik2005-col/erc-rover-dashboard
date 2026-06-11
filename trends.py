import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import load_data


def show():
    st.title(" Score Trends")
    st.markdown("How ERC performance metrics have shifted across competition years.")
    st.markdown("---")

    df = load_data()

    # avg scores per year
    yearly = df.groupby("year")[["terrain_score", "science_score", "presentation_score", "sensor_payload_score"]].mean()

    st.subheader("Average Scores Over Years")
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, color in zip(yearly.columns, ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
        ax.plot(yearly.index, yearly[col], marker="o", label=col.replace("_", " ").title(), color=color)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # weight trends
    st.subheader("Average Rover Weight by Year")
    weight_yearly = df.groupby("year")["weight_kg"].mean()
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(weight_yearly.index, weight_yearly.values, marker="s", color="#DD8452", linewidth=2)
    ax2.fill_between(weight_yearly.index, weight_yearly.values, alpha=0.15, color="#DD8452")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Avg Weight (kg)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # category-wise performance breakdown
    st.subheader("Category-wise Score Distribution (Boxplot)")
    score_cols = ["terrain_score", "science_score", "presentation_score"]

    fig3, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, col in enumerate(score_cols):
        data_by_year = [df[df["year"] == y][col].values for y in sorted(df["year"].unique())]
        axes[i].boxplot(data_by_year, labels=sorted(df["year"].unique()))
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Year")
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")

    # selection rate trend
    st.subheader("Selection Rate Trend")
    sel_rate = df.groupby("year")["selected"].mean() * 100
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.bar(sel_rate.index, sel_rate.values, color="#4C72B0", alpha=0.8)
    ax4.plot(sel_rate.index, sel_rate.values, "o--", color="#C44E52", linewidth=1.5)
    ax4.set_ylabel("Selection Rate (%)")
    ax4.set_xlabel("Year")
    ax4.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()
