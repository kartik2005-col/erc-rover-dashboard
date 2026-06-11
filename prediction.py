import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import load_data
from models.selection_model import get_trained_model, reset_model


def show():
    st.title(" AI Selection Predictor")
    st.markdown("Enter your rover's design parameters and get an estimated selection probability.")
    st.markdown("---")

    df = load_data()
    model = get_trained_model(df)

    st.subheader("Rover Configuration")

    col1, col2 = st.columns(2)

    with col1:
        weight = st.slider("Weight (kg)", min_value=10.0, max_value=80.0, value=35.0, step=0.5)
        sensor_score = st.slider("Sensor Payload Score (1-10)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
        autonomy = st.slider("Autonomy Score (1-10)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)

    with col2:
        terrain = st.slider("Terrain Score (0-100)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        science = st.slider("Science Task Score (0-100)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        presentation = st.slider("Presentation Score (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

    col3, col4 = st.columns(2)
    with col3:
        mobility = st.selectbox(
            "Mobility Type",
            ["6-wheel rocker bogie", "4-wheel differential", "tracked", "8-wheel", "custom"]
        )
    with col4:
        power = st.selectbox(
            "Power System",
            ["LiPo battery", "Li-ion battery", "hybrid solar-battery", "fuel cell"]
        )

    st.markdown("---")

    if st.button(" Predict Selection Probability", type="primary"):
        input_data = {
            "weight_kg": weight,
            "sensor_payload_score": sensor_score,
            "autonomy_score": autonomy,
            "terrain_score": terrain,
            "science_score": science,
            "presentation_score": presentation,
            "mobility_type": mobility,
            "power_system": power
        }

        try:
            prob = model.predict_proba(input_data)

            st.markdown("### Prediction Result")

            # color based on probability
            if prob >= 0.7:
                color = "green"
                verdict = "High chance of selection "
            elif prob >= 0.45:
                color = "orange"
                verdict = "Borderline — could go either way "
            else:
                color = "red"
                verdict = "Low selection probability "

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric("Selection Probability", f"{prob*100:.1f}%")
                st.markdown(f"**{verdict}**")

            with col_res2:
                # probability gauge using a simple bar
                fig, ax = plt.subplots(figsize=(5, 1.2))
                ax.barh([""], [prob], color=color, height=0.4)
                ax.barh([""], [1 - prob], left=[prob], color="#e0e0e0", height=0.4)
                ax.set_xlim(0, 1)
                ax.axvline(0.55, color="black", linestyle="--", linewidth=1, label="Selection threshold")
                ax.set_xlabel("Probability")
                ax.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # feature importance breakdown
            st.markdown("---")
            st.subheader("What's driving this prediction?")
            coefs = model.get_coefficients()
            if coefs is not None:
                # show top 6
                top_feats = coefs.head(6)
                fig2, ax2 = plt.subplots(figsize=(7, 3))
                colors_coef = ["#5cb85c" if v > 0 else "#d9534f" for v in top_feats["coefficient"]]
                ax2.barh(top_feats["feature"], top_feats["coefficient"], color=colors_coef)
                ax2.axvline(0, color="black", linewidth=0.8)
                ax2.set_xlabel("Coefficient (positive = helps selection)")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                st.caption("Coefficients from logistic regression — shows direction and magnitude of each feature's effect.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Try resetting the model below.")
            if st.button("Reset and retrain model"):
                reset_model()
                st.rerun()
