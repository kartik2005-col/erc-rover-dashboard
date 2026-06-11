import pandas as pd
import numpy as np

# seeding so results are reproducible across runs
np.random.seed(42)


def generate_erc_dataset(n_teams=200):
    """
    Generates synthetic ERC historical data.
    In a real setup you'd scrape this from the ERC website or use actual CSVs.
    
    Features that matter for selection (based on ERC rules):
    - weight_kg: rover weight
    - mobility_type: wheel config
    - power_system: battery/solar/hybrid
    - sensor_payload_score: how many sensors, rated 1-10
    - autonomy_score: how autonomous the rover is (1-10)
    - terrain_score: scored by judges during the terrain task
    - science_score: science task score
    - presentation_score: project presentation rating
    - year: competition year
    """

    years = [2019, 2020, 2021, 2022, 2023, 2024]
    mobility_types = ["6-wheel rocker bogie", "4-wheel differential", "tracked", "8-wheel", "custom"]
    power_systems = ["LiPo battery", "Li-ion battery", "hybrid solar-battery", "fuel cell"]

    rows = []
    for _ in range(n_teams):
        year = np.random.choice(years)
        weight = np.random.uniform(15, 75)
        mobility = np.random.choice(mobility_types)
        power = np.random.choice(power_systems)
        sensor_score = np.random.uniform(3, 10)
        autonomy = np.random.uniform(2, 10)
        terrain = np.random.uniform(20, 100)
        science = np.random.uniform(10, 100)
        presentation = np.random.uniform(30, 100)

        # selection logic — heavier penalised a bit, sensors and autonomy help
        # this is a rough model of real ERC scoring
        base_prob = (
            0.35 * (terrain / 100)
            + 0.25 * (science / 100)
            + 0.15 * (sensor_score / 10)
            + 0.15 * (autonomy / 10)
            + 0.10 * (presentation / 100)
            - 0.002 * max(0, weight - 40)  # penalty for heavy rovers
        )

        # mobility bonus
        if mobility == "6-wheel rocker bogie":
            base_prob += 0.05
        elif mobility == "tracked":
            base_prob += 0.03

        # add noise so it's not deterministic
        noise = np.random.normal(0, 0.06)
        final_prob = np.clip(base_prob + noise, 0, 1)
        selected = int(final_prob > 0.55)

        rows.append({
            "year": year,
            "weight_kg": round(weight, 2),
            "mobility_type": mobility,
            "power_system": power,
            "sensor_payload_score": round(sensor_score, 2),
            "autonomy_score": round(autonomy, 2),
            "terrain_score": round(terrain, 2),
            "science_score": round(science, 2),
            "presentation_score": round(presentation, 2),
            "selected": selected
        })

    df = pd.DataFrame(rows)
    return df


def load_data():
    """Main entry point for loading data throughout the app."""
    # TODO: once we get the real ERC CSVs, replace this with pd.read_csv()
    # and add a cleaning step — the raw exports have some inconsistent column names
    df = generate_erc_dataset(n_teams=200)
    return df


def get_feature_columns():
    """returns just the input features used for prediction"""
    return [
        "weight_kg",
        "sensor_payload_score",
        "autonomy_score",
        "terrain_score",
        "science_score",
        "presentation_score"
    ]


def get_categorical_columns():
    return ["mobility_type", "power_system"]
