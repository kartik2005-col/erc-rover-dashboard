# ERC Rover Selection Intelligence Dashboard

A Streamlit dashboard I built to analyse historical European Rover Challenge data and predict whether a rover design is likely to get selected — based on its specs and task scores.

The idea came from watching teams at ERC make hardware decisions without any data backing. This tries to fix that a bit.

---

## what it does

- aggregates historical ERC competition data (rover specs, terrain/science/presentation scores, outcomes)
- lets you filter and explore score distributions, trends across years, mobility type breakdowns
- runs a logistic regression model that estimates selection probability from your rover's parameters
- shows which features are actually driving the prediction (via coefficients), not just a black-box score

---

## running it locally

```bash
git clone https://github.com/your-username/erc-rover-dashboard
cd erc-rover-dashboard
pip install -r requirements.txt
streamlit run app.py
```

should open at `http://localhost:8501`

tested on Python 3.10. if you hit issues with scikit-learn version just pin it to what's in requirements.txt.

---

## project structure

```
├── app.py                     # entry point, sidebar routing
├── requirements.txt
├── data/
│   └── data_loader.py         # dataset generation + loading
├── models/
│   └── selection_model.py     # logistic regression, preprocessing, inference
├── pages/
│   ├── overview.py            # key metrics homepage
│   ├── historical.py          # historical analysis with filters
│   ├── prediction.py          # the actual prediction UI
│   └── trends.py              # year-over-year trend charts
└── docs/
    └── model_math.md          # full writeup of the maths behind the model
```

---

## the model

Logistic regression on 8 features (6 numerical, 2 categorical). Outputs a probability between 0 and 1.

Chose logistic regression over something fancier (RF, XGBoost) because:
- interpretable coefficients — you can see exactly which feature is hurting or helping
- n=200 is small, tree-based models overfit badly here
- inference is instant, which matters for the real-time slider UI

Full derivation of the maths (sigmoid function, loss function, standardisation, how to read the coefficients) is in `docs/model_math.md`.

---

## data note

The dataset right now is synthetically generated to match real ERC data distributions — weights, score ranges, mobility types, selection rates across years. 

If you have access to actual ERC CSV exports, replace `generate_erc_dataset()` in `data/data_loader.py` with `pd.read_csv(your_file)` and make sure the column names match.

---

## known issues / todo

- [ ] label encoding on mobility_type implies an ordinal relationship that doesn't exist — should probably switch to one-hot encoding
- [ ] model retrains on every server restart (it's fast but still). should pickle and cache to disk
- [ ] no handling for NaN values yet — if real data has missing scores it'll break
- [ ] year-level effects not captured (2021 ERC was online, scoring was different)
- [ ] presentation score is probably underweighted in the synthetic data

---

## built with

Python, Pandas, Scikit-learn, Streamlit, Matplotlib, Seaborn
