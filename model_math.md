# how the prediction model works

This is a writeup of the actual maths behind the selection probability model. Not a textbook explanation — just enough to understand what's happening under the hood and why I made the choices I did.

---

## the problem

Given a rover's design parameters, predict: will it get selected at ERC?

This is binary classification. Output is either 1 (selected) or 0 (not selected). But rather than just outputting a hard label, I wanted a probability — so teams can see *how close* they are to the cutoff, not just a yes/no.

---

## input features

8 features total — 6 numerical, 2 categorical.

| feature | type | range |
|---|---|---|
| weight_kg | numerical | ~15–75 kg |
| sensor_payload_score | numerical | 1–10 |
| autonomy_score | numerical | 1–10 |
| terrain_score | numerical | 0–100 |
| science_score | numerical | 0–100 |
| presentation_score | numerical | 0–100 |
| mobility_type | categorical | 5 options |
| power_system | categorical | 4 options |

---

## preprocessing

### standardisation

The numerical features are on totally different scales. Weight is in kg (15–75), scores are 0–100, sensor/autonomy are 1–10. If you feed these raw into logistic regression, features with larger ranges dominate the gradient updates and the model learns weird weights.

Fix: z-score normalisation via `StandardScaler`.

```
z = (x - mean) / std
```

After this, every feature has mean = 0 and std = 1. The model treats them on equal footing.

**Important:** the mean and std are computed on the *training set only*. When predicting on new inputs, I apply the same training-set statistics. If you refit the scaler on each prediction, you'd get different scaled values every time — the model would break.

### label encoding (categorical features)

Logistic regression needs numbers, not strings. `LabelEncoder` maps each category to an integer:

```
mobility_type:
  "4-wheel differential" → 0
  "6-wheel rocker bogie" → 1
  "8-wheel"              → 2
  "custom"               → 3
  "tracked"              → 4
```

The honest limitation here: integer encoding implies an ordering (like 4 > 3 > 2 > 1), which doesn't actually exist for mobility types. One-hot encoding would be cleaner mathematically. I kept label encoding because the dataset is small (n=200) and adding 5+ binary columns for each categorical feature risks overfitting. With more data, switch to one-hot.

---

## logistic regression

### why not something fancier

I tried to keep this model decision honest. With ~200 training examples:

- Random Forest / XGBoost would overfit badly
- Neural nets are completely overkill and uninterpretable
- Logistic regression gives you interpretable coefficients — you can directly see which feature is hurting vs helping selection probability

The interpretability is actually the point. An engineer should be able to look at the coefficients and go "oh, terrain score matters more than presentation score" and make a design decision.

### the sigmoid function

The core of logistic regression is squashing a linear combination of inputs into a probability between 0 and 1.

```
P(selected | x) = σ(z) = 1 / (1 + e^(-z))
```

where z is the linear score:

```
z = w₀ + w₁x₁ + w₂x₂ + ... + w₈x₈
```

w₀ is the bias. w₁...w₈ are learned weights for each feature.

The sigmoid does the right thing intuitively:
- z = 0 → P = 0.5 (uncertain)
- z >> 0 → P → 1 (confident: selected)
- z << 0 → P → 0 (confident: not selected)

### expanded z for this model

After preprocessing, z looks like:

```
z = w₀
  + w₁ · (weight_kg - μ_w) / σ_w
  + w₂ · (sensor_score - μ_s) / σ_s
  + w₃ · (autonomy_score - μ_a) / σ_a
  + w₄ · (terrain_score - μ_t) / σ_t
  + w₅ · (science_score - μ_sc) / σ_sc
  + w₆ · (presentation_score - μ_p) / σ_p
  + w₇ · mobility_encoded
  + w₈ · power_encoded
```

All the μ and σ values come from the training set StandardScaler.

---

## training

### loss function

The model is trained by minimising binary cross-entropy over all training examples:

```
L = -(1/m) · Σ [ yᵢ · log(pᵢ) + (1 - yᵢ) · log(1 - pᵢ) ]
```

m = number of training examples, yᵢ = true label, pᵢ = predicted probability.

What this penalises: confident wrong answers. If the model outputs P=0.99 and the true label is 0, the loss term for that example is `-log(1 - 0.99) = -log(0.01) ≈ 4.6`. Very high. If it outputs P=0.6 for a true label of 1, the loss is `-log(0.6) ≈ 0.51`. Much lower.

### optimisation

scikit-learn uses L-BFGS-B by default (with `max_iter=1000`). It's a quasi-Newton method — basically gradient descent but smarter, it uses an approximation of the second derivative (Hessian) to take better steps.

The gradient of the loss w.r.t. each weight is:

```
∂L/∂wⱼ = (1/m) · Σ (pᵢ - yᵢ) · xᵢⱼ
```

Training stops when this gradient is close enough to zero — meaning the weights are at a local minimum of the log loss.

### train/test split

80/20 split with `stratify=y`. Stratification matters because if the selection rate is ~40%, a naive random split could give the test set a very different class ratio by chance. Stratifying preserves the ratio in both splits.

---

## decision boundary

The model predicts "selected" when:

```
P(y=1 | x) ≥ threshold
```

Default threshold in the dashboard is **0.55**, not 0.5. Slightly above 0.5 to reduce false positives — I'd rather the model be conservative about calling a rover "selected" than overconfident.

Geometrically the boundary is a hyperplane in 8D feature space:

```
w₀ + w₁x₁ + ... + w₈x₈ = 0
```

Points on one side get P > 0.5, points on the other get P < 0.5.

---

## reading the coefficients

After training, each weight wⱼ tells you the direction and size of that feature's effect.

Formally, the model is modelling the log-odds:

```
log(P / (1-P)) = w₀ + w₁x₁ + ... + w₈x₈
```

A one-unit increase in xⱼ (post-standardisation) shifts the log-odds by wⱼ, which multiplies the odds by `exp(wⱼ)`.

Concrete example: if w_terrain = 0.8, a one standard-deviation increase in terrain score multiplies the odds of selection by `exp(0.8) ≈ 2.23`. So roughly doubling the odds.

Negative coefficients mean the feature hurts selection probability. The weight_kg coefficient should come out negative — heavier rovers generally perform worse on terrain tasks.

The "What's driving this prediction?" chart in the dashboard just plots these coefficients sorted by absolute magnitude.

---

## what-if analysis

The prediction sliders work because inference is just:

```
P = σ(w · x)
```

A dot product + sigmoid. Runs in microseconds. The dashboard recomputes this on every slider change without retraining. That's why it feels instant.

---

## known weaknesses

**label encoding**: as mentioned, implies ordinal relationship for categorical features. Not a huge deal at n=200 but worth fixing if the dataset grows.

**small dataset**: 200 teams is genuinely not a lot. The model's accuracy estimates should be taken with some salt. More data = better model.

**no year effects**: ERC 2021 was fully remote, scoring criteria shifted. The model doesn't know this — it treats 2021 the same as 2023. A `year` feature or year-specific adjustments would help.

**calibration**: logistic regression probabilities are generally well-calibrated, but I haven't formally verified this with a reliability diagram. For high-stakes decisions it'd be worth checking.

**synthetic data**: the current dataset is generated, not real ERC data. Selection probabilities and feature importances will shift once real data is plugged in.
