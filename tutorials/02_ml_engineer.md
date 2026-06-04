# Tutorial 02 — ML Engineer

**Your job:** produce one calibrated probability of default per borrower
(`model_pd ∈ [0,1]`). Everything in the dependence and metric layers is only as
good as this number.

**Your modules:** `src/pd_model.py`, `src/structural_pd.py`
(import via `from src.ml import ...`).

> If the bank already has a PD model, the Data Engineer supplies `model_pd`
> directly and you can skip training — your job becomes monitoring/validation.

---

## Step 1 — train the individual PD model

```python
from src.ml import IndividualPDModel

# feature_columns is set on the CONSTRUCTOR (defaults to a sensible credit set):
model = IndividualPDModel(
    model_type="gradient_boosting",                 # or "logistic"
    feature_columns=["age", "income", "employment_years", "debt_to_income",
                     "num_credit_lines", "missed_payments",
                     "credit_utilization", "account_age_months"],
)
metrics = model.fit(persons, target_col="default", validation_split=0.2)
print("Validation AUC:", metrics["val_auc"])      # keys: train_auc, val_auc, ...
persons["model_pd"] = model.predict_proba(persons)   # the contract you provide
```

## Step 2 — inspect what drives it

```python
print(model.feature_importance_.sort_values(ascending=False).head(10))
# explain one borrower
print(model.explain_prediction(persons.iloc[0]))
```

## Step 3 — pick a decision threshold (if you classify)

```python
thr = model.get_optimal_threshold(persons, target_col="default", metric="f1")
flags = model.predict(persons, threshold=thr)
```

## Step 4 — add the Merton structural second signal (optional, recommended)

```python
from src.ml import StructuralPDModel
struct = StructuralPDModel()
persons = struct.fit_transform(persons, statistical_pd_col="model_pd")
# adds merton_pd, blended_pd, distance_to_default, pd_signal_divergence
# Large pd_signal_divergence = market-implied deterioration not yet in the
# statistical PD → an early-warning flag for Risk.
```

---

## Quality bar (what "good" means here)
- **Discrimination:** AUC / KS / Gini on a held-out, out-of-time sample.
- **Calibration:** the predicted PD must equal the observed default rate by
  decile (reliability diagram, Brier score). A miscalibrated PD poisons every
  downstream EL/VaR/metric. *(Roadmap §8 adds a calibration/backtest module;
  until then, validate externally.)*
- **Monotonicity:** more `missed_payments` ⇒ higher PD, etc.

## Roadmap (where this is going)
`ROADMAP.md §2` swaps the backend to XGBoost/CatBoost with GPU and monotonic
constraints, keeping the same `.fit/.predict_proba/.feature_importance_`
interface — so nothing downstream changes when you upgrade.

## Checklist
- [ ] `persons["model_pd"]` ∈ [0,1], one row per borrower, no NaN
- [ ] AUC reported on out-of-time data (not just train)
- [ ] calibration checked (predicted ≈ observed by decile)
- [ ] feature importances reviewed for sanity
