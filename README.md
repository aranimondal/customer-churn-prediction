# Customer Churn Prediction — Telco

End-to-end churn solution for a telecommunications retention team: data preparation, EDA, feature engineering, a Decision Tree classifier, a persisted scikit-learn pipeline, and a FastAPI scoring service.

Dataset: IBM Telco Customer Churn (7,043 customers, 21 columns). Target: `Churn` (Yes/No).

## Important URLs

| Resource | URL | Purpose |
|---|---|---|
| GitHub Repository | https://github.com/aranimondal/customer-churn-prediction | Complete project source code and documentation |
| GitHub README | https://github.com/aranimondal/customer-churn-prediction/blob/main/README.md | Project overview, setup and usage guide |
| Swagger UI | http://127.0.0.1:8000/docs | Interactive API UI — test `/health` and `/predict` in the browser |
| ReDoc | http://127.0.0.1:8000/redoc | Alternative interactive API documentation |
| OpenAPI JSON | http://127.0.0.1:8000/openapi.json | Machine-readable API specification |
| Health API | http://127.0.0.1:8000/health | Check whether the API and model artifact are available |
| Prediction API | http://127.0.0.1:8000/predict | POST customer data and receive churn prediction + probability |
| Sample Request | https://github.com/aranimondal/customer-churn-prediction/blob/main/sample_request.json | Ready-to-use example payload for `/predict` |
| Analysis Notebook | https://github.com/aranimondal/customer-churn-prediction/blob/main/notebook/churn_analysis.ipynb | EDA, feature engineering, model training and evaluation |
| Trained Model | https://github.com/aranimondal/customer-churn-prediction/blob/main/model/churn_model.pkl | Persisted scikit-learn model pipeline used by the API |
| Test Suite | https://github.com/aranimondal/customer-churn-prediction/blob/main/tests/test_pipeline.py | Automated data, pipeline and API tests |
| GitHub Actions | https://github.com/aranimondal/customer-churn-prediction/actions | CI test execution and build status |
| Local Development & Evaluation Guide | https://github.com/aranimondal/customer-churn-prediction/blob/main/DEVELOPMENT.md | Step-by-step local setup, testing, model training and development-mode API startup |

> **Local application URLs:** Start the API with `uvicorn app:app --reload` before opening the Swagger UI, ReDoc, OpenAPI, Health API, or Prediction API URLs above. These `127.0.0.1` URLs are available only on the machine where the application is running.


## Why the design looks like this

The one decision that shapes everything else is that **all preprocessing lives inside a single scikit-learn `Pipeline`**, and both the notebook and the API import the same modules from `src/`. There is no second copy of the feature logic to drift out of sync, and the artifact loaded by the API is byte-identical to the one fitted in training. Two consequences worth noting:

- Feature engineering is implemented as a **stateless transformer** (`src/features.py`). Because it learns nothing from the data, there is no path for train/test leakage through it.
- Anything that *does* learn from data — imputer statistics, one-hot vocabularies, the tree — is fitted on the training split only, then persisted as one object.

There is deliberately no scaler: a Decision Tree splits on thresholds and is invariant to monotone rescaling, so standardising would add a step that buys nothing.

## Project layout

```text
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb      # full analysis narrative
├── model/
│   ├── churn_model.pkl           # created by `python -m src.train`
│   └── metrics.json              # comparison table + test metrics
├── src/
│   ├── config.py                 # schema + paths, single source of truth
│   ├── data.py                   # loading and row-local cleaning
│   ├── features.py               # ChurnFeatureBuilder transformer
│   ├── pipeline.py               # features -> encode -> DecisionTree
│   ├── train.py                  # config comparison, evaluation, persistence
│   ├── persistence.py            # save/load the artifact
│   └── schemas.py                # pydantic request/response models
├── tests/
│   └── test_pipeline.py
├── app.py                        # FastAPI service
├── requirements.txt
├── sample_request.json
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

From the project root:

```bash
python -m src.train
```

This prints the cross-validated comparison of all candidate configurations, the held-out test metrics, the confusion matrix and the top feature importances, then writes `model/churn_model.pkl` and `model/metrics.json`. Run it before starting the API — the artifact is what the service loads.

## Run the API

```bash
uvicorn app:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Request body: see [`sample_request.json`](sample_request.json) — all 19 predictive fields are required. `customerID` is an identifier, not a feature; sending it is ignored.

Response shape:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

The values above illustrate the format. The actual numbers for the sample payload come from your trained artifact — run `python -m src.train` and then the curl above, or the last cell of the notebook, which prints the same response.

### `GET /health`

Reports whether the model artifact loaded. Useful as a container readiness probe; the service starts even when the artifact is missing so this endpoint can report the reason instead of the process crash-looping.

### Error handling

| Situation | Status | Behaviour |
|---|---|---|
| Missing field, wrong type, out-of-range number, or a category not in the data dictionary | `422` | Field-level `detail` listing each offending field |
| Malformed JSON | `422` | Parse error detail |
| Model artifact not on disk | `503` | Message telling the caller to run training |
| Unexpected inference failure | `500` | Generic message; full traceback logged server-side, never returned |

Allowed categorical values are transcribed from the supplied data dictionary into `src/config.py`, so validation cannot drift from the schema the model was trained on.

## Data preparation decisions

**`TotalCharges`.** Ships as `object` because 11 rows contain a blank string. Every one of those rows has `tenure = 0` — customers who signed up but have not completed a billing cycle — so their lifetime spend is genuinely `0`, not unknown, and they are filled with `0.0`. Median imputation would be actively wrong here: it would credit a brand-new customer with the accumulated spend of an average long-tenured one, contradicting `tenure = 0` and corrupting the engineered `monthly_charge_delta`. `src/data.py` asserts the `tenure == 0` relationship, so if future data violates it the pipeline fails loudly rather than filling wrong values silently.

**`customerID`.** Dropped from the feature set. It is a surrogate key with no generalisable signal, and a tree would happily memorise it.

**Duplicates.** Checked on both full rows and `customerID`; the source file is clean. The `drop_duplicates` call is a safety net for re-runs on appended data.

**Split.** 70:30, `random_state=42`, **stratified on the target**. Stratification is a deliberate choice given the ~26.5% positive rate: an unstratified 30% split can drift a couple of points off the population churn rate, and since recall is the selection metric that drift would appear directly as noise in the number being optimised.

## Engineered features

Three domain-motivated features are created by the stateless transformer and carried through the reusable train/API pipeline.

| Feature | Construction | Business rationale |
|---|---|---|
| `monthly_charge_delta` | `MonthlyCharges − TotalCharges/tenure`, defined as 0 when `tenure = 0` | Highlights a customer's current bill relative to their historical average monthly spend — a simple price-change signal. |
| `num_addon_services` | Count of the six optional services whose value is `Yes` | Captures breadth of the customer's service bundle. It provides a compact view of engagement while retaining the six raw service flags for the model. |
| `tenure_bucket` | Fixed lifecycle bands: `0-12`, `13-24`, `25-48`, `49+` months | Gives the model a business-readable lifecycle view alongside the raw continuous tenure value. |

All three are row-local calculations: they use only information belonging to the customer being scored, so they do not learn statistics from the target or from other rows. The raw fields remain available as well; the Decision Tree can choose the representation that gives the strongest split.

The notebook also checks these engineered features directly and reports their relationship with churn. Feature importance is used later to show which representations the fitted tree actually relies on. A feature having low importance is treated as a modelling result, not as a reason to remove a requirement from the assignment.

## Model selection

Four configurations differing in kind, not degree, compared by 5-fold stratified cross-validation **inside the training split**; the test set is touched once, at the end.

| Configuration | Purpose |
|---|---|
| `baseline_unpruned` | Overfitting reference point |
| `pruned_gini` | Depth and leaf-size constraints |
| `pruned_entropy` | Same capacity, different split criterion |
| `balanced_pruned` | Pruned plus `class_weight="balanced"` for the 3:1 imbalance |

Selection metric is **cross-validated recall**, fixed in `src/train.py` before the comparison runs so the criterion is not retrofitted to the winner. The comparison table, the chosen configuration and the resulting test metrics are written to `model/metrics.json` on every training run — that file, not this README, is the record of results.

## Evaluation and the precision/recall question

### Results on the held-out test set

Selected configuration: **`balanced_pruned`** — a pruned tree (`criterion=gini`, `max_depth=5`, `min_samples_split=40`, `min_samples_leaf=20`) with `class_weight="balanced"`. It was chosen by cross-validated recall on the training split, where it scored **0.777** against 0.528 / 0.536 / 0.513 for `pruned_gini`, `pruned_entropy` and `baseline_unpruned`.

All figures below are read from `model/metrics.json`, produced by `python -m src.train` on a stratified 30% hold-out (2,113 customers, 26.5% churn rate).

| Metric | Test value |
|---|---|
| Accuracy | **0.7080** (70.80%) |
| Precision | **0.4713** (47.13%) |
| Recall | **0.8182** (81.82%) |
| F1 | **0.5980** |
| ROC-AUC | 0.8301 |

Confusion matrix: 459 true positives, 102 false negatives, 515 false positives, 1037 true negatives.

Reading these numbers honestly: recall of **81.8%** means the model catches roughly 8 in 10 of the customers who actually leave — that is the number the design optimises for, and it is the reason `balanced_pruned` was selected over the higher-accuracy `pruned_gini`. Accuracy of **70.8%** is *below* the ~73% no-information baseline, and that is a deliberate trade rather than a defect: the class weighting pushes the model to flag aggressively, converting accuracy into recall. Precision of **47.1%** is the cost of that trade — a little under half the flagged customers would not have left, so of the 974 customers flagged, 515 are contacted unnecessarily while only 102 real churners are missed.

Whether that trade is right is a business question, not a statistical one, and it is exactly the argument below.

The model is scored on accuracy, precision, recall, F1, ROC-AUC and a confusion matrix. Read accuracy against the **~73% no-information baseline**, not against 100%: predicting "No" for everyone already scores about 73% while catching zero churners, which is why accuracy is the wrong headline metric for this problem.

**Recall is the priority, because the two errors have asymmetric cost.** A false positive costs one retention contact — some agent time, possibly a discount. A false negative costs a departed subscriber: their remaining lifetime revenue plus the acquisition spend to replace them, which in telecom typically runs to several months of revenue. When one error is roughly an order of magnitude more expensive than the other, accepting many false positives to avoid one false negative is the correct trade.

The asymmetry is also about *timing*, which is easy to overlook. A false positive is recoverable — stop calling that segment next month. A false negative is not: once a customer has ported their number, no model improvement brings them back.

Precision still matters, as the **operational constraint**. Recall alone is trivially maximised by flagging everyone, which is useless: the retention team has finite capacity, so a low-precision list means agents spend the month on customers who were never going to leave while genuinely at-risk customers go uncalled.

The practical resolution is to stop treating this as a binary at 0.5. The API returns `churn_probability`, so the team can rank customers by score and work down the list as far as capacity allows. That turns the precision/recall choice into a capacity decision, which is where it belongs. The notebook includes a threshold-sensitivity table stating, for each cut-off, how many customers must be contacted, how many real churners that catches, and how many are still missed.

Note that the cost magnitudes above are reasoning about relative orders of magnitude. The dataset contains no cost or CLV columns, so they are not derived from it and should be replaced with the company's real figures before a threshold is set in production.

## Interpretation

`python -m src.train` prints ranked Gini importances; the notebook adds a top-15 chart, a depth-limited tree plot and the `export_text` rule dump, which reads as a plain-language targeting rule the retention team can act on without reference to the model.

Two caveats stated explicitly, because they are easy to overstate: Gini importance is **not causal** — "month-to-month is important" does not license "moving customers to annual contracts will cut churn by X", which needs an experiment — and importance is unstable under correlated features, so with `tenure` and `TotalCharges` overlapping heavily the tree picks one and the other looks less important than it is.

## Tests

```bash
pytest -q
```

19 tests covering the documented `TotalCharges` defect, identifier and target exclusion, statelessness of the feature transformer, the `tenure = 0` division guard, artifact round-trip equality, a regression guard that the two removed features stay removed, and API success and validation-failure paths. The API tests skip cleanly if the model artifact has not been trained yet.

## Reproducibility

`random_state=42` throughout (split, CV folds, tree). All paths derive from `src/config.py` relative to the project root, so there are no machine-specific absolute paths.


## Bonus: independent model comparison

The required Decision Tree remains the deployed model. As an additional model-family check, `src/bonus_models.py` trains a leakage-safe Logistic Regression pipeline on the same raw input contract and evaluates it with the same 5-fold stratified cross-validation and held-out test metrics (accuracy, precision, recall, F1 and ROC-AUC).

The comparison is intentionally kept separate from model selection: adding another algorithm does not silently change the assignment-required deployed model. This makes the bonus experiment reproducible while preserving the original Decision Tree decision.

## Operational threshold analysis

The API returns a churn probability rather than only a class label. `src/decision_policy.py` provides a small, model-agnostic threshold-sensitivity utility that converts those probabilities into an operational table containing contact volume, contact rate, true positives, false positives, false negatives, precision and recall.

No threshold is selected automatically. The appropriate cut-off is a business capacity decision: a retention team with more contact capacity can work further down the ranked probability list, while a constrained team can use a higher cut-off. This keeps statistical evaluation separate from an unsupported assumption about campaign cost.

## Automated quality gate

A GitHub Actions workflow runs the full `pytest` suite on pushes to `main` and on pull requests. The suite now covers the required data/pipeline/API contracts plus the bonus Logistic Regression path and threshold-analysis validation. This is an execution safeguard only; it does not alter training behaviour or the persisted model.
