# Wildlife Strike Damage Classification

Predict aircraft damage from FAA wildlife-strike incident records using structured features, text features, and a five-model ensemble. The final notebook records **90.32% validation balanced accuracy**, compared with **87.04%** for the initial CatBoost model.

## Results

Validation uses the **2015 incidents**: 12,402 rows, with pre-2015 incidents forming the training pool. These figures are recorded in the notebook's written results summary; they have not been independently reproduced in this documentation update.

| Metric | Recorded result |
|---|---:|
| Balanced accuracy | **90.32%** |
| Accuracy | Approximately 90.84% |
| ROC-AUC | Approximately 0.9602 |
| Recall on damage class | Approximately 89.8% |
| Specificity on no-damage class | Approximately 90.9% |

| Version | Change | Validation balanced accuracy |
|---|---|---:|
| Initial model | CatBoost with default class weights | 87.04% |
| Class balancing | Balanced CatBoost class weights | 88.79% |
| Boosted blend | Add LightGBM and three-fold bagging | 89.73% |
| Feature and model expansion | Add Naive Bayes, regex keywords, and Apriori features | 89.90% |
| Final thresholds | Use thresholds by flight phase | **90.32%** |

The recorded improvement is **3.28 percentage points** over the initial model. Thresholds and ensemble weights were selected using the 2015 validation labels, so these are tuning-set results, not scores from an untouched final test set or a competition leaderboard.

## Technical approach

- Repair Excel-corrupted count ranges and handle missing numerical and categorical values.
- Engineer TF-IDF text features, regex damage-keyword flags, and Apriori-mined association-rule features.
- Train CatBoost, LightGBM, Multinomial Naive Bayes, Bernoulli Naive Bayes, and logistic regression with three-fold out-of-fold predictions on pre-2015 data.
- Combine base predictions using an L2 logistic stacker and a convex-weight blend search.
- Tune global and flight-phase-specific decision thresholds for balanced accuracy. Small flight-phase groups fall back to the global threshold.

## Contribution

Rishav Gadiyar contributed across preprocessing, feature engineering, model development, tuning, validation, and prediction generation.

## Run the final notebook

```sh
git clone https://github.com/rishgad/ML-Aviation-Prediction-Project.git
cd ML-Aviation-Prediction-Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain the original competition data separately and place `train.csv` and `test.csv` in the repository root. Those files are not included in the repository.

```sh
python scripts/reproduce.py --data-dir . --output-dir results
```

Use Python 3.12+ for the pinned environment. The runner executes the final notebook and saves `metrics.json`, `inputs.json` with dataset and source hashes, and an executed notebook under `results/`. Metrics include the confusion matrix, accuracy, balanced accuracy, ROC-AUC, recall, thresholds, and installed package versions. The notebook writes `submission.csv` in the data directory, replacing any existing prediction file there.

The competition URL and original CSVs are still unavailable; the runner does not substitute another FAA dataset. No newly reproduced scores are claimed. Keep raw data and generated outputs local until competition redistribution rules are known.

## Repository contents

| File | Purpose |
|---|---|
| `damage_prediction.ipynb` | Final five-model ensemble pipeline and recorded results summary |
| `build_notebook.py` | Earlier notebook generator using a different modeling pipeline; do not run it to reproduce the final notebook |
| `requirements.txt` | Dependency versions |
| `submission.csv` | Saved test predictions; labels alone cannot establish prediction accuracy |
| `submission_v2_catboost.csv` | Additional saved prediction file |

## Limitations and reproducibility

The notebook currently has no saved metric outputs, and the execution log contains only a notebook-conversion message. Reproducing the recorded scores requires the original data and a completed run. Random seeds are fixed, but byte-identical results across environments are not guaranteed.

Some text and investigator fields are populated after the incident and can reveal damage information. These results describe incident-record classification and should not be interpreted as demonstrated pre-incident risk prediction. Training-fold out-of-fold predictions support the stacker, but selecting thresholds and ensemble weights on 2015 can still overfit that validation set.
