# Insurance Cross-Sell Prediction

A machine learning project that predicts whether a health insurance customer is likely to be interested in purchasing add-on **vehicle insurance**. The trained model is served behind a FastAPI REST endpoint for real-time predictions.

Built on the [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) dataset from Kaggle.

## Overview

Insurance companies want to know which existing health insurance policyholders are likely to say "yes" to a vehicle insurance cross-sell offer. This project trains a gradient-boosted classifier (XGBoost) on customer demographics, vehicle details, and policy history to output a probability of interest, then exposes that model through a lightweight prediction API.

## Project Structure

```
.
├── train.py                              # Feature engineering + model training pipeline
├── predict.py                            # FastAPI app that serves predictions
├── insurance_comp.py                     # Example client script hitting the API
├── insurance_cross_sell_prediction.ipynb # Exploratory data analysis / experimentation notebook
├── insurance_model.bin                   # Pickled model artifacts (model, scaler, encodings)
├── src/insurance_cross_sell_prediction/  # Package source
├── Dockerfile                            # Container build for the prediction service
├── pyproject.toml                        # Project metadata & dependencies (uv)
├── uv.lock                               # Locked dependency versions
├── requirements.txt                      # Plain pip-compatible dependency list
└── .python-version                       # Python 3.12
```

## Exploratory Notebook

`insurance_cross_sell_prediction.ipynb` documents the experimentation behind `train.py` — open it to see how the modeling decisions were made:

- **EDA** — missing values, class balance, distributions and outliers for numeric columns, response rates by category (e.g. `Policy_Sales_Channel`, `Region_Code`)
- **Class imbalance strategies** — compares no resampling against **SMOTE** oversampling and **Neighbourhood Cleaning Rule** undersampling
- **Model comparison** — Logistic Regression, Random Forest, and XGBoost, each evaluated across all three resampling strategies
- **Conclusion** — XGBoost with no resampling performed best and was carried forward
- **Hyperparameter tuning** — grid search over XGBoost parameters, followed by threshold tuning to maximize F1 on the validation set (this is where the `BEST_PARAMS` and `BEST_THRESHOLD` values in `train.py` come from)
- **Final evaluation** — retrains on the full training set and reports metrics on the held-out test set

`train.py` is the productionized version of this notebook: same feature engineering, but with the exploration and model-comparison steps stripped out in favor of the winning configuration.

> **Note:** the notebook additionally uses `imbalanced-learn` (for SMOTE / NeighbourhoodCleaningRule) and `matplotlib` for plotting. These are exploratory-only dependencies — they're not required to run `train.py` or `predict.py`, and aren't listed in `requirements.txt`. Install them separately if you want to run the notebook:
> ```bash
> uv add imbalanced-learn matplotlib
> # or: pip install imbalanced-learn matplotlib
> ```

## Model Pipeline

`train.py` handles the full training pipeline:

1. **Data loading** — downloads the dataset via `kagglehub`.
2. **Feature engineering**
   - `Annual_Premium_log`: log-transformed premium
   - `Premium_per_Vintage`: premium normalized by customer tenure
   - `Damaged_and_Uninsured`: flag for customers with vehicle damage and no prior insurance
3. **Encoding**
   - Ordinal encoding for `Vehicle_Age`
   - Binary encoding for `Vehicle_Damage` and `Gender`
   - Smoothed target encoding for high-cardinality `Region_Code` and `Policy_Sales_Channel`
4. **Scaling** — `StandardScaler` on the final feature set
5. **Model** — `XGBClassifier`, tuned via grid search, with `scale_pos_weight` to handle class imbalance
6. **Thresholding** — a custom decision threshold (0.65) applied to predicted probabilities rather than the default 0.5
7. **Artifacts** — model, scaler, encoding maps, and column lists are pickled together to `insurance_model.bin` for use at inference time

## Getting Started

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

### Setup with uv

```bash
git clone https://github.com/sanvi-g07/insurance-cross-sell-prediction.git
cd insurance-cross-sell-prediction
uv sync
```

### Setup with pip

```bash
git clone https://github.com/sanvi-g07/insurance-cross-sell-prediction.git
cd insurance-cross-sell-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train the model

Downloads the dataset, trains the XGBoost model, and writes `insurance_model.bin`.

```bash
uv run train.py
```

### 2. Serve predictions

Starts the FastAPI app on port 8080.

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 8080
```

### 3. Request a prediction

```bash
uv run insurance_comp.py
```

Or call the endpoint directly:

```bash
curl -X POST http://0.0.0.0:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Gender": "Male",
        "Age": 35,
        "Driving_License": 1,
        "Region_Code": 28,
        "Previously_Insured": 0,
        "Vehicle_Age": "1-2 Year",
        "Vehicle_Damage": "Yes",
        "Annual_Premium": 32000.0,
        "Policy_Sales_Channel": 26,
        "Vintage": 150
      }'
```

**Example response:**

```json
{
  "convert_prob": 0.71,
  "convert": true,
  "convert_response": "Interested in vehicle insurance"
}
```

## API Reference

### `POST /predict`

**Request body**

| Field | Type | Description |
|---|---|---|
| `Gender` | `"Male"` \| `"Female"` | Customer gender |
| `Age` | int | Customer age |
| `Driving_License` | `0` \| `1` | Whether the customer holds a driving license |
| `Region_Code` | int | Encoded region identifier |
| `Previously_Insured` | `0` \| `1` | Whether the customer already has vehicle insurance |
| `Vehicle_Age` | `"< 1 Year"` \| `"1-2 Year"` \| `"> 2 Years"` | Age of the customer's vehicle |
| `Vehicle_Damage` | `"Yes"` \| `"No"` | Whether the customer had previous vehicle damage |
| `Annual_Premium` | float | Health insurance premium paid annually |
| `Policy_Sales_Channel` | int | Encoded sales channel identifier (mail, phone, in-person, etc.) |
| `Vintage` | int | Number of days the customer has been associated with the company |

**Response body**

| Field | Type | Description |
|---|---|---|
| `convert_prob` | float | Predicted probability of interest in vehicle insurance |
| `convert` | bool | Predicted class using the model's decision threshold |
| `convert_response` | string | Human-readable prediction label |

## Docker

Build and run the prediction service in a container:

```bash
docker build -t insurance-cross-sell-prediction .
docker run -p 8080:8080 insurance-cross-sell-prediction
```

The image installs dependencies via `uv`, copies in `predict.py`, `train.py`, and the pre-trained `insurance_model.bin`, and starts the API with `uvicorn`.

## Dependencies

- `kagglehub` — dataset download
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, scaling, metrics
- `xgboost` — classification model
- `fastapi`, `uvicorn`, `pydantic` — prediction API and request/response validation
- `requests` — example client

Notebook-only (not required for training or serving):
- `imbalanced-learn` — SMOTE / NeighbourhoodCleaningRule resampling experiments
- `matplotlib` — plots