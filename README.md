# Cardiac Risk Assessment

A machine learning web app that estimates heart disease risk from patient vitals, built with scikit-learn and deployed with Streamlit.

## Dataset
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) — 918 patient records, 11 clinical attributes, combining 5 independent hospital datasets (Cleveland, Hungary, Switzerland, Long Beach VA, Stalog).

| Attribute | Description |
|---|---|
| Age | Patient's age in years |
| Sex | M = Male, F = Female |
| ChestPainType | TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | 1 if fasting blood sugar > 120 mg/dl, else 0 |
| RestingECG | Normal, ST, or LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y = Yes, N = No |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, or Down |
| HeartDisease | Target: 1 = disease present, 0 = normal |

## Model
Three models were trained and compared by accuracy and ROC-AUC: Logistic Regression, Random Forest, and Gradient Boosting. The best-performing model is used in the app (see `model_metadata.json` for the comparison table).

## Tech stack
- scikit-learn (preprocessing pipeline + model training)
- Streamlit (web app + deployment)

## Files
- `app.py` — Streamlit app
- `heart_model.joblib` — trained pipeline (preprocessing + model)
- `model_metadata.json` — feature ranges, options, and model comparison stats used by the app
- `requirements.txt` — Python dependencies

## Disclaimer
Educational project only — not a medical device and not a substitute for professional diagnosis.
