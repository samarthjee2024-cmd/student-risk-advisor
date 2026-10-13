from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student-por.csv"
MODEL_PATH = BASE_DIR / "student_risk_pipeline.joblib"

frame = pd.read_csv(DATA_PATH, sep=";")
y = (frame["G3"] < 10).astype(int)
X = frame.drop(columns=["G1", "G2", "G3"])
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = [column for column in X.columns if column not in categorical]
preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numerical", StandardScaler(), numerical),
    ]
)
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=300,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)
pipeline.fit(X, y)
joblib.dump(
    {
        "pipeline": pipeline,
        "features": X.columns.tolist(),
        "training_note": "Refitted on all 649 public records for demonstration inference only; not the held-out evaluation model.",
    },
    MODEL_PATH,
)
print(f"Saved {MODEL_PATH.name} with {len(X.columns)} predictors and {len(X)} records.")
