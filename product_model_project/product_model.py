import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
DATA = "merged_dataset.csv"
TARGET = "product_category"

NUM = ["purchase_amount", "log_amount", "customer_rating", "purchase_month", "purchase_dow",
       "is_weekend", "rating_missing", "has_social_profile", "n_social_profiles", "n_platforms",
       "mean_engagement", "max_engagement", "mean_interest", "max_interest", "mean_sentiment",
       "pct_positive", "pct_negative", "interest_x_engagement", "prior_txn_count",
       "prior_avg_amount", "prior_avg_rating"]
CAT = ["primary_platform", "prev_category"]


def load():
    df = pd.read_csv(DATA, keep_default_na=False, na_values=[""])  # keep "None" as a category
    df["prev_category"] = df["prev_category"].fillna("None")
    return df


def make_preprocessor(scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])


def get_models(n_classes):
    return {
        "Baseline (majority)": Pipeline([("pre", make_preprocessor(False)),
                                         ("clf", DummyClassifier(strategy="prior"))]),
        "Logistic Regression": Pipeline([("pre", make_preprocessor(True)),
                                         ("clf", LogisticRegression(C=0.1, max_iter=2000))]),
        "Random Forest": Pipeline([("pre", make_preprocessor(False)),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=300, max_depth=4, min_samples_leaf=3,
                                       random_state=SEED))]),
        "XGBoost": Pipeline([("pre", make_preprocessor(False)),
                             ("clf", XGBClassifier(
                                 n_estimators=100, max_depth=2, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                                 num_class=n_classes, eval_metric="mlogloss", random_state=SEED))]),
    }


def cross_validate(models, X, y, groups, n_repeats=5):
    rows = {name: {"acc": [], "f1": [], "loss": []} for name in models}
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for tr, te in cv.split(X, y, groups):
            for name, model in models.items():
                model.fit(X.iloc[tr], y[tr])
                proba = model.predict_proba(X.iloc[te])
                pred = proba.argmax(axis=1)
                rows[name]["acc"].append(accuracy_score(y[te], pred))
                rows[name]["f1"].append(f1_score(y[te], pred, average="macro"))
                rows[name]["loss"].append(log_loss(y[te], proba, labels=range(proba.shape[1])))
    out = pd.DataFrame({n: {"Accuracy": np.mean(r["acc"]), "Acc std": np.std(r["acc"]),
                            "Macro F1": np.mean(r["f1"]), "Log loss": np.mean(r["loss"])}
                        for n, r in rows.items()}).T
    return out.round(3)


def main():
    df = load()
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X = df[NUM + CAT]
    groups = df["customer_id"].values

    models = get_models(len(le.classes_))
    results = cross_validate(models, X, y, groups)
    print("Classes:", list(le.classes_))
    print("\nGrouped 5-fold CV x5 repeats:\n", results.to_string())

    candidates = results.drop(index="Baseline (majority)")
    best = candidates["Log loss"].idxmin()
    print(f"\nBest model by log loss: {best}")

    final = get_models(len(le.classes_))[best].fit(X, y)
    joblib.dump({"model": final, "label_encoder": le, "num": NUM, "cat": CAT},
                "product_model.joblib")
    results.to_csv("product_model_results.csv")
    print("Saved product_model.joblib and product_model_results.csv")


if __name__ == "__main__":
    main()
