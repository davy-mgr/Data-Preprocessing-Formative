import os
import numpy as np
import pandas as pd
import joblib

DATA_PATH = "merged_dataset.csv"
MODEL_PATH = "product_model.joblib"

_cache = {}


def _load():
    if "df" in _cache:
        return _cache["df"], _cache["bundle"]

    df = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[""])
    df["prev_category"] = df["prev_category"].fillna("None")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception:
        import product_model  # fallback: retrain with the same pipeline
        product_model.main()
        bundle = joblib.load(MODEL_PATH)

    _cache["df"], _cache["bundle"] = df, bundle
    return df, bundle


def _parse_id(customer_id):
    return int(str(customer_id).upper().lstrip("A"))


def _build_features(cust):
    cust = cust.sort_values(["purchase_date", "transaction_id"])
    last = cust.iloc[-1]
    amount = cust["purchase_amount"].mean()
    rating = cust["customer_rating"].mean()  # NaN if they never rated

    row = {
        "purchase_amount": amount,
        "log_amount": np.log1p(amount),
        "customer_rating": rating,
        "purchase_month": last["purchase_month"],
        "purchase_dow": last["purchase_dow"],
        "is_weekend": last["is_weekend"],
        "rating_missing": int(np.isnan(rating)),
        "has_social_profile": last["has_social_profile"],
        "n_social_profiles": last["n_social_profiles"],
        "n_platforms": last["n_platforms"],
        "mean_engagement": last["mean_engagement"],
        "max_engagement": last["max_engagement"],
        "mean_interest": last["mean_interest"],
        "max_interest": last["max_interest"],
        "mean_sentiment": last["mean_sentiment"],
        "pct_positive": last["pct_positive"],
        "pct_negative": last["pct_negative"],
        "primary_platform": last["primary_platform"],
        "interest_x_engagement": last["interest_x_engagement"],
        "prior_txn_count": len(cust),
        "prior_avg_amount": amount,
        "prior_avg_rating": rating,
        "prev_category": last["product_category"],
    }
    return row


def predict_product(customer_id, top_k=3):
    df, bundle = _load()
    cid = _parse_id(customer_id)

    cust = df[df["customer_id"] == cid]
    if cust.empty:
        raise KeyError(f"Customer {customer_id} has no transaction history.")

    row = _build_features(cust)
    X = pd.DataFrame([row])[bundle["num"] + bundle["cat"]]

    proba = bundle["model"].predict_proba(X)[0]
    classes = bundle["label_encoder"].classes_
    order = np.argsort(proba)[::-1][:top_k]
    ranked = [(str(classes[i]), round(float(proba[i]), 3)) for i in order]

    return {
        "customer_id": cid,
        "top_prediction": ranked[0][0],
        "confidence": ranked[0][1],
        "top_k": ranked,
    }

if __name__ == "__main__":
    import sys
    cid = sys.argv[1] if len(sys.argv) > 1 else "151"
    try:
        res = predict_product(cid)
    except KeyError as e:
        print(f"Unknown customer: {e}")
        sys.exit(1)
    print(f"Customer {res['customer_id']} -> recommended: {res['top_prediction']} "
          f"(confidence {res['confidence']:.0%})")
    for cat, p in res["top_k"]:
        print(f"  {cat:12s} {p:.1%}")