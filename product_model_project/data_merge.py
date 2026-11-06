import numpy as np
import pandas as pd

TXN_PATH = "customer_transactions - customer_transactions.csv"
SOC_PATH = "customer_social_profiles - customer_social_profiles.csv"
OUT_PATH = "merged_dataset.csv"

SENTIMENT_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}


def load_and_clean(txn_path=TXN_PATH, soc_path=SOC_PATH):
    txn = pd.read_csv(txn_path)
    soc = pd.read_csv(soc_path)

    txn = txn.rename(columns={"customer_id_legacy": "customer_id"})
    soc["customer_id"] = soc["customer_id_new"].str.extract(r"(\d+)").astype(int)
    soc = soc.drop(columns="customer_id_new").drop_duplicates()

    txn["purchase_date"] = pd.to_datetime(txn["purchase_date"])
    return txn, soc


def aggregate_social(soc):
    soc = soc.copy()
    soc["sentiment_score"] = soc["review_sentiment"].map(SENTIMENT_MAP)
    g = soc.groupby("customer_id")
    agg = g.agg(
        n_social_profiles=("social_media_platform", "size"),
        n_platforms=("social_media_platform", "nunique"),
        mean_engagement=("engagement_score", "mean"),
        max_engagement=("engagement_score", "max"),
        mean_interest=("purchase_interest_score", "mean"),
        max_interest=("purchase_interest_score", "max"),
        mean_sentiment=("sentiment_score", "mean"),
        pct_positive=("review_sentiment", lambda s: (s == "Positive").mean()),
        pct_negative=("review_sentiment", lambda s: (s == "Negative").mean()),
        primary_platform=("social_media_platform", lambda s: s.mode().iloc[0]),
    ).reset_index()
    return agg


def add_history_features(df):
    df = df.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)
    g = df.groupby("customer_id")
    df["prior_txn_count"] = g.cumcount()
    df["prior_avg_amount"] = g["purchase_amount"].transform(
        lambda s: s.shift().expanding().mean())
    df["prior_avg_rating"] = g["customer_rating"].transform(
        lambda s: s.shift().expanding().mean())
    df["prev_category"] = g["product_category"].shift().fillna("None")
    return df


def build_merged_dataset(save=True):
    txn, soc = load_and_clean()
    soc_agg = aggregate_social(soc)

    df = txn.merge(soc_agg, on="customer_id", how="left")
    df["has_social_profile"] = df["n_social_profiles"].notna().astype(int)

    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dow"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_dow"] >= 5).astype(int)
    df["rating_missing"] = df["customer_rating"].isna().astype(int)
    df["log_amount"] = np.log1p(df["purchase_amount"])

    df = add_history_features(df)

    df["interest_x_engagement"] = df["mean_interest"] * df["mean_engagement"] / 100

    if save:
        df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_merged_dataset()
    print(f"Merged dataset: {df.shape}  -> saved to {OUT_PATH}")
    print(f"Transactions with a social profile: {df.has_social_profile.sum()}/{len(df)}")
    print(df.isna().sum()[lambda s: s > 0])
