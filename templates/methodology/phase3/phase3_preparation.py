from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS, extract_features, normalize_url


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = PROJECT_ROOT / "static" / "datasets"
IOC_PATH = DATASETS_DIR / "indicators_of_compromise_ioc_20261106.csv"
LEGITIMATE_URLS_PATH = DATASETS_DIR / "urls_leg.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3_output"


def main():
    print("[*] CRISP-DM Phase 3: Data Preparation")

    print(f"\n[*] Loading malicious IoC records from {IOC_PATH.name}...")
    df_ioc = pd.read_csv(IOC_PATH, usecols=["value", "type"])
    df_ioc = df_ioc[df_ioc["type"].isin(["FQDN", "URL"])].dropna(
        subset=["value"]
    )
    df_ioc = df_ioc.drop_duplicates(subset=["value"])
    df_ioc = df_ioc[["value"]].rename(columns={"value": "raw_url"})
    df_ioc["is_phishing"] = 1

    print(f"\n[*] Loading legitimate URLs from {LEGITIMATE_URLS_PATH.name}...")
    df_legit = pd.read_csv(LEGITIMATE_URLS_PATH, usecols=["url", "etiqueta"])
    df_legit = df_legit.dropna(subset=["url", "etiqueta"]).copy()
    labels = df_legit["etiqueta"].astype(str).str.strip().str.casefold()
    if not labels.eq("legitima").all():
        raise ValueError(
            "urls_leg.csv debe contener únicamente etiquetas 'legitima'."
        )
    df_legit = df_legit[["url"]].rename(columns={"url": "raw_url"})
    df_legit["is_phishing"] = 0

    df_final = pd.concat([df_ioc, df_legit], ignore_index=True)
    print("[*] Normalizing and deduplicating URLs...")
    df_final["normalized_url"] = df_final["raw_url"].map(normalize_url)
    df_final = df_final[df_final["normalized_url"] != ""].copy()
    # Keep the malicious label if a URL appears in both source datasets.
    df_final = (
        df_final.sort_values("is_phishing", kind="stable")
        .drop_duplicates(subset=["normalized_url"], keep="last")
        .reset_index(drop=True)
    )

    class_counts = df_final["is_phishing"].value_counts()
    if set(class_counts.index) != {0, 1}:
        raise ValueError("Se requieren URLs legítimas y maliciosas para entrenar.")
    balance_size = int(class_counts.min())
    df_final = (
        df_final.groupby("is_phishing", group_keys=False)
        .sample(n=balance_size, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    print(f" -> {balance_size} URLs por clase después del balanceo 50/50.")

    print("[*] Extracting features from normalized URLs...")
    features = pd.DataFrame.from_records(
        df_final["normalized_url"].map(extract_features),
        index=df_final.index,
    )
    model_data = pd.concat(
        [df_final[["raw_url", "normalized_url", "is_phishing"]], features],
        axis=1,
    )

    print("\n[*] Splitting the dataset (80% training / 20% test)...")
    X = model_data[FEATURE_COLUMNS]
    y = model_data["is_phishing"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    print(f"\n[*] Saving generated files to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(OUTPUT_DIR / "balanced_dataset_complete.csv", index=False)
    df_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    df_test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nPhase 3 completed successfully!")
    print(f" -> Training output (train.csv): {len(df_train)} rows")
    print(f" -> Test output (test.csv): {len(df_test)} rows")


if __name__ == "__main__":
    main()
