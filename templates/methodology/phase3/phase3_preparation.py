import os
import pandas as pd
from sklearn.model_selection import train_test_split
from features import FEATURE_COLUMNS, extract_features, normalize_url


def main():
    print("[*] CRISP-DM Phase 3: Data Preparation")

    # 1. LOAD AND CLEAN THE MALICIOUS CLASS (IoC)
    print("\n[*] Loading malicious dataset (IoC)...")
    df_ioc = pd.read_csv(
        r"T:\Machine_Learning_6\Proyecto_MachineL\static\datasets\indicators_of_compromise_ioc_20261106.csv"
    )

    # Keep only domains and URLs, then remove duplicates.
    df_ioc = df_ioc[df_ioc["type"].isin(["FQDN", "URL"])].copy()
    df_ioc = df_ioc.drop_duplicates(subset=["value"])
    df_ioc = df_ioc[["value"]].rename(columns={"value": "raw_url"})

    # Assign the label: 1 = phishing/malicious.
    df_ioc["is_phishing"] = 1
    num_malicious = len(df_ioc)
    print(f" -> {num_malicious} unique malicious records ready.")

    # 2. LOAD AND CLEAN THE LEGITIMATE CLASS (top-1m.csv)
    print("\n[*] Loading legitimate dataset (top-1m.csv)...")
    # This file has no headers: column 0 is rank and column 1 is the domain.
    df_legit = pd.read_csv(
        r"T:\Machine_Learning_6\Proyecto_MachineL\static\datasets\top-1m.csv",
        header=None,
        names=["rank", "raw_url"],
    )
    df_legit = df_legit[["raw_url"]].copy()

    # CRITICAL BALANCING: randomly select as many benign records as malicious records.
    if len(df_legit) > num_malicious:
        df_legit = df_legit.sample(n=num_malicious, random_state=42)

    # Assign the label: 0 = legitimate/safe.
    df_legit["is_phishing"] = 0
    print(f" -> {len(df_legit)} legitimate records selected (50/50 balance).")

    # 3. MERGE DATASETS (integration to prevent bias)
    print("\n[*] Merging and shuffling datasets...")
    df_final = pd.concat([df_ioc, df_legit], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. NORMALIZATION AND FEATURE ENGINEERING
    print("[*] Normalizing URLs and extracting features... (may take a few minutes)")
    df_final["normalized_url"] = df_final["raw_url"].apply(normalize_url)
    df_final = df_final[df_final["normalized_url"] != ""]
    features = df_final["normalized_url"].apply(
        lambda value: pd.Series(extract_features(value))
    )
    model_data = pd.concat(
        [df_final[["raw_url", "normalized_url", "is_phishing"]], features], axis=1
    )

    # 5. SPLIT INTO TRAINING AND TEST SETS
    print("\n[*] Splitting the dataset (80% training / 20% test)...")
    X = model_data[FEATURE_COLUMNS]
    y = model_data["is_phishing"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # 6. EXPORT OUTPUT FILES
    print("[*] Saving generated files...")
    output_folder = "phase3_output"
    os.makedirs(output_folder, exist_ok=True)
    model_data.to_csv(f"{output_folder}/balanced_dataset_complete.csv", index=False)
    df_train.to_csv(f"{output_folder}/train.csv", index=False)
    df_test.to_csv(f"{output_folder}/test.csv", index=False)

    print("\nPhase 3 completed successfully!")
    print(f" -> Training output (train.csv): {len(df_train)} rows")
    print(f" -> Test output (test.csv):  {len(df_test)} rows")


if __name__ == "__main__":
    main()
