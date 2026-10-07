"""
explore_dataset.py
------------------
Executes Phase 2 dataset exploration on KDDTrain+.txt and KDDTest+.txt.
Computes genuine statistics, generates exploratory visualizations,
and exports exploration_summary.json to results/.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (
    load_dataset,
    COLUMN_NAMES,
    TRAFFIC_FEATURES,
    METADATA_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    map_attack_category
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def run_exploration():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")
    
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)
    
    print(f"KDDTrain+ shape: {train_df.shape}")
    print(f"KDDTest+  shape: {test_df.shape}")
    
    # 1. Verification of Structure
    print("\n" + "=" * 60)
    print("DATA STRUCTURE & TYPES")
    print("=" * 60)
    print(f"Total Columns: {len(COLUMN_NAMES)}")
    print(f"Traffic Features: {len(TRAFFIC_FEATURES)}")
    print(f"Categorical Features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"Numerical Features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    
    # Check column dtypes
    dtypes_summary = train_df.dtypes.value_counts().to_dict()
    print(f"Train Dtypes count: {dtypes_summary}")
    
    # First 5 rows sample
    print("\n--- KDDTrain+ First 5 rows sample (key columns) ---")
    print(train_df[["duration", "protocol_type", "service", "flag", "src_bytes", "label", "difficulty"]].head())
    
    # 2. Missing Values Analysis
    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    train_nulls = int(train_df.isnull().sum().sum())
    test_nulls = int(test_df.isnull().sum().sum())
    print(f"Train total missing values: {train_nulls}")
    print(f"Test total missing values:  {test_nulls}")
    if train_nulls == 0 and test_nulls == 0:
        print("No missing values detected.")
    else:
        print(f"Train columns with nulls: {train_df.isnull().sum()[train_df.isnull().sum() > 0]}")
        print(f"Test columns with nulls: {test_df.isnull().sum()[test_df.isnull().sum() > 0]}")
        
    # 3. Duplicate Analysis
    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)
    train_duplicates = int(train_df.duplicated().sum())
    test_duplicates = int(test_df.duplicated().sum())
    print(f"Train duplicate rows: {train_duplicates}")
    print(f"Test duplicate rows:  {test_duplicates}")
    
    # 4. Label Exploration
    print("\n" + "=" * 60)
    print("LABEL EXPLORATION")
    print("=" * 60)
    train_labels = train_df["label"].value_counts()
    test_labels = test_df["label"].value_counts()
    print(f"Distinct attack/normal labels in Train: {len(train_labels)}")
    print(f"Distinct attack/normal labels in Test:  {len(test_labels)}")
    
    train_normal_count = int((train_df["label"] == "normal").sum())
    train_attack_count = int((train_df["label"] != "normal").sum())
    test_normal_count = int((test_df["label"] == "normal").sum())
    test_attack_count = int((test_df["label"] != "normal").sum())
    
    print(f"Train Normal: {train_normal_count:,} ({train_normal_count/len(train_df)*100:.2f}%)")
    print(f"Train Attack: {train_attack_count:,} ({train_attack_count/len(train_df)*100:.2f}%)")
    print(f"Test Normal:  {test_normal_count:,} ({test_normal_count/len(test_df)*100:.2f}%)")
    print(f"Test Attack:  {test_attack_count:,} ({test_attack_count/len(test_df)*100:.2f}%)")
    
    # 5. Attack Category Mapping
    print("\n" + "=" * 60)
    print("ATTACK CATEGORY ANALYSIS")
    print("=" * 60)
    train_categories = train_df["label"].apply(map_attack_category).value_counts()
    test_categories = test_df["label"].apply(map_attack_category).value_counts()
    
    print("\nTrain Attack Categories:")
    print(train_categories.to_string())
    print("\nTest Attack Categories:")
    print(test_categories.to_string())
    
    # Check for unmapped labels
    train_unmapped = train_df[train_df["label"].apply(map_attack_category) == "Unknown"]["label"].unique()
    test_unmapped = test_df[test_df["label"].apply(map_attack_category) == "Unknown"]["label"].unique()
    if len(train_unmapped) > 0:
        print(f"Unmapped in Train: {train_unmapped}")
    if len(test_unmapped) > 0:
        print(f"Unmapped in Test: {test_unmapped}")
        
    # 6. Basic Numerical Statistics
    print("\n" + "=" * 60)
    print("NUMERICAL DESCRIPTIVE STATISTICS (SAMPLE FEATURES)")
    print("=" * 60)
    sample_num_cols = ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count"]
    print(train_df[sample_num_cols].describe().round(2).to_string())
    
    # 7. Generate Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Plot 1: Normal vs Attack Distribution (Train vs Test)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    dist_data = pd.DataFrame({
        "Split": ["KDDTrain+", "KDDTrain+", "KDDTest+", "KDDTest+"],
        "Class": ["Normal", "Attack", "Normal", "Attack"],
        "Count": [train_normal_count, train_attack_count, test_normal_count, test_attack_count],
        "Percentage": [
            train_normal_count / len(train_df) * 100,
            train_attack_count / len(train_df) * 100,
            test_normal_count / len(test_df) * 100,
            test_attack_count / len(test_df) * 100
        ]
    })
    
    # Count Barplot
    palette = ["#2b5c8f", "#d95f02"]
    sns.barplot(data=dist_data, x="Split", y="Count", hue="Class", palette=palette, ax=axes[0])
    axes[0].set_title("Normal vs Attack Sample Counts", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Samples", fontsize=11)
    axes[0].set_xlabel("Dataset Split", fontsize=11)
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(f"{int(height):,}",
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                             textcoords='offset points')
            
    # Percentage Barplot
    sns.barplot(data=dist_data, x="Split", y="Percentage", hue="Class", palette=palette, ax=axes[1])
    axes[1].set_title("Class Ratio Proportion (%)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].set_xlabel("Dataset Split", fontsize=11)
    axes[1].set_ylim(0, 100)
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f"{height:.1f}%",
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                             textcoords='offset points')
            
    plt.tight_layout()
    plot1_path = os.path.join(RESULTS_DIR, "label_distribution.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # Plot 2: Attack Category Distribution (Train vs Test)
    categories_df = pd.DataFrame({
        "Category": list(train_categories.index) + list(test_categories.index),
        "Count": list(train_categories.values) + list(test_categories.values),
        "Split": ["KDDTrain+"] * len(train_categories) + ["KDDTest+"] * len(test_categories)
    })
    
    plt.figure(figsize=(10, 6))
    cat_palette = {"KDDTrain+": "#1f77b4", "KDDTest+": "#ff7f0e"}
    ax = sns.barplot(data=categories_df, x="Category", y="Count", hue="Split", palette=cat_palette,
                     order=["Normal", "DoS", "Probe", "R2L", "U2R"])
    plt.title("Distribution of Traffic by Broad Category (NSL-KDD)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Record Count", fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height):,}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plot2_path = os.path.join(RESULTS_DIR, "attack_category_distribution.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")
    
    # 8. Export Summary JSON
    summary_data = {
        "dataset_files": {
            "train_file": train_path,
            "test_file": test_path
        },
        "shapes": {
            "train_rows": int(train_df.shape[0]),
            "train_columns": int(train_df.shape[1]),
            "test_rows": int(test_df.shape[0]),
            "test_columns": int(test_df.shape[1])
        },
        "features": {
            "total_traffic_features": len(TRAFFIC_FEATURES),
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features_count": len(NUMERICAL_FEATURES),
            "metadata_columns": METADATA_COLUMNS
        },
        "missing_values": {
            "train_missing_total": train_nulls,
            "test_missing_total": test_nulls,
            "status": "No missing values detected" if (train_nulls == 0 and test_nulls == 0) else "Missing values found"
        },
        "duplicates": {
            "train_duplicates": train_duplicates,
            "test_duplicates": test_duplicates
        },
        "class_distribution": {
            "train": {
                "normal": train_normal_count,
                "attack": train_attack_count,
                "normal_pct": round(train_normal_count / len(train_df) * 100, 2),
                "attack_pct": round(train_attack_count / len(train_df) * 100, 2)
            },
            "test": {
                "normal": test_normal_count,
                "attack": test_attack_count,
                "normal_pct": round(test_normal_count / len(test_df) * 100, 2),
                "attack_pct": round(test_attack_count / len(test_df) * 100, 2)
            }
        },
        "attack_categories": {
            "train": {k: int(v) for k, v in train_categories.items()},
            "test": {k: int(v) for k, v in test_categories.items()}
        },
        "visualizations": [
            "results/label_distribution.png",
            "results/attack_category_distribution.png"
        ]
    }
    
    summary_json_path = os.path.join(RESULTS_DIR, "exploration_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved: {summary_json_path}")
    print("\nPhase 2 Dataset Exploration completed successfully.")

if __name__ == "__main__":
    run_exploration()
