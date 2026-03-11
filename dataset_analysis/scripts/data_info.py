import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

train_df = pd.read_csv("./open/train.csv")
dev_df   = pd.read_csv("./open/dev.csv")
sub_df   = pd.read_csv("./open/sample_submission.csv")

print("=== train_df ===")
print(train_df.head())
print(train_df.columns)
print(train_df["label"].value_counts(dropna=False))

print("\n=== dev_df ===")
print(dev_df.head())
print(dev_df.columns)
print(dev_df["label"].value_counts(dropna=False))

print("\n=== sample_submission ===")
print(sub_df.head())
print(sub_df.columns)

def check_integrity(df, root_dir, n_check=200):
    missing = []
    for i in range(min(n_check, len(df))):
        sample_id = str(df.iloc[i]["id"])
        folder = os.path.join(root_dir, sample_id)
        front = os.path.join(folder, "front.png")
        top   = os.path.join(folder, "top.png")

        ok = os.path.isdir(folder) and os.path.isfile(front) and os.path.isfile(top)
        if not ok:
            missing.append((sample_id, os.path.isdir(folder), os.path.isfile(front), os.path.isfile(top)))
    return missing

miss_train = check_integrity(train_df, "./open/train", n_check=500)
miss_dev   = check_integrity(dev_df,   "./open/dev",   n_check=500)
miss_test  = check_integrity(sub_df,   "./open/test",  n_check=500)

print("train missing sample examples:", miss_train[:5], "count:", len(miss_train))
print("dev missing sample examples:", miss_dev[:5], "count:", len(miss_dev))
print("test missing sample examples:", miss_test[:5], "count:", len(miss_test))

def show_sample(root_dir, sample_id):
    folder = os.path.join(root_dir, str(sample_id))
    front_path = os.path.join(folder, "front.png")
    top_path   = os.path.join(folder, "top.png")

    front = Image.open(front_path).convert("RGB")
    top   = Image.open(top_path).convert("RGB")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(front)
    plt.title(f"{sample_id} - front")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(top)
    plt.title(f"{sample_id} - top")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# 예시: train 첫 샘플
show_sample("./open/train", train_df.iloc[0]["id"])