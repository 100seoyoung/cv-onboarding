import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


Path("images").mkdir(exist_ok=True)


runs = {
    "seed=0": "results/base_w32_lr0.05_s0.csv",
    "seed=1": "results/base_w32_lr0.05_s1.csv",
    "seed=2": "results/base_w32_lr0.05_s2.csv",
}


plt.figure(figsize=(10, 6))


for label, csv_path in runs.items():

    df = pd.read_csv(csv_path)

    plt.plot(
        df["epoch"],
        df["val_acc"] * 100,
        label=label
    )


plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Seed Comparison - Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/seed_val_acc_compare.png",
    dpi=150
)

plt.close()


print(
    "Saved: images/seed_val_acc_compare.png"
)