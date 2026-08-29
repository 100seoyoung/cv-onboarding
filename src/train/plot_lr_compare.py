import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


Path("images").mkdir(exist_ok=True)


runs = {
    "lr=0.5": "results/lr0.5_w32_n45000_s42.csv",
    "lr=0.05": "results/base_w32_lr0.05_s42.csv",
    "lr=0.005": "results/lr0.005_w32_n45000_s42.csv",
    "lr=0.0005": "results/lr0.0005_w32_n45000_s42.csv",
}


plt.figure(figsize=(10, 6))


for label, csv_path in runs.items():

    df = pd.read_csv(csv_path)

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label=label
    )


plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Learning Rate Comparison - Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/lr_val_loss_compare.png",
    dpi=150
)

plt.close()


print(
    "Saved: images/lr_val_loss_compare.png"
)