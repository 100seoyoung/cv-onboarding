import csv
from pathlib import Path

import matplotlib.pyplot as plt


csv_path = "results/overfit.csv"

epochs = []
train_losses = []
val_losses = []
train_accs = []
val_accs = []


with open(csv_path, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        epochs.append(int(row["epoch"]))
        train_losses.append(float(row["train_loss"]))
        val_losses.append(float(row["val_loss"]))
        train_accs.append(float(row["train_acc"]))
        val_accs.append(float(row["val_acc"]))


Path("images").mkdir(exist_ok=True)


# Loss
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting Loss Curve")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("images/overfit_loss.png", dpi=150)
plt.close()


# Accuracy
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Overfitting Accuracy Curve")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("images/overfit_accuracy.png", dpi=150)
plt.close()


print("Saved: images/overfit_loss.png")
print("Saved: images/overfit_accuracy.png")