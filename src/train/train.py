import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml

from data import get_dataloaders
from model import SmallCNN


# ==========================================
# Argument
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    default="src/train/configs/baseline.yaml"
)

parser.add_argument(
    "--lr",
    type=float,
    default=None
)

parser.add_argument(
    "--seed",
    type=int,
    default=None
)

parser.add_argument(
    "--width",
    type=int,
    default=None
)

parser.add_argument(
    "--train-size",
    type=int,
    default=None
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=None
)

parser.add_argument(
    "--dropout",
    type=float,
    default=None
)

args = parser.parse_args()


# ==========================================
# Config 읽기
# ==========================================

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)


# ==========================================
# CLI 값이 있으면 YAML 덮어쓰기
# ==========================================

if args.lr is not None:
    cfg["lr"] = args.lr

if args.seed is not None:
    cfg["seed"] = args.seed

if args.width is not None:
    cfg["width"] = args.width

if args.train_size is not None:
    cfg["train_size"] = args.train_size

if args.batch_size is not None:
    cfg["batch_size"] = args.batch_size

if args.dropout is not None:
    cfg["dropout"] = args.dropout


print("\nConfig loaded:")

for key, value in cfg.items():
    print(f"{key}: {value}")


# ==========================================
# Seed 고정
# ==========================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg["seed"])


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\ndevice:", device)


# ==========================================
# DataLoader
# ==========================================

train_loader, val_loader, test_loader = get_dataloaders(
    train_size=cfg["train_size"],
    batch_size=cfg["batch_size"],
    seed=cfg["seed"],
    augmentation=cfg["augmentation"]
)


print(
    "train size:",
    len(train_loader.dataset)
)

print(
    "val size:",
    len(val_loader.dataset)
)

print(
    "test size:",
    len(test_loader.dataset)
)


# ==========================================
# Model
# ==========================================

model = SmallCNN(
    width=cfg["width"],
    dropout_p=cfg["dropout"]
).to(device)


# ==========================================
# Loss
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# Optimizer
# ==========================================

optimizer = optim.SGD(
    model.parameters(),
    lr=cfg["lr"],
    momentum=cfg["momentum"],
    weight_decay=cfg["weight_decay"]
)


# ==========================================
# Scheduler
# ==========================================

scheduler = None

scheduler_setting = cfg.get(
    "scheduler",
    False
)

if (
    scheduler_setting is True
    or scheduler_setting == "cosine"
):

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"]
    )


# ==========================================
# 설정
# ==========================================

EPOCHS = cfg["epochs"]

PATIENCE = cfg.get(
    "patience",
    10
)

EARLY_STOPPING = cfg.get(
    "early_stopping",
    False
)

RUN_NAME = cfg["run_name"]


# ==========================================
# 저장 폴더
# ==========================================

Path("results").mkdir(
    exist_ok=True
)

Path("checkpoints").mkdir(
    exist_ok=True
)


csv_path = (
    f"results/{RUN_NAME}.csv"
)


# Validation Loss 기준 checkpoint
best_loss_checkpoint_path = (
    f"checkpoints/{RUN_NAME}_best_loss.pt"
)


# Validation Accuracy 기준 checkpoint
best_acc_checkpoint_path = (
    f"checkpoints/{RUN_NAME}_best_acc.pt"
)


# ==========================================
# W&B
# ==========================================

wandb.init(
    project="cifar10-onboarding",
    name=RUN_NAME,
    config=cfg
)


# ==========================================
# CSV 준비
# ==========================================

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc"
    ])


# ==========================================
# Best 결과
# ==========================================

best_val_loss = float("inf")
best_val_loss_epoch = 0

best_val_acc = 0.0
best_val_acc_epoch = 0

patience_counter = 0


# ==========================================
# Training
# ==========================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------
    # Train
    # --------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0


    for x, y in train_loader:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(x)


        loss = criterion(
            outputs,
            y
        )


        loss.backward()


        optimizer.step()


        train_loss += (
            loss.item()
            * x.size(0)
        )


        predicted = outputs.argmax(
            dim=1
        )


        train_correct += (
            predicted == y
        ).sum().item()


        train_total += y.size(0)


    train_loss /= train_total

    train_acc = (
        train_correct
        / train_total
    )


    # --------------------------------------
    # Validation
    # --------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )


            outputs = model(x)


            loss = criterion(
                outputs,
                y
            )


            val_loss += (
                loss.item()
                * x.size(0)
            )


            predicted = outputs.argmax(
                dim=1
            )


            val_correct += (
                predicted == y
            ).sum().item()


            val_total += y.size(0)


    val_loss /= val_total

    val_acc = (
        val_correct
        / val_total
    )


    # ======================================
    # Best Validation Accuracy
    # ======================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_val_acc_epoch = epoch


        torch.save(
            model.state_dict(),
            best_acc_checkpoint_path
        )


        print(
            f"  -> Best Accuracy model saved "
            f"(Val Acc: {val_acc * 100:.2f}%)"
        )


    # ======================================
    # Best Validation Loss
    # ======================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_loss_epoch = epoch

        patience_counter = 0


        torch.save(
            model.state_dict(),
            best_loss_checkpoint_path
        )


        print(
            f"  -> Best Loss model saved "
            f"(Val Loss: {val_loss:.4f})"
        )

    else:

        patience_counter += 1


    # ======================================
    # Learning Rate
    # ======================================

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ======================================
    # CSV 저장
    # ======================================

    with open(
        csv_path,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            current_lr,
            train_loss,
            train_acc,
            val_loss,
            val_acc
        ])


    # ======================================
    # W&B 기록
    # ======================================

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/acc": val_acc,
        "lr": current_lr
    })


    # ======================================
    # 출력
    # ======================================

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}%"
    )


    # ======================================
    # Scheduler
    # ======================================

    if scheduler is not None:
        scheduler.step()


    # ======================================
    # Early Stopping
    # ======================================

    if (
        EARLY_STOPPING
        and patience_counter >= PATIENCE
    ):

        print(
            f"\nEarly stopping "
            f"at epoch {epoch}"
        )

        break


# ==========================================
# 학습 결과
# ==========================================

print(
    "\nTraining finished"
)


print(
    f"Best Val Loss: "
    f"{best_val_loss:.4f} "
    f"(Epoch {best_val_loss_epoch})"
)


print(
    f"Best Val Accuracy: "
    f"{best_val_acc * 100:.2f}% "
    f"(Epoch {best_val_acc_epoch})"
)


print(
    f"Best Loss checkpoint: "
    f"{best_loss_checkpoint_path}"
)


print(
    f"Best Accuracy checkpoint: "
    f"{best_acc_checkpoint_path}"
)


print(
    f"CSV: {csv_path}"
)


# ==========================================
# 평가 함수
# ==========================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0


    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )


            outputs = model(x)


            loss = criterion(
                outputs,
                y
            )


            total_loss += (
                loss.item()
                * x.size(0)
            )


            predicted = outputs.argmax(
                dim=1
            )


            total_correct += (
                predicted == y
            ).sum().item()


            total += y.size(0)


    avg_loss = (
        total_loss / total
    )

    accuracy = (
        total_correct / total
    )


    return (
        avg_loss,
        accuracy
    )


# ==========================================
# Best Loss checkpoint 재로드
# ==========================================

best_loss_model = SmallCNN(
    width=cfg["width"],
    dropout_p=cfg["dropout"]
).to(device)


best_loss_model.load_state_dict(
    torch.load(
        best_loss_checkpoint_path,
        map_location=device,
        weights_only=True
    )
)


reload_loss_val_loss, reload_loss_val_acc = (
    evaluate_model(
        best_loss_model,
        val_loader
    )
)


print(
    "\nReloaded Best Loss checkpoint"
)


print(
    f"Val Loss: "
    f"{reload_loss_val_loss:.4f}"
)


print(
    f"Val Acc: "
    f"{reload_loss_val_acc * 100:.2f}%"
)


# ==========================================
# Best Accuracy checkpoint 재로드
# ==========================================

best_acc_model = SmallCNN(
    width=cfg["width"],
    dropout_p=cfg["dropout"]
).to(device)


best_acc_model.load_state_dict(
    torch.load(
        best_acc_checkpoint_path,
        map_location=device,
        weights_only=True
    )
)


reload_acc_val_loss, reload_acc_val_acc = (
    evaluate_model(
        best_acc_model,
        val_loader
    )
)


print(
    "\nReloaded Best Accuracy checkpoint"
)


print(
    f"Val Loss: "
    f"{reload_acc_val_loss:.4f}"
)


print(
    f"Val Acc: "
    f"{reload_acc_val_acc * 100:.2f}%"
)


# ==========================================
# W&B Summary
# ==========================================

wandb.summary[
    "best_val_loss"
] = best_val_loss

wandb.summary[
    "best_val_loss_epoch"
] = best_val_loss_epoch

wandb.summary[
    "best_val_acc"
] = best_val_acc

wandb.summary[
    "best_val_acc_epoch"
] = best_val_acc_epoch


wandb.summary[
    "reload_best_loss_val_loss"
] = reload_loss_val_loss

wandb.summary[
    "reload_best_loss_val_acc"
] = reload_loss_val_acc


wandb.summary[
    "reload_best_acc_val_loss"
] = reload_acc_val_loss

wandb.summary[
    "reload_best_acc_val_acc"
] = reload_acc_val_acc


wandb.finish()