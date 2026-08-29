import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from data import (
    make_split,
    make_small_train_indices,
    CIFAR10_MEAN,
    CIFAR10_STD
)

from model import SmallCNN


# ==========================================
# 기본 설정
# ==========================================

SEED = 42

TRAIN_SIZE = 500
VAL_SIZE = 5000

BATCH_SIZE = 32

EPOCHS = 30

LR = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device:", device)


# ==========================================
# 저장 폴더
# ==========================================

Path("results").mkdir(
    exist_ok=True
)

Path("checkpoints").mkdir(
    exist_ok=True
)


# ==========================================
# Seed
# ==========================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================
# Transform
# ==========================================

# SmallCNN / ResNet18 scratch
cifar_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ImageNet pretrained ResNet18
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225
)


imagenet_transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ==========================================
# Dataset index 준비
# ==========================================

raw_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)


train_indices, val_indices = make_split(
    SEED
)


selected_train_indices = (
    make_small_train_indices(
        train_indices,
        raw_dataset.targets,
        TRAIN_SIZE
    )
)


print("\nDataset")

print(
    "Train:",
    len(selected_train_indices)
)

print(
    "Validation:",
    len(val_indices)
)


# ==========================================
# DataLoader 생성 함수
# ==========================================

def get_loaders(transform):

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    val_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    train_set = Subset(
        train_dataset,
        selected_train_indices
    )


    val_set = Subset(
        val_dataset,
        val_indices
    )


    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    return (
        train_loader,
        val_loader
    )


# ==========================================
# 평가 함수
# ==========================================

def evaluate(
    model,
    loader,
    criterion
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
# 학습 함수
# ==========================================

def train_model(
    model_name,
    model,
    train_loader,
    val_loader
):

    print(
        "\n"
        "=========================================="
    )

    print(
        f"Training: {model_name}"
    )

    print(
        "=========================================="
    )


    # 같은 조건에서 시작
    set_seed(SEED)


    model = model.to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )


    checkpoint_path = (
        f"checkpoints/"
        f"{model_name}_best_acc.pt"
    )


    csv_path = (
        f"results/"
        f"{model_name}.csv"
    )


    # ======================================
    # W&B
    # ======================================

    wandb.init(
        project="cifar10-onboarding",
        name=model_name,
        config={
            "experiment": "resnet_comparison",
            "model": model_name,
            "train_size": TRAIN_SIZE,
            "val_size": VAL_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED
        }
    )


    # ======================================
    # CSV
    # ======================================

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])


    best_val_acc = 0.0

    best_val_loss = 0.0

    best_epoch = 0


    start_time = time.time()


    # ======================================
    # Epoch
    # ======================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------
        # Train
        # ----------------------------------

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


        train_loss /= (
            train_total
        )


        train_acc = (
            train_correct
            / train_total
        )


        # ----------------------------------
        # Validation
        # ----------------------------------

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion
        )


        # ----------------------------------
        # Best Accuracy 저장
        # ----------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_val_loss = val_loss

            best_epoch = epoch


            torch.save(
                model.state_dict(),
                checkpoint_path
            )


            print(
                f"  -> Best model saved "
                f"(Val Acc: "
                f"{val_acc * 100:.2f}%)"
            )


        # ----------------------------------
        # CSV
        # ----------------------------------

        with open(
            csv_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            ])


        # ----------------------------------
        # W&B
        # ----------------------------------

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc
        })


        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_acc * 100:.2f}% | "
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Acc: "
            f"{val_acc * 100:.2f}%"
        )


    # ======================================
    # Best checkpoint 재로드
    # ======================================

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )


    reload_val_loss, reload_val_acc = (
        evaluate(
            model,
            val_loader,
            criterion
        )
    )


    elapsed_seconds = (
        time.time()
        - start_time
    )


    elapsed_minutes = (
        elapsed_seconds / 60
    )


    # ======================================
    # 결과
    # ======================================

    print(
        f"\n{model_name} finished"
    )


    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )


    print(
        f"Best Val Accuracy: "
        f"{reload_val_acc * 100:.2f}%"
    )


    print(
        f"Best Val Loss: "
        f"{reload_val_loss:.4f}"
    )


    print(
        f"Time: "
        f"{elapsed_minutes:.2f} min"
    )


    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )


    # ======================================
    # W&B Summary
    # ======================================

    wandb.summary[
        "best_epoch"
    ] = best_epoch


    wandb.summary[
        "best_val_acc"
    ] = reload_val_acc


    wandb.summary[
        "best_val_loss"
    ] = reload_val_loss


    wandb.summary[
        "runtime_minutes"
    ] = elapsed_minutes


    wandb.finish()


    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "val_acc": reload_val_acc,
        "val_loss": reload_val_loss,
        "time": elapsed_minutes
    }


# ==========================================
# Loader
# ==========================================

cifar_train_loader, cifar_val_loader = (
    get_loaders(
        cifar_transform
    )
)


imagenet_train_loader, imagenet_val_loader = (
    get_loaders(
        imagenet_transform
    )
)


# ==========================================
# 1. SmallCNN Scratch
# ==========================================

smallcnn = SmallCNN(
    width=32,
    dropout_p=0.0
)


result_smallcnn = train_model(
    model_name="compare_smallcnn_scratch",
    model=smallcnn,
    train_loader=cifar_train_loader,
    val_loader=cifar_val_loader
)


# ==========================================
# 2. ResNet18 Scratch
# ==========================================

resnet_scratch = models.resnet18(
    weights=None
)


resnet_scratch.fc = nn.Linear(
    resnet_scratch.fc.in_features,
    10
)


result_resnet_scratch = train_model(
    model_name="compare_resnet18_scratch",
    model=resnet_scratch,
    train_loader=cifar_train_loader,
    val_loader=cifar_val_loader
)


# ==========================================
# 3. ResNet18 ImageNet Pretrained
# ==========================================

weights = (
    models
    .ResNet18_Weights
    .IMAGENET1K_V1
)


resnet_pretrained = models.resnet18(
    weights=weights
)


resnet_pretrained.fc = nn.Linear(
    resnet_pretrained.fc.in_features,
    10
)


result_resnet_pretrained = train_model(
    model_name="compare_resnet18_pretrained",
    model=resnet_pretrained,
    train_loader=imagenet_train_loader,
    val_loader=imagenet_val_loader
)


# ==========================================
# 최종 결과
# ==========================================

print(
    "\n"
    "=========================================="
)

print(
    "Model Comparison Result"
)

print(
    "=========================================="
)


print(
    f"SmallCNN Scratch       | "
    f"Val Acc: "
    f"{result_smallcnn['val_acc'] * 100:.2f}%"
)


print(
    f"ResNet18 Scratch       | "
    f"Val Acc: "
    f"{result_resnet_scratch['val_acc'] * 100:.2f}%"
)


print(
    f"ResNet18 Pretrained    | "
    f"Val Acc: "
    f"{result_resnet_pretrained['val_acc'] * 100:.2f}%"
)


print(
    "=========================================="
)


# ==========================================
# Summary CSV
# ==========================================

summary_path = (
    "results/"
    "resnet_comparison_summary.csv"
)


with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)


    writer.writerow([
        "model",
        "best_epoch",
        "val_accuracy",
        "val_loss",
        "runtime_minutes"
    ])


    for result in [
        result_smallcnn,
        result_resnet_scratch,
        result_resnet_pretrained
    ]:

        writer.writerow([
            result["model"],
            result["best_epoch"],
            result["val_acc"],
            result["val_loss"],
            result["time"]
        ])


print(
    f"\nSaved: {summary_path}"
)