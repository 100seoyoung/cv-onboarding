import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split
)

from torchvision import datasets, transforms

from model import SmallCNN


# ==========================================
# 기본 설정
# ==========================================

SEED = 42

TRAIN_SIZE = 500

BATCH_SIZE = 128

WIDTH = 32

EPOCHS = 30

LR = 0.05

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
# Seed 고정
# ==========================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================
# CIFAR-10 Normalize
# ==========================================

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616
)


eval_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ==========================================
# 3가지 view
#
# 원본
# Horizontal Flip
# Random Crop + Flip
# ==========================================

original_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(
        p=1.0
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


crop_flip_transform = transforms.Compose([
    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ==========================================
# CIFAR-10 원본 데이터
# ==========================================

train_raw = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)


test_set = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=eval_transform
)


test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ==========================================
# 기존 45,000 / 5,000 split과 같은 방식
# ==========================================

generator = torch.Generator().manual_seed(
    SEED
)

all_indices = torch.randperm(
    len(train_raw),
    generator=generator
).tolist()


train_pool_indices = all_indices[:45000]


# ==========================================
# Train pool에서
# 클래스당 50장씩 총 500장 선택
# ==========================================

class_counts = [0] * 10

selected_indices = []


for idx in train_pool_indices:

    label = train_raw.targets[idx]

    if class_counts[label] < 50:

        selected_indices.append(idx)

        class_counts[label] += 1

    if len(selected_indices) == 500:

        break


print("\nSelected data")

print(
    "total:",
    len(selected_indices)
)

print(
    "class counts:",
    class_counts
)


# ==========================================
# 같은 500장의 3가지 view를 미리 생성
# ==========================================

set_seed(SEED)

view_bank = {}


for idx in selected_indices:

    image, label = train_raw[idx]

    views = [
        original_transform(image),

        flip_transform(image),

        crop_flip_transform(image)
    ]

    view_bank[idx] = {
        "views": views,
        "label": label
    }


# ==========================================
# 올바른 방법 A
#
# 500장
# → 먼저 400 / 100 split
# → Train 400장만 3배 augmentation
# ==========================================

def make_correct_split():

    class_indices = {
        i: []
        for i in range(10)
    }


    for idx in selected_indices:

        label = train_raw.targets[idx]

        class_indices[label].append(idx)


    rng = random.Random(SEED)


    train_indices = []

    val_indices = []


    for class_id in range(10):

        indices = class_indices[
            class_id
        ]

        rng.shuffle(indices)

        # 클래스당 40장 train
        train_indices.extend(
            indices[:40]
        )

        # 클래스당 10장 validation
        val_indices.extend(
            indices[40:50]
        )


    # --------------------------------------
    # Train 400장 → 3배 → 1,200장
    # --------------------------------------

    train_x = []

    train_y = []


    for idx in train_indices:

        item = view_bank[idx]

        for view in item["views"]:

            train_x.append(view)

            train_y.append(
                item["label"]
            )


    train_x = torch.stack(
        train_x
    )

    train_y = torch.tensor(
        train_y,
        dtype=torch.long
    )


    # --------------------------------------
    # Validation 100장은 augmentation 없음
    # --------------------------------------

    val_x = []

    val_y = []


    for idx in val_indices:

        image, label = train_raw[idx]

        image = eval_transform(
            image
        )

        val_x.append(image)

        val_y.append(label)


    val_x = torch.stack(
        val_x
    )

    val_y = torch.tensor(
        val_y,
        dtype=torch.long
    )


    train_dataset = TensorDataset(
        train_x,
        train_y
    )

    val_dataset = TensorDataset(
        val_x,
        val_y
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    print(
        "\n[A] Correct Split"
    )

    print(
        "Original Train:",
        len(train_indices)
    )

    print(
        "Augmented Train:",
        len(train_dataset)
    )

    print(
        "Validation:",
        len(val_dataset)
    )

    print(
        "Train-Val original overlap: 0"
    )


    return (
        train_loader,
        val_loader
    )


# ==========================================
# 잘못된 방법 B
#
# 500장
# → 먼저 3배 augmentation
# → 1,500장
# → 그 후 1,200 / 300 split
# ==========================================

def make_wrong_split():

    all_x = []

    all_y = []

    origin_ids = []


    for idx in selected_indices:

        item = view_bank[idx]

        for view in item["views"]:

            all_x.append(view)

            all_y.append(
                item["label"]
            )

            # 어떤 원본 이미지에서
            # 만들어졌는지 기록
            origin_ids.append(idx)


    all_x = torch.stack(
        all_x
    )

    all_y = torch.tensor(
        all_y,
        dtype=torch.long
    )


    full_dataset = TensorDataset(
        all_x,
        all_y
    )


    split_generator = (
        torch.Generator()
        .manual_seed(SEED)
    )


    train_subset, val_subset = (
        random_split(
            full_dataset,
            [1200, 300],
            generator=split_generator
        )
    )


    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # --------------------------------------
    # 실제로 Train/Val에 같은 원본이
    # 얼마나 섞였는지 확인
    # --------------------------------------

    train_origins = {
        origin_ids[i]
        for i in train_subset.indices
    }


    val_origins = {
        origin_ids[i]
        for i in val_subset.indices
    }


    overlap = (
        train_origins
        & val_origins
    )


    print(
        "\n[B] Wrong Split"
    )

    print(
        "Augmented total:",
        len(full_dataset)
    )

    print(
        "Train:",
        len(train_subset)
    )

    print(
        "Validation:",
        len(val_subset)
    )

    print(
        "Overlapping original images:",
        len(overlap)
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
# 한 실험 학습 함수
# ==========================================

def run_experiment(
    experiment_name,
    method_name,
    train_loader,
    val_loader
):

    # A/B 모두 같은 초기 seed 사용
    set_seed(SEED)


    model = SmallCNN(
        width=WIDTH,
        dropout_p=0.0
    ).to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )


    csv_path = (
        f"results/"
        f"{experiment_name}.csv"
    )


    checkpoint_path = (
        f"checkpoints/"
        f"{experiment_name}_best.pt"
    )


    # --------------------------------------
    # W&B
    # --------------------------------------

    wandb.init(
        project="cifar10-onboarding",
        name=experiment_name,
        config={
            "experiment": "data_leakage",
            "method": method_name,
            "seed": SEED,
            "train_original_size": TRAIN_SIZE,
            "width": WIDTH,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS
        }
    )


    # --------------------------------------
    # CSV
    # --------------------------------------

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

    best_epoch = 0


    start_time = time.time()


    # ======================================
    # Training
    # ======================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

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
        # Best Validation Accuracy 모델 저장
        # ----------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_epoch = epoch


            torch.save(
                model.state_dict(),
                checkpoint_path
            )


        # ----------------------------------
        # CSV 저장
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
            f"{experiment_name} | "
            f"Epoch {epoch:03d} | "
            f"Train Acc: "
            f"{train_acc * 100:.2f}% | "
            f"Val Acc: "
            f"{val_acc * 100:.2f}%"
        )


    # ======================================
    # Best checkpoint 불러오기
    # ======================================

    best_model = SmallCNN(
        width=WIDTH,
        dropout_p=0.0
    ).to(device)


    best_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )


    # ======================================
    # Best model Validation
    # ======================================

    best_val_loss, best_val_acc_reload = (
        evaluate(
            best_model,
            val_loader,
            criterion
        )
    )


    # ======================================
    # 진짜 CIFAR-10 Test
    # ======================================

    test_loss, test_acc = evaluate(
        best_model,
        test_loader,
        criterion
    )


    elapsed_seconds = (
        time.time()
        - start_time
    )


    elapsed_minutes = (
        elapsed_seconds / 60
    )


    print(
        f"\n{experiment_name} finished"
    )


    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )


    print(
        f"Best Val Accuracy: "
        f"{best_val_acc_reload * 100:.2f}%"
    )


    print(
        f"Test Accuracy: "
        f"{test_acc * 100:.2f}%"
    )


    print(
        f"Time: "
        f"{elapsed_minutes:.2f} min"
    )


    # --------------------------------------
    # W&B Summary
    # --------------------------------------

    wandb.summary[
        "best_epoch"
    ] = best_epoch


    wandb.summary[
        "best_val_acc"
    ] = best_val_acc_reload


    wandb.summary[
        "test_acc"
    ] = test_acc


    wandb.summary[
        "runtime_minutes"
    ] = elapsed_minutes


    wandb.finish()


    return {
        "name": experiment_name,
        "best_epoch": best_epoch,
        "val_acc": best_val_acc_reload,
        "test_acc": test_acc,
        "time": elapsed_minutes
    }


# ==========================================
# A. 올바른 Split
# ==========================================

correct_train_loader, correct_val_loader = (
    make_correct_split()
)


result_a = run_experiment(
    experiment_name="leakage_A_correct_s42",
    method_name="split_before_augmentation",
    train_loader=correct_train_loader,
    val_loader=correct_val_loader
)


# ==========================================
# B. 잘못된 Split
# ==========================================

wrong_train_loader, wrong_val_loader = (
    make_wrong_split()
)


result_b = run_experiment(
    experiment_name="leakage_B_wrong_s42",
    method_name="augmentation_before_split",
    train_loader=wrong_train_loader,
    val_loader=wrong_val_loader
)


# ==========================================
# 최종 비교
# ==========================================

print(
    "\n"
    "=========================================="
)

print(
    "Data Leakage Experiment Result"
)

print(
    "=========================================="
)


print(
    f"A Correct | "
    f"Val: {result_a['val_acc'] * 100:.2f}% | "
    f"Test: {result_a['test_acc'] * 100:.2f}%"
)


print(
    f"B Wrong   | "
    f"Val: {result_b['val_acc'] * 100:.2f}% | "
    f"Test: {result_b['test_acc'] * 100:.2f}%"
)


print(
    "=========================================="
)


# ==========================================
# 요약 CSV
# ==========================================

summary_path = (
    "results/leakage_summary.csv"
)


with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "method",
        "best_epoch",
        "val_accuracy",
        "test_accuracy",
        "runtime_minutes"
    ])


    writer.writerow([
        "correct_split",
        result_a["best_epoch"],
        result_a["val_acc"],
        result_a["test_acc"],
        result_a["time"]
    ])


    writer.writerow([
        "wrong_split",
        result_b["best_epoch"],
        result_b["val_acc"],
        result_b["test_acc"],
        result_b["time"]
    ])


print(
    f"\nSaved: {summary_path}"
)