from collections import Counter
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ==========================================
# Transform 만들기
# ==========================================

def get_transforms(augmentation=False):

    # Validation / Test에는 augmentation 사용 안 함
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD
        )
    ])

    # Baseline
    if augmentation is False or augmentation == "false" or augmentation == "none":

        train_transform = eval_transform

    # RandomCrop만 사용
    elif augmentation == "crop":

        train_transform = transforms.Compose([
            transforms.RandomCrop(
                32,
                padding=4
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR10_MEAN,
                CIFAR10_STD
            )
        ])

    # RandomCrop + HorizontalFlip
    elif augmentation == "crop_flip":

        train_transform = transforms.Compose([
            transforms.RandomCrop(
                32,
                padding=4
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR10_MEAN,
                CIFAR10_STD
            )
        ])

    else:
        raise ValueError(
            f"Unknown augmentation: {augmentation}"
        )

    return train_transform, eval_transform


# ==========================================
# Train / Val 인덱스 생성
# ==========================================

def make_split(seed):

    generator = torch.Generator().manual_seed(seed)

    index_dataset = list(range(50000))

    train_indices, val_indices = random_split(
        index_dataset,
        [45000, 5000],
        generator=generator
    )

    return (
        list(train_indices),
        list(val_indices)
    )


# ==========================================
# 작은 Train set 만들기
# ==========================================

def make_small_train_indices(
    train_indices,
    targets,
    train_size
):

    # 45,000장이면 기존 train 전체 사용
    if train_size == 45000:
        return train_indices

    if train_size > 45000:
        raise ValueError(
            "train_size는 45000을 넘을 수 없습니다."
        )

    if train_size % 10 != 0:
        raise ValueError(
            "현재 코드는 train_size가 10의 배수여야 합니다."
        )

    per_class = train_size // 10

    selected_indices = []
    class_counts = [0] * 10

    for idx in train_indices:

        label = targets[idx]

        if class_counts[label] < per_class:

            selected_indices.append(idx)

            class_counts[label] += 1

        if len(selected_indices) == train_size:
            break

    if len(selected_indices) != train_size:

        raise RuntimeError(
            f"{train_size}장을 만들지 못했습니다."
        )

    return selected_indices


# ==========================================
# DataLoader 생성
# ==========================================

def get_dataloaders(
    train_size=45000,
    batch_size=128,
    seed=42,
    augmentation=False,
    num_workers=4
):

    train_transform, eval_transform = get_transforms(
        augmentation
    )


    # Train용 Dataset
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )


    # Validation용 Dataset
    # 같은 CIFAR-10이지만 augmentation 없음
    val_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=eval_transform
    )


    # Test
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=eval_transform
    )


    # --------------------------------------
    # Train / Validation split
    # --------------------------------------

    train_indices, val_indices = make_split(
        seed
    )


    # --------------------------------------
    # 설정된 train_size만큼 추출
    # --------------------------------------

    selected_train_indices = make_small_train_indices(
        train_indices,
        train_dataset.targets,
        train_size
    )


    # --------------------------------------
    # Subset
    # --------------------------------------

    train_set = Subset(
        train_dataset,
        selected_train_indices
    )

    val_set = Subset(
        val_dataset,
        val_indices
    )


    # --------------------------------------
    # DataLoader
    # --------------------------------------

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )


# ==========================================
# data.py 직접 실행했을 때만 확인
# ==========================================

if __name__ == "__main__":

    SEED = 42
    BATCH_SIZE = 128


    # --------------------------------------
    # 원본 데이터 확인
    # --------------------------------------

    train_raw = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )


    print(
        "train_full:",
        len(train_raw)
    )

    print(
        "classes:",
        train_raw.classes
    )


    # --------------------------------------
    # 클래스별 개수
    # --------------------------------------

    counts = Counter(
        train_raw.targets
    )

    print("\nClass counts:")

    for class_idx, class_name in enumerate(
        train_raw.classes
    ):

        print(
            f"{class_name:10s}: "
            f"{counts[class_idx]}"
        )


    # --------------------------------------
    # 이미지 샘플 저장
    # --------------------------------------

    Path("images").mkdir(
        exist_ok=True
    )

    random.seed(SEED)

    class_indices = {
        class_idx: []
        for class_idx in range(10)
    }


    for idx, label in enumerate(
        train_raw.targets
    ):

        class_indices[label].append(idx)


    for class_idx in class_indices:

        random.shuffle(
            class_indices[class_idx]
        )


    fig, axes = plt.subplots(
        5,
        10,
        figsize=(15, 8)
    )


    for class_idx in range(10):

        selected = (
            class_indices[class_idx][:5]
        )

        for row, image_idx in enumerate(
            selected
        ):

            image, label = train_raw[
                image_idx
            ]

            ax = axes[
                row,
                class_idx
            ]

            ax.imshow(image)

            ax.set_title(
                train_raw.classes[label],
                fontsize=8
            )

            ax.axis("off")


    plt.tight_layout()

    plt.savefig(
        "images/cifar10_samples.png",
        dpi=150
    )

    plt.close()

    print(
        "\nSaved: "
        "images/cifar10_samples.png"
    )


    # --------------------------------------
    # Baseline DataLoader 확인
    # --------------------------------------

    train_loader, val_loader, test_loader = (
        get_dataloaders(
            train_size=45000,
            batch_size=BATCH_SIZE,
            seed=SEED,
            augmentation=False
        )
    )


    print("\nDataset split")

    print(
        "train:",
        len(train_loader.dataset)
    )

    print(
        "val:",
        len(val_loader.dataset)
    )

    print(
        "test:",
        len(test_loader.dataset)
    )


    x, y = next(
        iter(train_loader)
    )


    print("\nBatch check")

    print(
        "x.shape:",
        x.shape
    )

    print(
        "x.dtype:",
        x.dtype
    )

    print(
        "x.min():",
        x.min().item()
    )

    print(
        "x.max():",
        x.max().item()
    )

    print(
        "y[:10]:",
        y[:10]
    )


    # --------------------------------------
    # 500장 설정 확인
    # --------------------------------------

    small_loader, _, _ = get_dataloaders(
        train_size=500,
        batch_size=BATCH_SIZE,
        seed=SEED,
        augmentation=False
    )

    small_labels = []

    for index in small_loader.dataset.indices:

        small_labels.append(
            train_raw.targets[index]
        )

    small_counts = Counter(
        small_labels
    )


    print("\nSmall train set")

    print(
        "train_small:",
        len(small_loader.dataset)
    )

    print(
        "class counts:",
        [
            small_counts[i]
            for i in range(10)
        ]
    )