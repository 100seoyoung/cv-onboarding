import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SmallCNN


# ==========================================
# 설정
# ==========================================

CHECKPOINT_PATH = (
    "checkpoints/"
    "width64_n45000_lr0.05_s42_best_acc.pt"
)

BATCH_SIZE = 128

WIDTH = 64

VAL_ACCURACY = 88.36


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device:", device)


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


transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ==========================================
# Test Dataset
# ==========================================

test_set = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


print(
    "test size:",
    len(test_set)
)


# ==========================================
# Model
# ==========================================

model = SmallCNN(
    width=WIDTH,
    dropout_p=0.0
).to(device)


model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )
)


model.eval()


criterion = nn.CrossEntropyLoss()


# ==========================================
# Test
# ==========================================

test_loss = 0.0

test_correct = 0

test_total = 0


with torch.no_grad():

    for x, y in test_loader:

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


        test_loss += (
            loss.item()
            * x.size(0)
        )


        predicted = outputs.argmax(
            dim=1
        )


        test_correct += (
            predicted == y
        ).sum().item()


        test_total += y.size(0)


test_loss /= test_total

test_accuracy = (
    test_correct
    / test_total
    * 100
)


difference = (
    VAL_ACCURACY
    - test_accuracy
)


# ==========================================
# 결과
# ==========================================

print(
    "\n=========================================="
)

print(
    "Final Test Result"
)

print(
    "=========================================="
)

print(
    f"Validation Accuracy: "
    f"{VAL_ACCURACY:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"Difference: "
    f"{difference:.2f}%p"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    "=========================================="
)