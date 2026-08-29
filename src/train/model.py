import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2)
    )


class SmallCNN(nn.Module):

    def __init__(self, width=32, dropout_p=0.0):
        super().__init__()

        self.block1 = conv_block(
            in_channels=3,
            out_channels=width
        )

        self.block2 = conv_block(
            in_channels=width,
            out_channels=width * 2
        )

        self.block3 = conv_block(
            in_channels=width * 2,
            out_channels=width * 4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(
            width * 4,
            10
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x)
        x = self.flatten(x)

        x = self.dropout(x)

        x = self.fc(x)

        return x


if __name__ == "__main__":

    for width in [16, 32, 64]:

        model = SmallCNN(width=width)

        x = torch.randn(2, 3, 32, 32)

        output = model(x)

        num_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(f"width={width}")
        print("output shape:", output.shape)
        print("parameters:", num_params)
        print()