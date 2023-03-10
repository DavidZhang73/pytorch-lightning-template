import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = SimpleNet(num_classes=10)
    for name, param in model.named_parameters():
        print(name, param.shape)
    data = torch.randn(10, 1, 28, 28)
    print(model(data).shape)
