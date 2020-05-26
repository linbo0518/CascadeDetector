import torch
from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        x = self.flatten(x)
        return x


class PNet(nn.Module):
    """PNet in MTCNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(10),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU(32),
        )
        self.face_cls = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        self.bbox_reg = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.features(x)
        face_cls = self.face_cls(x)
        bbox_reg = self.bbox_reg(x)
        return face_cls, bbox_reg


class RNet(nn.Module):
    """RNet in MTCNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(28),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(48),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU(64),
            Flatten(),
            nn.Linear(576, 128),
            nn.PReLU(128),
        )
        self.face_cls = nn.Linear(128, 2)
        self.bbox_reg = nn.Linear(128, 4)

    def forward(self, x):
        x = self.features(x)
        face_cls = self.face_cls(x)
        bbox_reg = self.bbox_reg(x)
        return face_cls, bbox_reg


class ONet(nn.Module):
    """ONet in MTCNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU(128),
            Flatten(),
            nn.Linear(1152, 256),
            nn.Dropout(0.25),
            nn.PReLU(256),
        )
        self.face_cls = nn.Linear(256, 2)
        self.bbox_reg = nn.Linear(256, 4)
        self.point_reg = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        face_cls = self.face_cls(x)
        bbox_reg = self.bbox_reg(x)
        point_reg = self.point_reg(x)
        return face_cls, bbox_reg, point_reg