import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


class Encoder(nn.Module):
    def __init__(self, feature_size=100):
        super().__init__()
        self.feature_size = feature_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=self.feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


class Decoder(nn.Module):
    def __init__(self, feature_size=100):
        super().__init__()
        self.feature_size = feature_size

        self.fc2 = nn.Linear(in_features=self.feature_size, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(in_features=600, out_features=64 * 6 * 6)
        self.unflatten = nn.Unflatten(1, (64, 6, 6))

        self.layer2 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
        )

        self.layer1 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, output_padding=1),
        )

    def forward(self, x):
        out = self.fc2(x)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.unflatten(out)
        out = self.layer2(out)
        out = self.layer1(out)

        return out


class Autoencoder(nn.Module):
    def __init__(self, feature_size=100):
        super().__init__()
        self.encoder = Encoder(feature_size)
        self.decoder = Decoder(feature_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
