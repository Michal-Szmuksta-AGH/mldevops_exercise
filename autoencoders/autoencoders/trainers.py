import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb


class FashionMNISTTrainer:
    def __init__(
        self, model, criterion, learning_rate=0.001, batch_size=128, num_epochs=15, data_path="./data", device=None
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.data_path = data_path
        self.learning_rate = learning_rate

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_path, train=True, transform=self.transform, download=True
        )
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_path, train=False, transform=self.transform, download=True
        )
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loss_history = []
        self.test_loss_history = []

        wandb.init(project="fashion-mnist", config={
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "device": str(self.device)
        })

    def calculate_loss(self, loader):
        raise NotImplementedError("Subclasses should implement this method")

    def train(self):
        raise NotImplementedError("Subclasses should implement this method")

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label="Train Loss")
        plt.plot(self.test_loss_history, label="Test Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        wandb.log({"Train Loss": self.train_loss_history, "Test Loss": self.test_loss_history})  # Log to wandb

    def save_model(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)
        wandb.save(path)


class ClassificationTrainer(FashionMNISTTrainer):
    def __init__(
        self, model, criterion, learning_rate=0.001, batch_size=128, num_epochs=15, data_path="./data", device=None
    ):
        super().__init__(model, criterion, learning_rate, batch_size, num_epochs, data_path, device)
        self.train_accuracy_history = []
        self.test_accuracy_history = []

    def calculate_loss_and_accuracy(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
        accuracy = 100 * correct / total
        return running_loss / len(loader), accuracy

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100 * correct / total
            test_loss, test_accuracy = self.calculate_loss_and_accuracy(self.test_loader)
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.test_accuracy_history.append(test_accuracy)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%"
            )
            wandb.log({
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy
            })

    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracy_history, label="Train Accuracy")
        plt.plot(self.test_accuracy_history, label="Test Accuracy")
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.show()
        wandb.log({"Train Accuracy": self.train_accuracy_history, "Test Accuracy": self.test_accuracy_history})


class AutoencoderTrainer(FashionMNISTTrainer):
    def calculate_loss(self, loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                running_loss += loss.item()
        return running_loss / len(loader)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, _ in self.train_loader:
                images = images.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            test_loss = self.calculate_loss(self.test_loader)
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            wandb.log({
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Test Loss": test_loss
            })
