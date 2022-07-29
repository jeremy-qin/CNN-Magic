import os
import time
from unittest import loader
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import cnn 
from cnn import BasicNet
from comet_ml import Experiment

experiment = Experiment(
    api_key="yakDHEHhfgB8W7OcNVQbDh10M",
    project_name="cnn-magic",
    workspace="jeremy-qin",
)

batch_dim = 64
epoch_count = 10

device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

torch.manual_seed(0)

dataset_root = "data"
dataset_name = "CIFAR10"

dataset_transforms = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if dataset_name == "MNIST":
    dataset_train = datasets.MNIST(
        root = dataset_root,
        train = True,
        download = True,
        transform = dataset_transforms,
    )
    dataset_test = datasets.MNIST(
        root = dataset_root,
        train = False,
        transform = dataset_transforms,
    )

elif dataset_name == "CIFAR10":
    dataset_train = datasets.CIFAR10(
        root = dataset_root,
        train = True,
        download = True,
        transform = dataset_transforms,
    )

    dataset_test = datasets.CIFAR10(
        root = dataset_root,
        train = False,
        transform = dataset_transforms,
    )

elif dataset_name == "CIFAR100":
    dataset_train = datasets.CIFAR100(
        root = dataset_root,
        train = True,
        download = True,
        transform = dataset_transforms,
    )

    dataset_test = datasets.CIFAR100(
        root = dataset_root,
        train = False,
        transform = dataset_transforms,
    )

loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=batch_dim,
    shuffle=True,
    pin_memory=(device_string == "cuda"),
)

loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=batch_dim,
    shuffle=False,
    pin_memory=(device_string == "cuda"),
)

model = BasicNet(len(dataset_train.classes)).to(device)
# model = torchvision.models.resnet34(pretrained=True)
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, len(dataset_train.classes))
# model = model.to(device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
total_step = len(loader_train)

experiment.log_parameter("batch_dim", batch_dim)
experiment.log_parameter("learning_rate", learning_rate)
experiment.log_parameter("criterion", criterion)
experiment.log_parameter("optimizer", optimizer)
experiment.log_parameter("steps", total_step)

def train():
    with experiment.train():
        for epoch in range(epoch_count):
            correct = 0
            total = 0 
            for i, (images, labels) in enumerate(loader_train):
                
                images = images.to(device)
                labels = labels.to(device)
                
                #Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                    
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                        
                if (i+1) % 391 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {} , Loss: {:.4f}' 
                                .format(epoch+1, epoch_count, i+1, total_step,100 * correct / total, loss.item()))
            experiment.log_metric("accuracy", 100 * correct / total, epoch = epoch+1)
            experiment.log_metric("loss", loss.item(), epoch=epoch+1)

            with experiment.test():
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in loader_test:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
                    experiment.log_metric("test_accuracy", 100 * correct / total, epoch=epoch+1)

def test():
    with experiment.test():
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in loader_test:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
            experiment.log_metric("test_accuracy", 100 * correct / total)

def main():
    train()

main()
	 