default_params = {
    "batch_dim": 64,
    "epoch_count": 10,
    "device_string": "cuda",
    "output_root": "outputs",
    "seed": 21,
    "data_root": "data",
    "dataset": "CIFAR10",
    "normals": "True",
    "learning_rate": 1e-3,
    "criterion": "CrossEntropyLoss",
    "optimizer": "Adam"
}

def defaults(dictionary, dictionary_defaults):
    for key, value in dictionary_defaults.items():
        if key not in dictionary:
            dictionary[key] = value
        else:
            if isinstance(value, dict) and isinstance(dictionary[key], dict):
                dictionary[key] = defaults(dictionary[key], value)
            elif isinstance(value, dict) or isinstance(dictionary[key], dict):
                raise ValueError("Given dictionaries have incompatible structure")
    return dictionary


def image_dataset(dataset_name, dataset_root, normals):
    import torchvision
    from torchvision import datasets, transforms

    if normals == "true":
        dataset_transforms = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        dataset_transforms = transforms.Compose([transforms.ToTensor()])


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



    return dataset_train, dataset_test

def loss_function(type):
    import torch
    if type == "L1Loss":
        return torch.nn.L1Loss()
    elif type == "MSELoss":
        return torch.nn.MSELoss()
    elif type == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Loss not implemented or naming error")

def optimizer_function(type, model, lr):
    import torch.optim as optim
    if type == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif type == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not implemented or naming error")



def experiment(params):
    import os 
    import json
    import math 
    import time
    from contextlib import nullcontext
    import torch
    import cnn
    from cnn import BasicNet, MNISTNet
    from tqdm import tqdm
    from time import sleep
    import wandb
    
    batch_dim = params["batch_dim"]
    epoch_count = params["epoch_count"]
    device_string = params["device_string"]
    output_root = params["output_root"]
    seed = params["seed"]
    data_root = params["data_root"]
    dataset = params["dataset"]
    normals = params["normals"]
    learning_rate = params["learning_rate"]
    loss = params["criterion"]
    optimizer_string = params["optimizer"]

    device = torch.device(device_string)
    torch.manual_seed(seed)

    dataset_train, dataset_test = image_dataset(dataset, data_root, normals=normals)

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
    if dataset == "MNIST":
        model = MNISTNet(len(dataset_train.classes), in_channels=1).to(device)
    else:
        model = BasicNet(len(dataset_train.classes), in_channels=3).to(device)

    print(model)
    criterion = loss_function(loss)
    optimizer = optimizer_function(optimizer_string, model, lr=learning_rate)
    # wandb.init(project="Torch Points 3D", name = log_name, entity="qinjerem")

    
    for epoch in range(epoch_count):
        correct = 0
        total_loss = 0
        total = 0 
        with tqdm(loader_train, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tepoch:
            for (images, labels) in tepoch:

                tepoch.set_description('Epoch [{}/{}]'.format(epoch+1, epoch_count))
                images = images.to(device)
                labels = labels.to(device)
                
                #Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                    
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                tepoch.set_postfix(loss = loss.item(), accuracy = 100 * correct / total )
                sleep(0.1)
        # print ('Epoch [{}/{}], Accuracy: {:.3f} , Loss: {:.3f}' 
        #             .format(epoch+1, epoch_count,100 * correct / total, loss.item()))

            wandb.log({"loss": total_loss / total,
                        "accuracy": 100*correct / total,
                        "inputs": wandb.Image(images)})
        
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for images, labels in loader_test:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test set: Average loss: {loss / total:.3f}, Accuracy: {correct / total :.3f}")


    return

if __name__ == "__main__":
    import json
    import argparse
    import wandb

    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", "-p", type=str, help="JSON params file")
    parser.add_argument("--direct", "-d", type=str, help="JSON state string")
    
    arguments = parser.parse_args()
    
    if arguments.direct is not None:
        params = json.loads(arguments.direct)
    elif arguments.params is not None:
        with open(arguments.params) as file:
            params = json.load(file)
    else:
        params = {}

    params = defaults(params, default_params)
    log_name = params["dataset"] + "-" + current_time
    wandb.init(project="CNN-Magic", name = log_name, entity="qinjerem", config=params)

    experiment(params)

    wandb.finish()
    
    

