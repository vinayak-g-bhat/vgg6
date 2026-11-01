from model import VGG
from activation import Activation

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from enum import Enum

from subpolicy import SubPolicy
from cifarpolicy import CIFAR10Policy
from cutout import Cutout

def eval(model,data,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    acc = 100. * correct / total
    return acc


def GetCifar10(batchsize):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  # CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10('./data', train=True, transform=trans_t, download=False)
    test_data = datasets.CIFAR10('./data', train=False, transform=trans, download=False)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2)
    return train_dataloader, test_dataloader

criterion = nn.CrossEntropyLoss()
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

def train_model(model,epochs,optimizer,train_loader,test_loader,wandb_runner):
   
    for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      for data, target in train_loader:
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          outputs = model(data)
          loss = criterion(outputs, target)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      train_acc = eval(model,train_loader)
      test_acc = eval(model,test_loader)
      train_loss = running_loss / len(train_loader)

      wandb_runner.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })
      
      print(f"Epoch {epoch} - Train_Loss: {train_loss:.4f} , Train_acc: {train_acc}, Test_acc : {test_acc}")
    wandb_runner.summary["final_val_acc"] = test_acc


def main():

    config = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "activation":'tanh',
        "optimizer": 'Adam',
        "batch_norm": True,
    }
    
    train_loader,test_loader = GetCifar10(config["batch_size"])

    run = wandb.init(
            project="pytorch-starter",
            config=config,
            mode="online",
    )
    model = VGG(num_classes=10, activation_fuction=Activation(config["activation"]),batch_norm=config["batch_norm"])
    print(model)
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    # optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9,nesterov=True, weight_decay=5e-4)

    
    run.watch(model, log="all", log_freq=100)
    train_model(model,config["epochs"],optimizer,train_loader,test_loader,wandb_runner=run)
    run.finish()

if __name__ == "__main__":
    main()

