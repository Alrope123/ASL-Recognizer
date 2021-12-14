import os
import pandas as pd
import torch
from torchvision import datasets
from torchvision import transforms
import multiprocessing

RATIO = 0.8

def load_alphabet(dir, batch_size, train_transform=None, dev_transform=None):
    train_data = datasets.ImageFolder(root=os.path.join(dir, "Alphabet", "asl_alphabet_train"), transform=train_transform)
    train_data =  torch.utils.data.Subset(train_data, range(0, int(len(train_data) * RATIO)))
    test_data = datasets.ImageFolder(root=os.path.join(dir, "Alphabet", "asl_alphabet_train"), transform=dev_transform)
    test_data =  torch.utils.data.Subset(test_data, range(int(len(train_data) * RATIO), len(test_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_asl(dir, batch_size, test_transform=None):
    test_data = datasets.ImageFolder(root=os.path.join(dir, "asl"), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader


class ASLMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        df = pd.read_csv(root)
        self.images = torch.tensor(df.iloc[:, 1:].values).view(-1, 1, 28, 28)
        self.labels = torch.tensor(df.iloc[:, 0].values)
        
    def __len__(self):
        return self.labels.shape[0]
      
    def __getitem__(self, idx):
        data = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        return (data, label)

d = ASLMNISTDataset("data/MNIST/sign_mnist_test.csv")

def load_mnist(dir, batch_size, train_transform=None, test_transform=None):
    train_data = ASLMNISTDataset(root=os.path.join(dir, "MNIST", "sign_mnist_train.csv"), transform=train_transform)
    test_data = ASLMNISTDataset(root=os.path.join(dir, "MNIST", "sign_mnist_test.csv"), transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


