import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import util
import data
import argparse
from model import ASLImagenetNet, Resnet, Resnext
from run import train, test

BATCH_SIZE = 256
EPOCHS = 20
MOMENTUM = 0.9
SEED = 0
WEIGHT_DECAY = 0.0005
EXPERIMENT_VERSION = "0.4"
LOG_PATH = 'logs/' + EXPERIMENT_VERSION + '/'

def main(dir):
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    print(torch.cuda.get_device_name(0))

    
    print("Loading the data...")
    train_transform, dev_trasnform = util.get_transforms("Alphabet")
    train_loader, dev_loader = data.load_alphabet(dir, BATCH_SIZE, train_transform=train_transform, dev_transform=dev_trasnform)
    test_trasnform = util.get_transforms("asl")
    test_loader = data.load_asl(dir, BATCH_SIZE, test_transform=test_trasnform)

    model = Resnext().to(device)
    print("training a network with {} parameters...".format(sum([1 for _ in model.parameters()])))
    print("Loading the saved model")
    start_epoch = util.load_last_model(model, LOG_PATH)

    print("Starting training...")
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        if epoch <= 7:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        elif epoch <= 13:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        train_loss = train(model, device, train_loader, optimizer, epoch, int(max(1, len(train_loader) / 10)))
        dev_loss, dev_accuracy = test(model, device, dev_loader)
        util.save_best_model(model, dev_accuracy, LOG_PATH + '%03d.pt' % epoch)
    
    print("Finished training, beginning evaluating...")
    dev_loss, dev_accuracy = test(model, device, dev_loader)
    test_loss, test_accuracy = test(model, device, test_loader)
    print("Final dev loss: {0:.3f}, dev accuracy: {1:.3f}".format(dev_loss, dev_accuracy))
    print("Final test loss: {0:.3f}, test accuracy: {1:.3f}".format(test_loss, test_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    main(args.data_dir)