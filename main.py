import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import util
import data
import argparse
from model import ASLImagenetNet, Darknet64, Resnet, Resnext
from run import train, test


def main(dir):
    BATCH_SIZE = 256
    EPOCHS = 20
    # LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    SEED = 0
    WEIGHT_DECAY = 0.0005

    EXPERIMENT_VERSION = "0.4" # increment this to start a new experiment
    LOG_PATH = 'logs/' + EXPERIMENT_VERSION + '/'

    # Now the actual training code
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    print(torch.cuda.get_device_name(0))

    
    print("Loading the data...")
    # train_transform, test_trasnform = util.get_transforms("Alphabet")
    # train_loader, test_loader = data.load_alphabet(dir, BATCH_SIZE, train_transform=train_transform, test_transform=test_trasnform)
    train_transform, _ = util.get_transforms("Alphabet")
    _, test_trasnform = util.get_transforms("asl")
    train_loader, _ = data.load_alphabet(dir, BATCH_SIZE, train_transform=train_transform, test_transform=test_trasnform)
    _, test_loader = data.load_asl(dir, BATCH_SIZE, train_transform=train_transform, test_transform=test_trasnform)
    print(len(train_loader))
    print(len(test_loader))

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print(images.size())

    # def imshow(img):
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # # show images
    # imshow(torchvision.utils.make_grid(images))

    model = Resnext().to(device)
    print("training a network with {} parameters...".format(sum([1 for _ in model.parameters()])))
    start_epoch = util.load_last_model(model, LOG_PATH)
    
    print("Trying to find the old logs...")
    train_losses, test_losses, test_accuracies = util.read_log(LOG_PATH + 'log.pkl', ([], [], []))
    
    test_loss, test_accuracy = test(model, device, test_loader)
    if start_epoch == 0:
        test_losses.append((start_epoch, test_loss))
        test_accuracies.append((start_epoch, test_accuracy))

    print("Starting training...")
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        if epoch <= 7:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        elif epoch <= 13:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        train_loss = train(model, device, train_loader, optimizer, epoch, int(max(1, len(train_loader) / 10)))
        test_loss, test_accuracy = test(model, device, test_loader)
        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        test_accuracies.append((epoch, test_accuracy))
        util.write_log(LOG_PATH + '.pkl', (train_losses, test_losses, test_accuracies))
        util.save_best_model(model, test_accuracy, LOG_PATH + '%03d.pt' % epoch)

    if start_epoch != EPOCHS:
        util.save_model(model, LOG_PATH + '%03d.pt' % epoch, 0)
    ep, val = zip(*train_losses)
    util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
    ep, val = zip(*test_losses)
    util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
    ep, val = zip(*test_accuracies)
    util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    main(args.data_dir)