import string
import torch
import torch.optim as optim
import util
import data
import argparse
from model import ASLImagenetNet
from run import train, test


def main(dir):
    BATCH_SIZE = 256
    EPOCHS = 30
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    SEED = 0
    PRINT_INTERVAL = 100
    WEIGHT_DECAY = 0.0005

    EXPERIMENT_VERSION = "0.0" # increment this to start a new experiment
    LOG_PATH = 'logs/' + EXPERIMENT_VERSION + '/'

    # Now the actual training code
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    class_names = list(string.ascii_uppercase)

    train_transform, test_trasnform = util.get_transforms("Alphabet")

    train_loader, test_loader = data.load_alphabet(dir, BATCH_SIZE, train_transform=train_transform, test_transform=test_trasnform)
    print(len(train_loader))
    print(len(test_loader))

    model = ASLImagenetNet().to(device)
    print("training a network with {} parameters...".format(len(model.parameters())))
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    start_epoch = model.load_last_model(LOG_PATH)

    train_losses, test_losses, test_accuracies = util.read_log(LOG_PATH + 'log.pkl', ([], [], []))
    test_loss, test_accuracy, correct_images, correct_val, error_images, predicted_val, gt_val = test(model, device, test_loader, True)

    if start_epoch == 0:
        correct_images = util.to_scaled_uint8(correct_images.transpose(0, 2, 3, 1))
        error_images = util.to_scaled_uint8(error_images.transpose(0, 2, 3, 1))
        util.show_images(correct_images, ['correct: %s' % class_names[aa] for aa in correct_val])
        util.show_images(error_images, ['pred: %s, actual: %s' % (class_names[aa], class_names[bb]) for aa, bb in zip(predicted_val, gt_val)])

        test_losses.append((start_epoch, test_loss))
        test_accuracies.append((start_epoch, test_accuracy))

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        test_loss, test_accuracy, correct_images, correct_val, error_images, predicted_val, gt_val = test(model, device, test_loader, True)
        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        test_accuracies.append((epoch, test_accuracy))
        util.write_log(LOG_PATH + '.pkl', (train_losses, test_losses, test_accuracies))
        model.save_best_model(test_accuracy, LOG_PATH + '%03d.pt' % epoch)

    model.save_model(LOG_PATH + '%03d.pt' % epoch, 0)
    ep, val = zip(*train_losses)
    util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
    ep, val = zip(*test_losses)
    util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
    ep, val = zip(*test_accuracies)
    util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')
    correct_images = util.to_scaled_uint8(correct_images.transpose(0, 2, 3, 1))
    error_images = util.to_scaled_uint8(error_images.transpose(0, 2, 3, 1))
    util.show_images(correct_images, ['correct: %s' % class_names[aa] for aa in correct_val])
    util.show_images(error_images, ['pred: %s, actual: %s' % (class_names[aa], class_names[bb]) for aa, bb in zip(predicted_val, gt_val)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    main(args.data_dir)