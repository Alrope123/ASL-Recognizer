import string

import cv2
import torchvision
import torch
from run import inference
from model import Resnext
import util
from main import LOG_PATH


def predict(frame):
    img = torchvision.transforms.ToTensor()(frame)
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    # print(img.shape)
    p, pred = inference(model, device, img)
    text = "Letter: {0:}    Probability: {1:.1f}%".format(list(string.ascii_uppercase)[pred], torch.exp(-p).item() * 100)
    print("=======================================")
    print(text)
    cv2.putText(frame, text, (50, 50), 0, 0.8, (0, 0, 0), 2)
    cv2.imshow("camera", frame)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Resnext().to(device)
    util.load_last_model(model, LOG_PATH)

    cv2.namedWindow("camera")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    stop = False
    while rval:
        cv2.imshow("camera", frame)
        if not stop:
            rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        elif key == 32:
            if stop:
                stop = False
            else:
                stop = True
                predict(frame)
            

    vc.release()
    cv2.destroyWindow("camera")