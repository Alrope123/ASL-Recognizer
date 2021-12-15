# ASL Recognizer

https://user-images.githubusercontent.com/46021285/146103551-c63340f8-c7b9-4d61-b068-bee8dbb09f39.mp4

## Abstract and Problem statement
- This project aims to read in images of American Sign Language (ASL) alphabet and classify them into the corresponding letter.
- This project can be extended do direct translation from video of doing a sequence of ASL into natural English if time permits in the future.
- The motive for doing this is to try exploring an easy and fast way to communicate with people who has speaking or hearing disability.

## Related work
- The project is partially inspired by the MNIST image dataset. If images for handwritten language could be recognized (translated), for people with disability then, direct recognizing (translating) images of sign language would be a much faster way if we are able to.
- Used a subset of a dataset called ASL Alphabet, which contains 3000 images for each alphabet letter. (https://github.com/grassknoted/Unvoiced)
- Used a subset of a dataset called American Sign Language Dataset, which contains 70 images for each alphabet letter. (https://www.kaggle.com/ayuraj/asl-dataset)

## Methodology
- Data Argumentation: performed lots of random crops, random rotations, random flips, etc. to try avoiding overfitting
- Network: Used three different network: 1. ASLNet: self-constructed Darknet –like network but with 4 residual units, also includes batch normalization. 2. pretrained ResNet. 3. pretrained ResNext.   
- Training: Trained (finetuned) the model with cross entropy loss for 20 epochs with annealing learning rate from 0.01 to 0.0001

## Experiments/evaluation
- Split the ASL Alphabet dataset into train and dev split. Trained on the train split and evaluate the performance on the dev split.
- Used the entire American Sign Language Dataset as a test set because we really cares about the generality of our trained network.

## Results
| Network | Dev Accuracy (ASL Alphabet) | Test Accuracy (American Sign Language Dataset) |
|---------|-----------------------------|------------------------------------------------|
| ASLNet  | 43%                         | 0%                                             |
| Resnet  | 46%                         | 6%                                             |
| Resnext | 46%                         | 20%                                            |

## Demo
Some recognization results running `webcam.py`:
![Semantic description of image](/demos/img0.png "demo 0") ![Semantic description of image](/demos/img1.png "demo 1")
