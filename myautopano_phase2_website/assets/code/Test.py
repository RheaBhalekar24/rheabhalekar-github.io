#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import json
from Network.Network import HomographyModel, Supervised, SupervisedNew
from torch.nn import functional as F

# Don't generate pyc codes
sys.dont_write_bytecode = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device IS: ", device)

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


def TestOperation(BasePath, ModelPath):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = Supervised().to(device)
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )
    pa_path = os.path.join(BasePath, 'Val/P_A')
    image_count = len(glob.glob(os.path.join(pa_path, '*.jpg')))

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        total_loss = 0
        for i in tqdm(range(1, image_count + 1)):
            # Load images
            P_A = cv2.imread(os.path.join(BasePath, f'Val/P_A/{i}_PA.jpg'))
            P_B = cv2.imread(os.path.join(BasePath, f'Val/P_B/{i}_PB.jpg'))
            label = json.load(open(os.path.join(BasePath,f'Val/labels/{i}_label.json')))
            H4pt = label['H4pt']
            # Convert to grayscale and normalize
            P_A = cv2.cvtColor(P_A, cv2.COLOR_BGR2GRAY) / 255.0
            P_B = cv2.cvtColor(P_B, cv2.COLOR_BGR2GRAY) / 255.0
            
            # Add channel dimension and stack
            P_A = P_A[..., np.newaxis]
            P_B = P_B[..., np.newaxis]
            stacked_image = np.float32(np.concatenate([P_A, P_B], axis=2))
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.from_numpy(stacked_image).permute(2, 0, 1).unsqueeze(0).to(device)
            label_tensor = torch.tensor(H4pt, dtype=torch.float32).unsqueeze(0).to(device)
            # Get model prediction
            # loss = model.validation_step((input_tensor, label_tensor))
            pred = model(input_tensor)
            loss = F.mse_loss(pred, label_tensor)
            print(f"Loss for image {i} is {loss}")
            total_loss += loss.item()
    
        avg_loss = total_loss / image_count
        print('Average_Loss: ', avg_loss)
        
            


def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints/model.pth",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="../Data/CustomData/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    # ImageeSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder("float", shape=(1, ImageSize[0], ImageSize[1], 3))
    # LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels

    TestOperation(BasePath, ModelPath)

    # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)


if __name__ == "__main__":
    main()