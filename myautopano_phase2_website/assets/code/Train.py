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
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW, SGD, lr_scheduler, Adam
from Network.Network import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Network.Network import HomographyModel, Supervised, SupervisedNew
from Network.Network import LossFn
import json
import kornia
import pdb

# Don't generate pyc codes
sys.dont_write_bytecode = True

#Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device IS: ", device)

def TensorDLT(H4PT, CA,MiniBatchSize):
    '''
    input - H4pt matrix of dim(4,2) 
            CA corners of the input og image
    output -- H matrix of 3x3
    '''
    H_batch = torch.tensor([]).to(device).reshape(0, 3, 3)  # Initialize an empty tensor to store the batch of H matrices
    A = []
    A = torch.tensor(A).to(device)
    H = torch.tensor([])
    H4PT = H4PT.reshape(MiniBatchSize,4, 2)
    for H4PT, CA in zip(H4PT, CA):
        CB = CA + H4PT 
        A = [] # 8 x 9 matrix for multiplying to h matrix 
        for i in range(4):
            u_i = CA[i][0]
            v_i = CA[i][1]
            u_dash_i = CB[i][0]
            v_dash_i = CB[i][1]
            # pdb.set_trace()
            one = torch.tensor(1, device=device)
            zero = torch.tensor(0, device=device)
            A.append(([u_i, v_i, one, zero, zero, zero, -u_dash_i*u_i, -u_dash_i*v_i, -u_dash_i]))
            A.append(([zero, zero, zero, u_i, v_i, one, -v_dash_i*u_i, -v_dash_i*v_i, -v_dash_i]))
        # pdb.set_trace()
        A = torch.tensor(A)
        _, _, V = torch.linalg.svd(A)
        h = V[-1]
        H = h.reshape(3, 3)
        # Normalize to make H[2,2] = 1 (scale-invariant)
        H /= H[2, 2]
        H = H.to(device)  # Ensure H is on the same device as H_batch
        # H = torch.cat((H, torch.unsqueeze(h.reshape(3, -1), dim=0)), axis=0)
        H_batch = torch.cat((H_batch, torch.unsqueeze(H, dim=0)), axis=0)
    return H_batch

def supervised_Batch(DataPath,MiniBatchSize,Train = True):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    patches_Batch = []
    H4t_batch = []

    if Train == True:
        Base_Path = os.path.join(DataPath, "Train")
    else:
        Base_Path = os.path.join(DataPath,"Val")
    
    pa_path = os.path.join(Base_Path, 'P_A')
    pb_path = os.path.join(Base_Path, 'P_B')
    labels_path = os.path.join(Base_Path, 'labels')
    image_count = len(glob.glob(os.path.join(pa_path, '*.jpg')))
        
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(1, image_count)
        RandImageNamePA = f"{RandIdx}_PA.jpg"
        RandImageNamePB = f"{RandIdx}_PB.jpg"
        RandLabelName = f"{RandIdx}_label.json"

        P_A = cv2.imread(os.path.join(pa_path, RandImageNamePA))
        P_B = cv2.imread(os.path.join(pb_path, RandImageNamePB))
        label = json.load(open(os.path.join(labels_path,RandLabelName)))
        # Convert directly to grayscale
        P_A = cv2.cvtColor(P_A, cv2.COLOR_BGR2GRAY) / 255.0
        P_B = cv2.cvtColor(P_B, cv2.COLOR_BGR2GRAY) / 255.0

        # Add channel dimension and stack
        P_A = P_A[..., np.newaxis]
        P_B = P_B[..., np.newaxis]

        stacked_image = np.float32(np.concatenate([P_A, P_B], axis=2))
        H4pt = label['H4pt']
        ImageNum += 1

        patches_Batch.append(torch.from_numpy(stacked_image).permute(2, 0, 1))
        H4t_batch.append(torch.tensor(H4pt))

    return torch.stack(patches_Batch).to(device, non_blocking=True), torch.stack(H4t_batch).to(device, non_blocking=True)

def unsupervised_Batch(DataPath,MiniBatchSize,Train = True):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    patches_Batch = []
    H4t_batch = []
    C_A_batch = []

    if Train == True:
        Base_Path = os.path.join(DataPath, "Train")
    else:
        Base_Path = os.path.join(DataPath,"Val")
    
    pa_path = os.path.join(Base_Path, 'P_A')
    pb_path = os.path.join(Base_Path, 'P_B')
    labels_path = os.path.join(Base_Path, 'labels')
    image_count = len(glob.glob(os.path.join(pa_path, '*.jpg')))
        
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(1, image_count)
        RandImageNamePA = f"{RandIdx}_PA.jpg"
        RandImageNamePB = f"{RandIdx}_PB.jpg"
        RandLabelName = f"{RandIdx}_label.json"

        P_A = cv2.imread(os.path.join(pa_path, RandImageNamePA))
        P_B = cv2.imread(os.path.join(pb_path, RandImageNamePB))
        label = json.load(open(os.path.join(labels_path,RandLabelName)))

        P_A = cv2.cvtColor(P_A, cv2.COLOR_BGR2GRAY) / 255.0
        P_B = cv2.cvtColor(P_B, cv2.COLOR_BGR2GRAY) / 255.0

        # Add channel dimension and stack
        P_A = P_A[..., np.newaxis]
        P_B = P_B[..., np.newaxis]

        stacked_image = np.float32(np.concatenate([P_A, P_B], axis=2))

        CA = label['C_A']
        ImageNum += 1
        patches_Batch.append(torch.from_numpy(stacked_image).permute(2, 0, 1))
        # H4t_batch.append(torch.tensor(H4pt))
        C_A_batch.append(torch.tensor(CA))

    return torch.stack(patches_Batch).to(device, non_blocking=True), torch.stack(C_A_batch).to(device, non_blocking=True)




def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # model = Supervised().to(device) #HomographyModel().to(device)
    model = SupervisedNew().to(device)

    # Predict output with forward pass

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    if ModelType == 'unsupervised':
        Optimizer = AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
    else:
        Optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
        
    lr_step = int(NumEpochs/3)
    scheduler = lr_scheduler.StepLR(
        Optimizer, 
        step_size=lr_step, 
        gamma=0.1
    )
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)
    LossThisBatch = 0
    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    
    #Training
    pa_path = os.path.join(BasePath, 'Train/P_A')
    image_count = len(glob.glob(os.path.join(pa_path, '*.jpg')))
    print("Pa path: ", pa_path, "Image count: ", image_count)
    NumTrainSamples = image_count
    print(f" Epochs: {NumEpochs}\n,Train samples: {NumTrainSamples} ,\n MiniBatchSize: {MiniBatchSize},\n Per Epoch Iterations: {int(NumTrainSamples / MiniBatchSize / DivTrain)}")
    # In TrainOperation(), add before the training loop:
    torch.cuda.empty_cache()
    model.train()
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        train_loss_epoch = 0
        val_loss_epoch = 0

        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            if ModelType == 'supervised':
                I1Batch = supervised_Batch(BasePath, MiniBatchSize)
                LossThisBatch = model.training_step(I1Batch)
                Optimizer.zero_grad()
                LossThisBatch.backward()
                train_loss_epoch += LossThisBatch.item()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping here
                Optimizer.step()
            elif ModelType == 'unsupervised':
                # model = Supervised().to(device) 
                I1Batch, CA_batch = unsupervised_Batch(BasePath, MiniBatchSize)
                PA_batch = I1Batch[:, 0:1, :, :] 
                PB_batch = I1Batch[:, 0:1, :, :] 
                H4pt_batch = model(I1Batch)
                H_batch = TensorDLT(H4pt_batch,CA_batch,MiniBatchSize)
                warped_pa_batch = kornia.geometry.transform.warp_perspective(PA_batch.float(), H_batch, (128, 128))
                LossThisBatch = torch.nn.L1Loss()(warped_pa_batch, PB_batch.float())
                H_batch.requires_grad = True
                warped_pa_batch.requires_grad = True
                LossThisBatch.requires_grad = True

                Optimizer.zero_grad()
                LossThisBatch.backward()
                Optimizer.step()    
            train_loss_epoch += LossThisBatch.item()        
        #Step scheduler
        scheduler.step()
        # At the end of each epoch, compute average loss and accuracy
        avg_train_loss = train_loss_epoch / NumIterationsPerEpoch
        # "------- validation step---------------------------------"
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            if ModelType == 'supervised':
                I1Batch, CoordinatesBatch = supervised_Batch(BasePath, MiniBatchSize)
                _val_loss += model.validation_step(I1Batch, CoordinatesBatch)
            elif ModelType == 'unsupervised':
                # stacked_patches_batch, c_a_batch, pa_batch, pb_batch = unsupervised_Batch(DatasetPath, DirNamesVal, train_batchsize, train=False)
                I1Batch, CA_batch = unsupervised_Batch(BasePath, MiniBatchSize)
                PA_batch = I1Batch[:, 0:1, :, :] 
                PB_batch = I1Batch[:, 0:1, :, :] 
                H4pt_batch = model(I1Batch)
                H_batch = TensorDLT(H4pt_batch,CA_batch,MiniBatchSize)
                warped_pa_batch = kornia.geometry.transform.warp_perspective(PA_batch.float(), H_batch, (128, 128))
                LossThisBatch_val = torch.nn.L1Loss()(warped_pa_batch, PB_batch.float())
            val_loss_epoch += LossThisBatch_val.item()        
        # At the end of each epoch, compute average loss and accuracy
        avg_val_loss = val_loss_epoch / NumIterationsPerEpoch
        print(f"Epoch {Epochs}: Train Loss: {avg_train_loss} , val Loss: {avg_val_loss}")
        # Tensorboard
        Writer.add_scalar(
                "LossEveryepoch",
                avg_train_loss,
                Epochs,
            )
        Writer.flush()
        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data/CustomData/",
        help="Base path of images, Default:../Data/CustomData",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/Unsuper2/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="supervised",
        help="Model type, Supervised or Unsupervised? Choose from [supervised, unsupervised] Default:supervised",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/Unsuper2/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType = ModelType,
    )


if __name__ == "__main__":
    main()
