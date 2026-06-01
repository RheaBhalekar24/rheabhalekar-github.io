"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pdb

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(delta, corners):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    lossfn = nn.MSELoss()  # ...
    loss = lossfn(delta, corners)
    return loss

class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        
    def training_step(self, batch):
        # img_a, patch_a, patch_b, corners, gt = batch
        img_a, corners = batch
        # Check if batch has NaN values
        if torch.isnan(img_a).any() or torch.isnan(corners).any():
            raise ValueError("Batch contains NaN values")
        pred_corners = self(img_a)
        if torch.isnan(pred_corners).any():
            raise ValueError("Prediction contains NaN values")
        loss = LossFn(pred_corners, corners)
        if torch.isnan(loss).any():
            raise ValueError("Loss contains NaN values")
        # logs = {"loss": loss}
        return loss

    def validation_step(self, batch):
        # img_a, patch_a, patch_b, corners, gt = batch
        img_a, corners = batch
        pred_corners = self(img_a)
        loss = LossFn(pred_corners, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Supervised(HomographyModel): #(nn.Module):
    def __init__(self, Input_Channels = 2, Output_channel = 8):
        """
        Inputs: Patches - PA and PB H4T matrix
        InputSize - Size of the Input 2 x 128 x 128 
        OutputSize - Size of the Output
        """
        # super().__init__()
        super(Supervised, self).__init__()
        #############################
        # Fill your network initialization of choice here!
        '''
        8 convolutional layers in the network 
        Maxpool layer with every 2 conv layer (2x2 stride = 2)
        '''
        
        #----------after every 2 layer we have a maxpool---------# 
        self.conv1 = nn.Conv2d(Input_Channels,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #----------after every 2 layer we have a maxpool---------# 

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        #----------after every 2 layer we have a maxpool---------# 

        self.conv5 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        #----------after every 2 layer we have a maxpool---------# 

        self.conv7 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn8 = nn.BatchNorm2d(128)


        #maxpool layer
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #dropout layer 
        self.drop = nn.Dropout(0.5)
        fc_input_features = 128 * 8 * 8  # Update this based on your input size and network architecture

        #fully connected layers 
        self.fc1 = nn.Linear(fc_input_features, 1024)
        self.fc2 = nn.Linear(1024,Output_channel)
        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        """
        Input:
        x = 2 channel 2 grayscale images
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        '---maxpool after every 2 layers --- '
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
    
        '---maxpool after every 2 layers --- '
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)

        '---maxpool after every 2 layers --- '
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.maxpool(x)

        # x = self.drop(x)
        # Flatten before FC layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.drop(x)

        x = self.fc1(x)
        x = self.fc2(x)

        out = x

        #############################
        return out

class SupervisedNew(HomographyModel):
    def __init__(self, Input_Channels = 2, Output_channel = 8):
        super(SupervisedNew, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(Input_Channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Layer 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Layer 7
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        
        # Final layer 
        self.fc2 = nn.Linear(1024, Output_channel)
        
    
    def forward(self, x):
        # Convolutional layers
        out = self.conv_layers(x)
        
        # Flatten and apply dropout
        out = out.view(-1, 128 * 16 * 16)
        out = self.dropout(out)
        
        # First fully connected layer
        out = F.relu(self.fc1(out))
        # out = self.dropout(out)
        out = self.fc2(out)
        return out