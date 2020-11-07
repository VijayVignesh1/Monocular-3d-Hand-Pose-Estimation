import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
import glob
import json
import torchvision
import os
import tqdm
import pickle as pkl
import numpy as np
import trimesh
from utils import GestureValDataset, validate
from model import ResNet

# Using custom GestureValDataset class to load validation data.
data_val=GestureValDataset("evaluation/temp")

# Using the in-built DataLoader to create batches of images and labels for training validation respectively. 
val_loader=torch.utils.data.DataLoader(dataset=data_val,batch_size=1,num_workers=0,shuffle=False)

# Initialize model

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
model=ResNet()
model=model.to(device)
model.eval()
checkpoint='checkpoints/checkpoint_augmented_90.pt'
model.load_state_dict(torch.load(checkpoint))

# Validation
hands=validate(val_loader,model)
