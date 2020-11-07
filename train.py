import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
import glob
import json
import torchvision
import os
import PIL
import numpy as np
import random
from utils import *
from model import ResNet

# Using custom GestureDataset class to load train and test data respectively.
data=GestureDataset("training/rgb","training_mano.json")
data_val=GestureDataset("training/rgb")

# Using the in-built DataLoader to create batches of images and labels for training validation respectively. 
train_loader=torch.utils.data.DataLoader(dataset=data,batch_size=16,num_workers=0,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=data_val,batch_size=1,num_workers=0,shuffle=True)

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# Initialize model parameters
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
model=ResNet()
model=model.to(device)
model.train()
checkpoint=None
learning_rate=1e-4
start_epoch=0
end_epoch=100
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint_after=25
# Load checkpoint file, if any
if checkpoint:
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    start_epoch=torch.load(checkpoint)['epoch']

# Start training
for epoch in range(start_epoch,end_epoch+1):
    for i, (images,labels) in enumerate(train_loader):
        images=images.squeeze(1)
        outputs=model(images.to(device))
        loss=criterion(outputs.to(device),labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("-----------------------------------------------------------------")
    print("Epoch [{}/{}], Training Loss: {:.4f}".format(epoch,end_epoch,loss))
    if epoch%checkpoint_after==0:
        checkpoint="checkpoints/checkpoint_augmented_{}.pt".format(epoch)
        torch.save(model.state_dict(),checkpoint)

