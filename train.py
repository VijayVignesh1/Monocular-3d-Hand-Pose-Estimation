import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import f1_score
import glob
import json
import torchvision
import os
import PIL
import numpy as np
import random
def motion_blur(img):
    # print(img.size)
    sizes=[1,7,9,11,13]
    size=random.choice(sizes)
    img=np.array(img)
    img=img[:, :, ::-1] 
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = PIL.Image.fromarray(img)
    return output


class GestureDataset(Dataset):
    def __init__(self,img_path, json_file=None):
        self.json_file=json_file
        if self.json_file!=None:
            with open(self.json_file,"r") as f:
                self.json=json.load(f)
        self.img_size=224
        self.img_path=img_path
        self.images=glob.glob(img_path+"/*.jpg")
        self.images=self.images[:20]
        self.transform=torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0), interpolation=2),
                                            torchvision.transforms.ColorJitter(brightness=(0.8,1.0),contrast=(0.8,1.0)),
                                            motion_blur])
        self.unique_imgs_num=32560
    def __getitem__(self,index):
        # print(index)
        img_name=self.img_path+"/"+str(index).zfill(8)+".jpg"
        img=cv2.imread(img_name)
        img=cv2.resize(img,(self.img_size,self.img_size))
        img_pil=PIL.Image.open(img_name)

        # tensor_image=torch.FloatTensor(img)

        tensor_image=self.transform(img_pil)
        # img=np.array(tensor_image)
        tensor_image=torch.FloatTensor(tensor_image)
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tensor_image=tensor_image.permute(2,0,1)
        tensor_image=tensor_image.unsqueeze(0)
        tensor_image/=255.

        label_index=index%self.unique_imgs_num
        if self.json_file:
            return tensor_image,torch.FloatTensor(self.json[label_index][0][:48])
        else:
            return tensor_image,img_name
    def __len__(self):
        return len(self.images)

# Using custom GestureDataset class to load train and test data respectively.
data=GestureDataset("training/rgb","training_mano.json")
data_val=GestureDataset("training/rgb")

# Using the in-built DataLoader to create batches of images and labels for training validation respectively. 
train_loader=torch.utils.data.DataLoader(dataset=data,batch_size=16,num_workers=0,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=data_val,batch_size=1,num_workers=0,shuffle=True)

# load=iter(train_loader).next()
# label = [element.item() for element in load[1]]
# print(label[3:])

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet=torchvision.models.resnet101(pretrained=True)
        self.fc1=torch.nn.Linear(1000,512)
        self.fc2=torch.nn.Linear(512,48)
    def forward(self,image):
        out=self.resnet(image)  
        out=self.fc1(out)
        out=self.fc2(out)
        return out

def validate(val_loader,model):
    model.eval()
    for i, (images,img_name) in enumerate(val_loader):
        images=images.squeeze(1)
        outputs=model(images.to(device))
        # print(outputs)
        # print(img_name[0])
        img=cv2.imread(img_name[0])
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        break


model=ResNet()
model=model.to("cuda")
model.train()
checkpoint=None
device="cuda"
learning_rate=1e-4
start_epoch=0
end_epoch=100
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose= True, min_lr=1e-6)
if checkpoint:
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    start_epoch=torch.load(checkpoint)['epoch']
for epoch in range(start_epoch,end_epoch+1):
    for i, (images,labels) in enumerate(train_loader):
        images=images.squeeze(1)
        outputs=model(images.to(device))
        loss=criterion(outputs.to(device),labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # predicted = torch.softmax(outputs,dim=1)
        # _,predicted=torch.max(predicted, 1)
        # f1=f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),average='weighted')
    validate(val_loader,model)
    print("-----------------------------------------------------------------")
    print("Epoch [{}/{}], Training Loss: {:.4f}".format(epoch,end_epoch,loss))
    # scheduler.step(val_accuracy)
    if epoch%10==0:
        checkpoint="checkpoints/checkpoint_augmented_{}.pt".format(epoch)
        torch.save(model.state_dict(),checkpoint)
