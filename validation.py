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
from webuser.render_hands import Render
import numpy as np
import trimesh

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)
class GestureDataset(Dataset):
    def __init__(self,img_path, json_file=None):
        self.json_file=json_file
        if self.json_file!=None:
            with open(self.json_file,"r") as f:
                self.json=json.load(f)
        self.img_size=224
        self.img_path=img_path
        self.images=glob.glob(img_path+"/*.jpg")
        # self.images=self.images[:20]
        # print(self.images)
        self.unique_imgs_num=32560
    def __getitem__(self,index):
        # print(index)
        # img_name=self.img_path+"/"+str(index).zfill(8)+".jpg"
        img_name=self.images[index]
        # print(img_name)
        img=cv2.imread(img_name)
        print(img.shape)
        img=cv2.resize(img,(self.img_size,self.img_size))
        # cv2.imshow("",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tensor_image=torch.FloatTensor(img)
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

# Using custom GestureDataset class to load train and test data respectively.
# data=GestureDataset("training/rgb","training_mano.json")
data_val=GestureDataset("evaluation/temp")

# Using the in-built DataLoader to create batches of images and labels for training validation respectively. 
# train_loader=torch.utils.data.DataLoader(dataset=data,batch_size=16,num_workers=0,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=data_val,batch_size=1,num_workers=0,shuffle=False)

# load=iter(train_loader).next()
# label = [element.item() for element in load[1]]
# print(label[3:])
device="cuda"
model=ResNet()
model=model.to("cuda")
model.eval()
checkpoint='checkpoints/checkpoint_augmented_90.pt'
if checkpoint:
    model.load_state_dict(torch.load(checkpoint))

def validate(val_loader,model):
    
    model.eval()
    hands={'global_orientation':[],'pose_parameters':[],'image_name':[]}
    render=Render()
    for i, (images,img_name) in tqdm.tqdm(enumerate(val_loader)):
        images=images.squeeze(1)
        outputs=model(images.to(device))
        # print(outputs)
        # print(img_name[0])
        img_input=cv2.imread(img_name[0])
        img_input=cv2.resize(img_input,(224,224))
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # break
        hands['global_orientation'].append(outputs[0,:3].cpu().detach())
        hands['pose_parameters'].append(outputs[0,3:].cpu().detach())
        hands['image_name'].append(img_name[0])
        # print(outputs)
        img, mesh=render.renderer(outputs)
        
        img=np.array(img)
        frame=hconcat_resize_min([img_input,img])
        cv2.imwrite("evaluation_output/output_"+os.path.basename(img_name[0]).split(".")[0]+".jpg",frame)
        mesh.export("evaluation_output/output_"+os.path.basename(img_name[0]).split(".")[0]+".obj")
        pkl_file=open("evaluation_output/hand-poses-freiHand.pkl",'wb')
        pkl.dump(hands,pkl_file)
        # break


# final=open("hand-poses-freiHand.pkl",'wb')

hands=validate(val_loader,model)

# pkl.dump(hands,final)

# print(hands)