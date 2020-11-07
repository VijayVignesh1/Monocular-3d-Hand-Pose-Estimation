
import torch
import cv2
from torch.utils.data import Dataset
import torch.nn as nn
import glob
import json
import torchvision
import PIL
import numpy as np
import random
from webuser.render_hands import Render
import tqdm, os
import pickle as pkl

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

# Concatenate two images horizontally
# Used for saving input image and output images as one images during validation
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

# Apply motion blur to incorporate detection of blurred hands
def motion_blur(img):
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
    return output

# Fuction to test/validate input images
# outputs mesh(.obj), image and pose parameters(.pkl) to "evaluation_outputs" folder
def validate(val_loader,model):
    model.eval()
    hands={'global_orientation':[],'pose_parameters':[],'image_name':[]}
    render=Render()
    for i, (images,img_name) in tqdm.tqdm(enumerate(val_loader)):
        images=images.squeeze(1)
        outputs=model(images.to(device))
        img_input=cv2.imread(img_name[0])
        img_input=cv2.resize(img_input,(224,224))
        hands['global_orientation'].append(outputs[0,:3].cpu().detach())
        hands['pose_parameters'].append(outputs[0,3:].cpu().detach())
        hands['image_name'].append(img_name[0])
        img, mesh=render.renderer(outputs)
        img=np.array(img)
        frame=hconcat_resize_min([img_input,img])
        cv2.imwrite("evaluation_output/output_"+os.path.basename(img_name[0]).split(".")[0]+".jpg",frame)
        mesh.export("evaluation_output/output_"+os.path.basename(img_name[0]).split(".")[0]+".obj")
        pkl_file=open("evaluation_output/output_"+os.path.basename(img_name[0]).split(".")[0]+".pkl",'wb')
        pkl.dump(hands,pkl_file)


# Dataset Loaderclass to load the data for training
class GestureDataset(Dataset):
    def __init__(self,img_path, json_file=None):
        self.json_file=json_file
        if self.json_file!=None:
            with open(self.json_file,"r") as f:
                self.json=json.load(f)
        self.img_size=224
        self.img_path=img_path
        self.images=glob.glob(img_path+"/*.jpg")
        self.transform=torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0), interpolation=2),
                                            torchvision.transforms.ColorJitter(brightness=(0.8,1.0),contrast=(0.8,1.0)),
                                            motion_blur])
        self.unique_imgs_num=32560
    def __getitem__(self,index):
        img_name=self.img_path+"/"+str(index).zfill(8)+".jpg"
        img=cv2.imread(img_name)
        img=cv2.resize(img,(self.img_size,self.img_size))
        img_pil=PIL.Image.open(img_name)
        tensor_image=self.transform(img_pil)
        tensor_image=torch.FloatTensor(tensor_image)
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


class GestureValDataset(Dataset):
    def __init__(self,img_path, json_file=None):
        self.json_file=json_file
        if self.json_file!=None:
            with open(self.json_file,"r") as f:
                self.json=json.load(f)
        self.img_size=224
        self.img_path=img_path
        self.images=glob.glob(img_path+"/*.jpg")
        self.unique_imgs_num=32560
    def __getitem__(self,index):
        img_name=self.images[index]
        img=cv2.imread(img_name)
        img=cv2.resize(img,(self.img_size,self.img_size))
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
