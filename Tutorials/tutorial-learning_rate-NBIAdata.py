#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import random
import glob
import numpy as np
from scipy.ndimage import rotate
import csv
import SimpleITK as sitk
#from lungtumormask import mask as tumormask
#from lungmask import mask as lungmask_fun

from monai.utils import first, set_determinism
from monai.transforms import (
    RandFlipd,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    KeepLargestConnectedComponent,
    RandCropByPosNegLabeld,
    SaveImaged,
    CenterSpatialCropd,
    SpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    AsDiscrete,
    SpatialCrop,
    RandSpatialCropd,
    SpatialPadd,
    EnsureTyped,
    EnsureType,
    Invertd,
    DivisiblePadd,
    MapTransform,
    HistogramNormalized,
    ToTensord,
    Transpose,
    ToTensor,
)
from monai.optimizers import LearningRateFinder
from monai.handlers.utils import from_engine
from monai.networks.nets import SwinUNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric,SurfaceDiceMetric,SurfaceDistanceMetric,HausdorffDistanceMetric
from monai.losses import DiceLoss,DiceCELoss,MaskedDiceLoss,DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,pad_list_data_collate


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


if False:
    get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#peregrine
#root_path = '/data/p308104/NBIA_Data/NIFTI_NBIA/imagesTr/'
#root_path = '/data/p308104/Nifti_Imgs_V0/' #UMCG data on peregrine
#root_path = '/data/p308104/MultipleBP/'
#local
root_path = '/home/umcg/Desktop/NBIA/NBIA_Nifti_v0/'
#root_path = '/home/umcg/Desktop/Dicom_UMCG/MultipleBP/' 
#root_path = '/home/umcg/OneDrive/MultipleBreathingP/'
#root_path = '/home/umcg/Desktop/AutomaticITV_code/MultipleBreathingP-OneDriveCopy/MultipleBreathingP/'

all_patientdir = []
all_patientdir = os.listdir(root_path)
all_patientdir.sort()
print(len(all_patientdir))


# In[4]:


#Presets
num_workers = 6
set_determinism(seed=0)
image_keys = ["image","lung","label"]
p = .5 #Data aug transform probability
size = 96
image_size = (size,size,size)
batch_size=2
pin_memory = True if num_workers > 0 else False  # Do not change 


# In[5]:


CT_fpaths = []
lbl_fpaths= []
lung_fpaths = []
for patient_path in all_patientdir:
    ct_miss = True
    gtv_miss = True
    lung_miss = True
    for root, dirs, files in os.walk(root_path+patient_path, topdown=False):
        for f in files:
            if "_ct.nii.gz" in f.lower() and ct_miss:
                CT_fpaths.append(os.path.join(root_path,patient_path,f))
                ct_miss = False
            if "gtv-1.nii.gz" in f.lower():
                lbl_fpaths.append(os.path.join(root_path,patient_path,f))
                gtv_miss =False
            if 'lungmask.nii.gz' in f.lower():
                lung_fpaths.append(os.path.join(root_path,patient_path,f))
                lung_miss = False
        if ct_miss:
            for f in files:
                if "ex_ct.nii.gz" in f.lower() and ct_miss:
                    CT_fpaths.append(os.path.join(root_path,patient_path,f))
                    ct_miss = False
                if 'ex_lungmask.nii.gz' in f.lower() and lung_miss:
                        lung_fpaths.append(os.path.join(root_path,patient_path,f))
                        lung_miss = False
        if gtv_miss and not(ct_miss):
            CT_fpaths.pop()
        if gtv_miss and not(lung_miss):
            lung_fpaths.pop()
        if not(gtv_miss) and (ct_miss):
            print(root)
            for f in files:
                if "mar_ct.nii.gz" in f.lower() and ct_miss:
                    CT_fpaths.append(os.path.join(root_path,patient_path,f))
                    ct_miss = False
                if "in_ct.nii.gz" in f.lower() and ct_miss:
                    CT_fpaths.append(os.path.join(root_path,patient_path,f))
                    ct_miss = False
                if 'mar_lungmask.nii.gz' in f.lower() and lung_miss:
                    lung_fpaths.append(os.path.join(root_path,patient_path,f))
                    lung_miss = False
                if 'in_lungmask.nii.gz' in f.lower() and lung_miss:
                    lung_fpaths.append(os.path.join(root_path,patient_path,f))
                    lung_miss = False
                    
            
print(len(CT_fpaths),len(lbl_fpaths),len(lung_fpaths))
CT_fpaths.sort()
lbl_fpaths.sort()
lung_fpaths.sort()

print(CT_fpaths[44])
print(lbl_fpaths[44])
print(lung_fpaths[44])


# In[6]:


#Create data dictionat
data_dicts = [
    {"image": image_name,"lung":lung_name,"label": label_name}
    for image_name,lung_name,label_name in zip(CT_fpaths,lung_fpaths,lbl_fpaths)
]
train_files, val_files = data_dicts[:-277], data_dicts[-5:]
print('train val len:',len(train_files),'-',len(val_files))

minmin_CT = -1024 #NBIA
maxmax_CT = 3071 #NBIA


# In[7]:


#class to transpose lung mask
class Create_sequences(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)        
        print(f"keys to transpose: {self.keys}")
     
    def __call__(self, dictionary):
        dictionary = dict(dictionary)
        for key in self.keys:
            data = dictionary[key]
            if key == 'lung':
                data = np.transpose(data, (0,2,3,1))
                data = rotate(data,270,axes=(1,2),reshape=False)
                data = np.flip(data,1)
                data[data==2] = int(1)
                data[data!=1] = int(0)
            dictionary[key] = data
            
        return dictionary        


# In[8]:


#Create Compose functions for preprocessing of train and validation
train_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        EnsureChannelFirstd(keys=image_keys),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        #Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT,b_min=0.0, b_max=1.0, clip=True,),
        Create_sequences(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key="lung",k_divisible = size),
        RandCropByPosNegLabeld(
            keys=image_keys,label_key='label',spatial_size=image_size,pos=1,neg=1,num_samples=2,
            image_key='image',image_threshold=0,),
        ToTensord(keys=image_keys),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        EnsureChannelFirstd(keys=image_keys),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        #Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT,b_min=0.0, b_max=1.0, clip=True,),
        Create_sequences(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key="lung",k_divisible = size),
        ToTensord(keys=image_keys),
    ]
)


# In[9]:


#Check the images after the preprocessing
if True:
    train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0,num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers)

    val_ds = CacheDataset(data=val_files, transform=val_transforms,cache_rate=1.0,num_workers=int(num_workers//2))
    #val_loader = DataLoader(val_ds, batch_size=1, num_workers=int(num_workers//2),pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=int(num_workers // 2))
else:
    check_ds =CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0,num_workers=num_workers)
    check_loader = DataLoader(check_ds, batch_size=1,num_workers=num_workers)


# In[10]:


if False:
    count = 1
    for batch_data in train_loader:
        #batch_data = first(check_loader)
        image,lung, label = (batch_data["image"][0][0],batch_data["lung"][0][0],batch_data["label"][0][0])
        print(f"px info:{count },image shape: {image.shape},lung shape: {lung.shape}, label shape: {label.shape}")
        count+=1
        for i in range(label.shape[2]):
            if torch.sum(label[:,:,i])>0:
                plt.subplot(1,3,1),plt.imshow(image[:,:,i]),plt.axis('off')
                plt.subplot(1,3,2),plt.imshow(label[:,:,i]),plt.axis('off')
                plt.subplot(1,3,3),plt.imshow(label[:,:,i]+image[:,:,i]),plt.axis('off')
                plt.tight_layout(),plt.show()
                break
            break
        break


# In[11]:


#Create the model
spatial_dims = 3
max_epochs = 100
in_channels = 1
out_channels=2 #including background
weight_decay = 1e-5

model = SwinUNETR(
    image_size, 
    in_channels, out_channels, 
    use_checkpoint=True, 
    feature_size=24,
    #spatial_dims=spatial_dims
).to(device)


# In[12]:


#loss function

LF_Dice = DiceLoss(include_background=False,to_onehot_y=True, sigmoid=True)
loss_function = LF_Dice

#optimizer
optimizer = torch.optim.Adam(model.parameters(), 1e-4)



# In[13]:


#%matplotlib inline
lower_lr, upper_lr = 1e-5, 1e-0
optimizer = torch.optim.Adam(model.parameters(), lower_lr,weight_decay=weight_decay)
lr_finder = LearningRateFinder(model, optimizer, loss_function, device=device)
lr_finder.range_test(train_loader, val_loader, end_lr=upper_lr, num_iter=20)
steepest_lr, _ = lr_finder.get_steepest_gradient()
ax = plt.subplots(1, 1, figsize=(15, 15), facecolor="white")[1]
_ = lr_finder.plot(ax=ax)


metric_values = []
nr_images = 8
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 100)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs,labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    


# In[ ]:


inputs.shape


# In[ ]:


def plot_range(data, wrapped_generator):
    plt.ion()
    for q in data.values():
        for d in q.values():
            if isinstance(d, dict):
                ax = d["line"].axes
                ax.legend()
                fig = ax.get_figure()
    fig.show()

    for i in wrapped_generator:
        yield i
        for q in data.values():
            for d in q.values():
                if isinstance(d, dict):
                    d["line"].set_data(d["x"], d["y"])
                    ax = d["line"].axes
                    ax.legend()
                    ax.relim()
                    ax.autoscale_view()
        fig.canvas.draw()


# In[ ]:


def get_model_optimizer_scheduler(d):
    d["model"] = get_new_net()

    if "lr_lims" in d:
        d["optimizer"] = torch.optim.Adam(
            d["model"].parameters(), d["lr_lims"][0]
        )
        d["scheduler"] = torch.optim.lr_scheduler.CyclicLR(
            d["optimizer"],
            base_lr=d["lr_lims"][0],
            max_lr=d["lr_lims"][1],
            step_size_up=d["step"],
            cycle_momentum=False,
        )
    elif "lr_lim" in d:
        d["optimizer"] = torch.optim.Adam(d["model"].parameters(), d["lr_lim"])
    else:
        d["optimizer"] = torch.optim.Adam(d["model"].parameters())


def train(max_epochs, axes, data):
    for d in data.keys():
        get_model_optimizer_scheduler(data[d])

        for q, i in enumerate(["train", "auc", "acc"]):
            data[d][i] = {"x": [], "y": []}
            (data[d][i]["line"],) = axes[q].plot(
                data[d][i]["x"], data[d][i]["y"], label=d
            )

        val_interval = 1

    for epoch in plot_range(data, trange(max_epochs)):

        for d in data.keys():
            data[d]["epoch_loss"] = 0
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            for d in data.keys():
                data[d]["optimizer"].zero_grad()
                outputs = data[d]["model"](inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                data[d]["optimizer"].step()
                if "scheduler" in data[d]:
                    data[d]["scheduler"].step()
                data[d]["epoch_loss"] += loss.item()
        for d in data.keys():
            data[d]["epoch_loss"] /= len(train_loader)
            data[d]["train"]["x"].append(epoch + 1)
            data[d]["train"]["y"].append(data[d]["epoch_loss"])

        if (epoch + 1) % val_interval == 0:
            with eval_mode(*[data[d]["model"] for d in data.keys()]):
                for d in data:
                    data[d]["y_pred"] = torch.tensor(
                        [], dtype=torch.float32, device=device
                    )
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    for d in data:
                        data[d]["y_pred"] = torch.cat(
                            [data[d]["y_pred"], data[d]["model"](val_images)],
                            dim=0,
                        )
                    y = torch.cat([y, val_labels], dim=0)

                for d in data:
                    y_onehot = [y_trans(i) for i in decollate_batch(y)]
                    y_pred_act = [y_pred_trans(i).cpu() for i in decollate_batch(data[d]["y_pred"])]
                    auc_metric(y_pred_act, y_onehot)
                    auc_result = auc_metric.aggregate()
                    auc_metric.reset()
                    del y_pred_act, y_onehot
                    data[d]["auc"]["x"].append(epoch + 1)
                    data[d]["auc"]["y"].append(auc_result)

                    acc_value = torch.eq(data[d]["y_pred"].argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    data[d]["acc"]["x"].append(epoch + 1)
                    data[d]["acc"]["y"].append(acc_metric)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
fig, axes = plt.subplots(3, 1, figsize=(10, 10), facecolor="white")
for ax in axes:
    ax.set_xlabel("Epoch")
axes[0].set_ylabel("Train loss")
axes[1].set_ylabel("AUC")
axes[2].set_ylabel("ACC")

# In the paper referenced at the top of this notebook, a step
# size of 8 times the number of iterations per epoch is suggested.
step_size = 8 * len(train_loader)

max_epochs = 100
data = {}
data["Default LR"] = {}
data["Steepest LR"] = {"lr_lim": steepest_lr}
data["Cyclical LR"] = {
    "lr_lims": (0.8 * steepest_lr, 1.2 * steepest_lr),
    "step": step_size,
}

train(max_epochs, axes, data)

