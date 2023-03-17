
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
    EnsureChannelFirstd,
    Compose,
    MaskIntensityd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    AsDiscrete,
    EnsureType,
    Invertd,
    DivisiblePadd,
    MapTransform,
    HistogramNormalized,
    ToTensord,
    Transpose,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet,VNet,SwinUNETR,UNETR,DynUNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric,SurfaceDiceMetric,SurfaceDistanceMetric,HausdorffDistanceMetric
from monai.losses import DiceLoss,DiceCELoss,MaskedDiceLoss,DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,pad_list_data_collate
import wandb


def LookSortFiles(root_path,all_patientdir):
    CT_fpaths = []
    lbl_fpaths = []
    lung_fpaths = []
    for patient_path in all_patientdir:
        ct_miss = True
        gtv_miss = True
        lung_miss = True
        for root, dirs, files in os.walk(root_path + patient_path, topdown=False):
            for f in files:
                if False:  # NBIA database
                    if "_ct.nii.gz" in f.lower() and ct_miss:
                        CT_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss = False
                    if "gtv-1.nii.gz" in f.lower():
                        lbl_fpaths.append(os.path.join(root_path, patient_path, f))
                        gtv_miss = False
                    if '_lungmask.nii.gz' in f.lower():
                        lung_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss = False
                if True:  # 4D Data local
                    if "50%_ct.nii.gz" in f.lower():
                        CT_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss = False
                    if "rtstruct_gtv.nii.gz" in f.lower():
                        lbl_fpaths.append(os.path.join(root_path, patient_path, f))
                        gtv_miss = False
                    if '50%_lungmask.nii.gz' in f.lower():
                        lung_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss = False
            if ct_miss and lung_miss:
                for f in files:
                    if "ex_ct.nii.gz" in f.lower() and ct_miss:
                        CT_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss = False
                    if 'ex_lungmask.nii.gz' in f.lower() and lung_miss:
                        lung_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss = False
            if ct_miss and lung_miss:
                for f in files:
                    if "mar_ct.nii.gz" in f.lower() and ct_miss:
                        CT_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss = False
                    if 'mar_lungmask.nii.gz' in f.lower() and lung_miss:
                        lung_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss = False
            if ct_miss and lung_miss:
                for f in files:
                    if "in_ct.nii.gz" in f.lower() and ct_miss:
                        CT_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss = False
                    if 'in_lungmask.nii.gz' in f.lower() and lung_miss:
                        lung_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss = False
            if gtv_miss and len(files) > 0:
                CT_fpaths.pop()
                ct_miss = True
                lung_fpaths.pop()
                lung_miss = True

    print('ct: ', len(CT_fpaths), 'label: ', len(lbl_fpaths), 'lung: ', len(lung_fpaths))
    CT_fpaths.sort()
    lbl_fpaths.sort()
    lung_fpaths.sort()

    print(CT_fpaths[121])
    print(lbl_fpaths[121])
    print(lung_fpaths[121])

    return CT_fpaths,lbl_fpaths,lung_fpaths

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

        # task06 Lung NSCLC tumor segmentation - medicaldecathlon.com/#tasks
        # pretrained functions where describred in the github of create_network.py

def get_kernels_strides(patch_size, spacing):
    sizes, spacings = patch_size, spacing
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

#params
def Loss_functionsSelector(loss_fold):
    if loss_fold==0:
        SelectedLoss = DiceLoss(include_background=False,to_onehot_y=True, sigmoid=True)
    if loss_fold==1:        
        SelectedLoss = MaskedDiceLoss(include_background=False,to_onehot_y=True, sigmoid=True)
    if loss_fold==2:
        SelectedLoss = DiceCELoss(include_background=False,to_onehot_y=True, sigmoid=True)
    if loss_fold==3:
        SelectedLoss = DiceFocalLoss(include_background=False,to_onehot_y=True, sigmoid=True)
    return SelectedLoss


def train(dice_metric,image_size,lf_select,optimizer,model,root_path,epoch, train_loader, val_loader, scheduler, best_metric, best_metric_epoch,device):
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, threshold=0.8)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)], )

    train_num_iterations = len(train_loader)
    epoch_loss = 0
    step = 0
    epoch_loss_values = []
    metric_values = []
    val_interval = 2
    test_loss=0
    
    model.train()
    
    for batch_data in train_loader:
        step += 1
        inputs,labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_function=Loss_functionsSelector(lf_select)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        scheduler.step(epoch + (step / train_num_iterations))
        
        if True:
            print(f"{step}/{len(train_loader) // train_loader.batch_size},"f"train_loss: {loss.item():.4f}")
        
        
        wandb.log({"train/loss": loss.item()})
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    wandb.log({"train/loss_epoch": epoch_loss})
    wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
    

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),)
                roi_size = image_size
                sw_batch_size = 1
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

                loss_function = Loss_functionsSelector(lf_select)
                loss_fun = loss_function(val_outputs, val_labels)
                test_loss += loss_fun.item()

                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                #postImg_monai = [post_transforms(i) for i in decollate_batch(val_outputs)]

                #compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                #dice_metric(y_pred=postImg_monai[0][1:2,:,:,:], y=val_labels[0])


            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            wandb.log({"val/dice_metric": metric})            
            dice_metric.reset()

            wandb.log({"test/loss": test_loss})


            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_path, "best_DynUnet_V11_UMCG_TestSet"+str(lf_select)+".pth"))
                print("saved new best metric model")
            else:
                print("Model not saved")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}")

    epoch +=1
    return epoch, best_metric, best_metric_epoch

def main(name_run,lf_select,cache,num_workers,root_path):

    all_patientdir = []
    all_patientdir = os.listdir(root_path)
    all_patientdir.sort()
    print(len(all_patientdir))

    CT_fpaths,lbl_fpaths,lung_fpaths = LookSortFiles(root_path,all_patientdir)
    
    #Create data dictionat
    data_dicts = [
        {"image": image_name,"lung":lung_name,"label": label_name}
        for image_name,lung_name,label_name in zip(CT_fpaths,lung_fpaths,lbl_fpaths)
    ]
    train_files, val_files = data_dicts[:-10], data_dicts[-10:]
    print('train val len:',len(train_files),'-',len(val_files))

    # HU are -1000 air , 0 water , usually normal tissue are around 0, top values should be around 100, bones are around 1000
    minmin_CT = -1024
    maxmax_CT = 200

    #Create Compose functions for preprocessing of train and validation
    set_determinism(seed=0)
    image_keys = ["image","lung","label"]
    p = .5 #Data aug transform probability
    size = 96
    image_size = (size,size,size)
    pin_memory = True if num_workers > 0 else False  # Do not change 

    train_transforms = Compose(
        [
            LoadImaged(keys=image_keys),
            EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            #Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT,b_min=0.0, b_max=1.0, clip=True,),
            Create_sequences(keys=image_keys),
            CropForegroundd(keys=image_keys, source_key="lung",k_divisible = size),
            MaskIntensityd(keys=["image"],mask_key="lung"),
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
            MaskIntensityd(keys=["image"],mask_key="lung"),
            ToTensord(keys=image_keys),
        ]
    )
    
    #Check the images after the preprocessing
    if cache:#Cache
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0,num_workers=num_workers)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers)
        val_ds = CacheDataset(data=val_files, transform=val_transforms,cache_rate=1.0,num_workers=int(num_workers//2))
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=int(num_workers//2),pin_memory=pin_memory)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)#,collate_fn=pad_list_data_collate)

        
    #Create the model
    spatial_dims = 3
    max_epochs = 250
    in_channels = 1
    out_channels=2 #including background
    lr = 1e-3#1e-4
    weight_decay = 1e-5
    T_0 = 40  # Cosine scheduler

    task_id = "06"
    deep_supr_num = 1  # when is 3 shape of outputs/labels dont match
    patch_size = image_size
    spacing = [1, 1, 1]
    kernels, strides = get_kernels_strides(patch_size, spacing)
    model = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=False,  # when is 3 shape of outputs/labels dont match
        deep_supr_num=deep_supr_num,
    ).to(device)


    
    # Load pretrained model
    pretrained_path = 'None'
    if pretrained_path != 'None':
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(device)))
        print('Using pretrained models!')
    else:
        print('No pretraining model')
    # Load Pretrained Weights for Swin Unet
    weights_path = 'None'#'/home/p308104/jobs/model_swinvit.pt'
    if weights_path != 'None':
        weight = torch.load(weights_path, map_location=torch.device(device))
        model.load_from(weights=weight)
        print('Using pretrained weights')
    else:
        print('No pre weights ')


    #optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    #learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0,T_mult=1, eta_min=1e-9)
    #metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean",get_not_nans=False)
    
    wandb.init(project="UMCG_V9", entity="ldelaoa",name=name_run)
    wandb.config = {
      "learning_rate": lr,
      "epochs": max_epochs,
      "batch_size": 2,
      "loss function":Loss_functionsSelector(lf_select)
    }

    wandb.watch(model, log_freq=100)
    
    best_metric = -1
    best_metric_epoch = -1
    best_metric = 0.0
    epoch_early_stopping = 10
    epoch = 0

    #for epoch in range(max_epochs):
    while (epoch < max_epochs) or (epoch - best_metric_epoch >= epoch_early_stopping):
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch, best_metric, best_metric_epoch  = train(dice_metric,image_size,lf_select,optimizer,model,root_path,epoch,train_loader,val_loader,scheduler,best_metric,best_metric_epoch,device)
    
    print(
            f'Train completed, best_metric: {best_metric:.4f} '
            f'at iteration: {best_metric_epoch}'
        )
    print('Epoch is:',epoch)
    
    wandb.log({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    num_workers=12

    #peregrine
    #root_path = '/data/p308104/NBIA_Data/NIFTI_NBIA/imagesTr/'
    #root_path = '/data/p308104/Nifti_Imgs_V0/' #UMCG data on peregrine
    root_path = '/data/p308104/MultipleBP/'
    #local
    #root_path = '/home/umcg/Desktop/NBIA/NBIA_Nifti_v0/'
    #root_path = '/home/umcg/Desktop/Dicom_UMCG/MultipleBP/'
    #root_path = '/home/umcg/OneDrive/MultipleBreathingP/'
    #root_path = '/home/umcg/Desktop/AutomaticITV_code/MultipleBreathingP-OneDriveCopy/MultipleBreathingP/'

    if False:
        get_ipython().run_line_magic('matplotlib', 'inline')
    
    cache=True
    lf_select =3
    print('LF is ',lf_select,'Db is UMCG','DynUnet')
    name_run = "DynUnet"+"LF"+str(lf_select)+"run4_lungMask"
    main(name_run,lf_select,cache,num_workers,root_path)

#Changes from NBIA versions are:
# 1- patients file seeker function optimized for umcg db
# 2- lr is higher and eta_min lower, aiming for a broader area on where to look
# 3- WandB project name is now UMCG_V9
# 4- Training with all the images, not anymore the 75-25 ratio
# 5- Max Epochs is 250 now
