import os
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import random
import glob
import nibabel as nib
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
from monai.networks.nets import UNet,VNet,SwinUNETR,UNETR,DynUNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric,SurfaceDiceMetric,SurfaceDistanceMetric,HausdorffDistanceMetric
from monai.losses import DiceLoss,DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,pad_list_data_collate
from monai.config import print_config
from monai.apps import download_and_extract

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


if False:
    get_ipython().run_line_magic('matplotlib', 'inline')


#peregrine
root_path = '/data/p308104/NBIA_Data/NIFTI_NBIA/imagesTr/'
#root_path = '/data/p308104/Nifti_Imgs_V0/' #UMCG data on peregrine
#root_path = '/data/p308104/MultipleBP/'
#local
#root_path = '/home/umcg/Desktop/NBIA/NBIA_Nifti_v0/'
#root_path = '/home/umcg/Desktop/Dicom_UMCG/MultipleBP/' 
#root_path = '/home/umcg/OneDrive/MultipleBreathingP/'
#root_path = '/home/umcg/Desktop/AutomaticITV_code/MultipleBreathingP-OneDriveCopy/MultipleBreathingP/'


all_patientdir = []
all_patientdir = os.listdir(root_path)
all_patientdir.sort()
print(len(all_patientdir))


CT_fpaths = []
lbl_fpaths= []
lung_fpaths = []
for patient_path in all_patientdir:
    flag_PxOk=0
    ct_miss = True
    lung_miss = True
    gtv_miss = True
    for root, dirs, files in os.walk(root_path+patient_path, topdown=False):   
        for f in files:
            if "ct.nii.gz" in f.lower() and ct_miss:
                CT_fpaths.append(os.path.join(root_path,patient_path,f))
                flag_PxOk+=1
                ct_miss = False
            if "gtv-1.nii.gz" in f.lower():
                lbl_fpaths.append(os.path.join(root_path,patient_path,f))
                flag_PxOk+=1
                gtv_miss =False
            if ("lungmask.nii.gz" in f.lower()) and lung_miss:
                lung_fpaths.append(os.path.join(root_path,patient_path,f))
                lung_miss = False
                flag_PxOk+=1

        if flag_PxOk!=3:
            list_PxNOTOk.append(patient_path)
            if ct_miss==False:
                print(patient_path,flag_PxOk,'GTV Miss: ',gtv_miss,'CT miss: ',ct_miss)
                CT_fpaths = CT_fpaths[:-1]
                lung_fpaths =lung_fpaths[:-1] 
        else:
            list_PxOk.append(patient_path)
                
    

print(len(CT_fpaths),len(lbl_fpaths),len(lung_fpaths))
CT_fpaths.sort()
lbl_fpaths.sort()
lung_fpaths.sort()

print(CT_fpaths[44])
print(lbl_fpaths[44])
print(lung_fpaths[44])


#Create data dictionat
data_dicts = [
    {"image": image_name,"lung":lung_name,"label": label_name}
    for image_name,lung_name,label_name in zip(CT_fpaths,lung_fpaths,lbl_fpaths)
]
train_files, val_files = data_dicts[:-50], data_dicts[-50:]
print('train val len:',len(train_files),'-',len(val_files))

minmin_CT = -1024 #NBIA
maxmax_CT = 3071 #NBIA


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

#Create Compose functions for preprocessing of train and validation
set_determinism(seed=0)
image_keys = ["image","lung","label"]
p = .5 #Data aug transform probability
size = 96
image_size = (size,size,size)
train_transforms = Compose(
    [
        LoadImaged(keys=image_keys),
        EnsureChannelFirstd(keys=image_keys),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        #Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT,b_min=0.0, b_max=1.0, clip=True,),
        Create_sequences(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key="lung",k_divisible = size),
        #SobelGradientsd(keys=["image"], kernel_size=5, dtype=torch.float32),
        #HistogramNormalized(keys=["image"],num_bins=256, min=0, max=1, dtype='float64'),
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
        #HistogramNormalized(keys=["image"],num_bins=256, min=0, max=1, dtype='float64'),

        ToTensord(keys=image_keys),
    ]
)


#Check the images after the preprocessing
if False:
    check_ds =Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1,num_workers=0)
    if True:
        count = 1
        for batch_data in check_loader:
            #batch_data = first(check_loader)
            image,lung, label = (batch_data["image"][0][0],batch_data["lung"][0][0],batch_data["label"][0][0])
            print(f"px info:{count },image shape: {image.shape},lung shape: {lung.shape}, label shape: {label.shape}")
            count+=1
            for i in range(label.shape[2]):
                if torch.sum(label[:,:,i])>0:
                    plt.subplot(2,3,1),plt.imshow(image[:,:,i]),plt.axis('off')
                    plt.subplot(2,3,2),plt.imshow(label[:,:,i]),plt.axis('off')
                    plt.subplot(2,3,3),plt.imshow(label[:,:,i]+image[:,:,i]),plt.axis('off')
                    plt.subplot(2,3,4),plt.imshow(image[:,:,i+2]),plt.axis('off')
                    plt.subplot(2,3,5),plt.imshow(label[:,:,i+2]),plt.axis('off')
                    plt.subplot(2,3,6),plt.imshow(label[:,:,i+2]+image[:,:,i]),plt.axis('off')
                    plt.tight_layout(),plt.show()
                    break
            if count>10:
                break


#Get the dataset ready for the model
if True:
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)#,collate_fn=pad_list_data_collate)

#Create the model
spatial_dims = 3
max_epochs = 100
in_channels = 1
out_channels=2 #including background

model = SwinUNETR(
    image_size, 
    in_channels, out_channels, 
    use_checkpoint=True, 
    feature_size=24,
    #spatial_dims=spatial_dims
).to(device)


#metrics
dice_metric = DiceMetric(include_background=False, reduction="mean")
surfDice_metric_1Class = SurfaceDiceMetric([10],include_background=False)
hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")


# Load pretrained model
#pretrained_path = '/home/umcg/Desktop/AutomaticITV_code/weights/best_m_MONAI_V3_NBIAWeightsretrainedWithUMCGdata.pth'
pretrained_path = '/data/p308104/MultipleBP/best_m_MONAI_V3_UMCGWeightsretrainedWithUMCGdata_X2RetrainedwithAITVdata.pth'

if pretrained_path is not(None):
    model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(device)))

    #weight = torch.load(pretrained_path, map_location=torch.device(device))
    #model.load_from(weights=weight)
    print('Using pretrained weights!')



#params
#loss_function = DiceLoss(include_background=False,to_onehot_y=True, sigmoid=True)
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)

#Previous optimizer
#optimizer = torch.optim.Adam(model.parameters(), 1e-4)
#New Optimizer LR finder
lower_lr, upper_lr = 1e-5, 1e-0
optimizer = torch.optim.Adam(model.parameters(), lower_lr)
lr_finder = LearningRateFinder(model, optimizer, loss_function, device=device)
lr_finder.range_test(train_loader, val_loader, end_lr=upper_lr, num_iter=20)
steepest_lr, _ = lr_finder.get_steepest_gradient()
ax = plt.subplots(1, 1, figsize=(15, 15), facecolor="white")[1]
_ = lr_finder.plot(ax=ax)


#Live plotting
#This function is just a wrapper around range/trange such that the plots are updated on every iteration.

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

#Training
#The training looks slightly different from a vanilla loop, but this is only because it loops across each of the different #learning rate methods (standard, steepest and cyclical), such that they can be updated simultaneously
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
      
        
        
        
if False:
    post_transforms = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True,to_onehot=out_channels, threshold=.5),
            FillHoles(applied_labels=1, connectivity=0),
            RemoveSmallObjects(min_size=5, connectivity=3, independent_channels=True),
            #KeepLargestConnectedComponent(applied_labels=None,is_onehot=True,connectivity=3, num_components=3),
       ]
    )

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True,threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)],)

    #print(postImg_monai[0].shape)

    #plt.subplot(1,2,1),plt.imshow(val_outputs[0,1,:,:,104].cpu())
    #plt.subplot(1,2,2),plt.imshow(postImg_monai[0][1,:,:,104].cpu())
    #plt.show()


    # In[ ]:


    #compute metric for current iteration
    #Batch x Channel x Height x Width - [B,C,H,W] - 1, 384, 288, 192
    #Channel is number of classes
    def surfDiceFun(val_labels,postImg_monai):
        ylabe_BCHW_neg = val_labels[0].cpu().numpy()
        ylabe_BCHW_neg = ylabe_BCHW_neg*-1+1
        ylabe_BCHW_pos = val_labels[0].cpu().numpy()
        ylabe_BCHW = np.concatenate((ylabe_BCHW_neg, ylabe_BCHW_pos),0)
        transpose_monai = Compose([Transpose([3, 0, 1, 2])])
        ylabe_BCHW = transpose_monai(ylabe_BCHW)
        ypred_BCHW = postImg_monai[0].permute(3, 0, 1, 2)

        list_surf=[]
        for i in range(ypred_BCHW.shape[0]-4):
            surfDice_metric(y_pred=ypred_BCHW[i:i+2,:,:,:], y=ylabe_BCHW[i:i+2,:,:,:])
            #print('SurfFice monai    :',surfDice_metric.aggregate().item())
            list_surf.append(surfDice_metric.aggregate().item())
            surfDice_metric.reset()
        return np.mean(list_surf)

    def surfDiceFun_1Class(val_labels,postImg_monai):
        ylabe_BCHW_neg = val_labels[0].cpu().numpy()
        ylabe_BCHW_neg = ylabe_BCHW_neg*-1+1
        ylabe_BCHW_pos = val_labels[0].cpu().numpy()
        ylabe_BCHW = np.concatenate((ylabe_BCHW_neg, ylabe_BCHW_pos),0)
        transpose_monai = Compose([Transpose([3, 0, 1, 2])])
        ylabe_BCHW = transpose_monai(ylabe_BCHW)
        ypred_BCHW = postImg_monai[0].permute(3, 0, 1, 2)

        #print(ypred_BCHW.shape,ylabe_BCHW.shape)
        list_surf=[]
        for i in range(ypred_BCHW.shape[0]-4):
            if  np.sum(ylabe_BCHW[i:i+2,1,:,:])>0:# or np.sum(ypred_BCHW[i:i+2,0,:,:])>0:
                surfDice_metric_1Class(y_pred=ypred_BCHW[i:i+2,:,:,:], y=ylabe_BCHW[i:i+2,0:,:,:])
                list_surf.append(surfDice_metric_1Class.aggregate().item())
                #print(list_surf[-1])
                surfDice_metric_1Class.reset()
        return np.mean(list_surf)
    #hausdorff_metric(y_pred=postImg_monai, y=val_labels)
    #print('hausdorff monai    :',hausdorff_metric.aggregate().item())
    #hausdorff_metric.reset()


    # In[ ]:





    # In[ ]:


    #Testing the model
    count=0
    nr_images=8
    figsize = (18, 12)
    #figures_folder_i = '/data/p308104/NBIA_Data/NIFTI_NBIA/Res-16092022/' #Peregrine
    figures_folder_i ='/home/umcg/Desktop/AutomaticITV_code/figures_folder_UMCG/' #Local
    all_metrics = []
    model.eval()
    with torch.no_grad():
        for val_data in check_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),)
            roi_size = image_size
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            #postImg_manual  = PostProcessing(val_outputs.cpu())
            #postImg_manual  = [post_transforms(i) for i in decollate_batch(postImg_manual)]

            postImg_monai = [post_transforms(i) for i in decollate_batch(val_outputs)]
            print('read')
            if True: #True if plot
                # Determine slices to be plotted
                # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
                slice_indices = []
                for i in range(val_inputs.shape[4]):
                    if np.sum(val_labels.cpu().numpy()[0, 0, :, :, i])>0:
                        slice_indices.append(i)
                if len(slice_indices) <nr_images:
                    slice_indices.append(random.sample(range(1, 95),nr_images-len(slice_indices))[0])
                else:
                    slice_indices = random.sample(slice_indices, k=nr_images)
                instance = random.randint(0, val_inputs.shape[0] - 1)
                j = 1
                px = val_files[count]['label'].split('/')[-2]
                for i, idx in enumerate(slice_indices):
                    j=1+i
                    fig = plt.figure('Instance = {}'.format(instance), figsize=figsize)
                    plt.subplot(4, nr_images, j),plt.title('CT ({})'.format(idx)),plt.imshow(val_inputs.cpu().numpy()[instance, 0, :, :, idx], cmap='gray', vmin=0, vmax=1),plt.axis('off')
                    plt.subplot(4, nr_images, nr_images+j),plt.title('GTV ({})'.format(idx)),plt.imshow(val_labels.cpu().numpy()[instance, 0, :, :, idx]),plt.axis('off')
                    plt.subplot(4, nr_images, 2*nr_images+j),plt.title('Prediction ({})'.format(idx)),plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[instance, :, :, idx]),plt.axis('off')
                    #plt.subplot(5, nr_images, j),plt.title('postImg Manual ({})'.format(idx)),plt.imshow(postImg_manual[0].detach().cpu()[instance, :, :, idx]),plt.axis('off')
                    plt.subplot(4, nr_images, 3*nr_images+j),plt.title('postImg Monai ({})'.format(idx)),plt.imshow(postImg_monai[0].detach().cpu()[1, :, :, idx]),plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(figures_folder_i,'final_{}.png'.format(px)))
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            #compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice0 = dice_metric.aggregate().item()
            print('Scores vanilla:',dice0)
            dice_metric.reset()

            dice_metric(y_pred=postImg_monai[0][1:2,:,:,:], y=val_labels[0])
            dice1 = dice_metric.aggregate().item()
            print('Scores monai    :',dice1)
            dice_metric.reset()
            sdice0 = surfDiceFun_1Class(val_labels,val_outputs)
            sdice1 = surfDiceFun_1Class(val_labels,postImg_monai)
            print('SurfFice vanilla 1 class:',sdice0)
            print('SurfFice monai 1 class:',sdice1)
            all_metrics.append([val_files[count]['label'].split('/')[-2],dice0,dice1,sdice0,sdice1])
            count+=1

    #Save csv of metrics
    if True: 
        f= open(figures_folder_i+'res_UMCG.csv','w', encoding='UTF8')
        writer = csv.writer(f)
        writer.writerow(['Patient','Dice V','Dice P','Sdice V','SDice P'])
        writer.writerows(all_metrics)
        f.close()


    # In[ ]:


    #Save csv of metrics
    f= open(figures_folder_i+'res','w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(['Patient','Dice V','Dice P','Sdice V','SDice P'])
    writer.writerows(all_metrics)
    f.close()


    # In[ ]:


    #Load csv of metrics

    if False:
        import pandas as pd
        df = pd.read_csv(r'/home/umcg/Desktop/AutomaticITV_code/figures_folder_i/res.csv')
        print(df.mean())
        print(df.std())


    # In[ ]:


    hausdorff_metric(y_pred=postImg_monai, y=val_labels)
      print('hausdorff monai    :',hausdorff_metric.aggregate().item())
      hausdorff_metric.reset()


    # In[ ]:


    from skimage.measure import label, regionprops
    from skimage.morphology import dilation,disk,erosion
    def PostProcessing(predicted):
        img = predicted
        for i in range(predicted.shape[-1]-10):
            j=i+5
            #Get imgs
            Before = predicted[0,1,:,:,j-1]
            Current = predicted[0,1,:,:,j]
            After = predicted[0,1,:,:,j+1]
            AAfter = predicted[0,1,:,:,j+2]+After
            AAfter = predicted[0,1,:,:,j+3]+AAfter
            AAfter = predicted[0,1,:,:,j+4]+AAfter
            AAfter = predicted[0,1,:,:,j+5]+AAfter
            BBefore = predicted[0,1,:,:,j-2]+Before
            BBefore = predicted[0,1,:,:,j-3]+BBefore
            BBefore = predicted[0,1,:,:,j-4]+BBefore
            BBefore = predicted[0,1,:,:,j-5]+BBefore

            #Binarize
            t = .5
            BBefore[BBefore>=t] =int(1)
            BBefore[BBefore<t] =int(0)
            Current[Current>=t] =int(1)
            Current[Current<t] =int(0)
            AAfter[AAfter>=t] =int(1)
            AAfter[AAfter<t] =int(0)

            #Morphology
            m = 15
            BBefore = dilation(BBefore, disk(m))
            Current = dilation(Current, disk(m))
            AAfter = dilation(AAfter, disk(m))
            m = 5
            BBefore = erosion(BBefore, disk(m))
            Current = erosion(Current, disk(m))
            AAfter = erosion(AAfter, disk(m))

            #And Between slices -> <- ... for missing 
            ABS = np.logical_and(BBefore,AAfter)
            Current = np.logical_or(ABS,Current)

            #And outside slices <-->  ... for false positives with RegionProps
            BBefore_label = label(BBefore)
            Current_label = label(Current)
            AAfter_label = label(AAfter)
            BBefore_regions = regionprops(BBefore_label)
            Current_regions = regionprops(Current_label)
            AAfter_regions = regionprops(AAfter_label)
            #Look for centroids distance
            limit =  20
            if False:
                for c_centroid in Current_regions:
                    for b_centroid in BBefore_regions:
                        dif = np.abs(np.asarray(c_centroid['centroid'])-np.asarray(b_centroid['centroid']))
                        if dif[0] > limit or dif[1] > limit: 
                            Current[Current==c_centroid['label']] = int(0)
                            print('delete blob')
                    for a_centroid in AAfter_regions:
                        dif = np.abs(np.asarray(c_centroid['centroid'])-np.asarray(a_centroid['centroid']))
                        if dif[0] > limit or dif[1] > limit: 
                            Current[Current==c_centroid['label']] = int(0)
                            print('delete blob')
            if False:
                plt.subplot(1,4,1),plt.imshow(BBefore,cmap='gray', vmin=0, vmax=1)
                plt.subplot(1,4,2),plt.imshow(Current,cmap='gray', vmin=0, vmax=1)
                plt.subplot(1,4,3),plt.imshow(predicted[0,1,:,:,j],cmap='gray', vmin=0, vmax=1)
                plt.subplot(1,4,4),plt.imshow(AAfter,cmap='gray', vmin=0, vmax=1)
                plt.show()

            img[0,0,:,:,j] = torch.from_numpy(Current)
        return img



    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:




