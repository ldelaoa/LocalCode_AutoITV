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
from skimage.measure import label, regionprops
from skimage.morphology import dilation,disk,erosion


from monai.utils import first, set_determinism
from monai.transforms import (
    RandFlipd,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    FillHoles,
    RemoveSmallObjects,
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
    RandWeightedCropd,
    ToTensord,
    Transpose,
    ToTensor,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet,VNet,SwinUNETR,UNETR,DynUNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric,SurfaceDiceMetric,SurfaceDistanceMetric,HausdorffDistanceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,pad_list_data_collate
from monai.config import print_config
from monai.apps import download_and_extract


def LookSortFiles(root_path, all_patientdir):
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

    return CT_fpaths, lbl_fpaths, lung_fpaths


# class to transpose lung mask
class Create_sequences(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        print(f"keys to transpose: {self.keys}")

    def __call__(self, dictionary):
        dictionary = dict(dictionary)
        for key in self.keys:
            data = dictionary[key]
            if key == 'lung':
                data = np.transpose(data, (0, 2, 3, 1))
                data = rotate(data, 270, axes=(1, 2), reshape=False)
                data = np.flip(data, 1)
                data[data == 2] = int(1)
                data[data != 1] = int(0)
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


# compute metric for current iteration
# Batch x Channel x Height x Width - [B,C,H,W] - 1, 384, 288, 192
# Channel is number of classes
def surfDiceFun_1Class(val_labels, postImg_monai):
    surfDice_metric_1Class = SurfaceDiceMetric([3], include_background=False)

    ylabe_BCHW_neg = val_labels[0].cpu().numpy()
    ylabe_BCHW_neg = ylabe_BCHW_neg * -1 + 1
    ylabe_BCHW_pos = val_labels[0].cpu().numpy()
    ylabe_BCHW = np.concatenate((ylabe_BCHW_neg, ylabe_BCHW_pos), 0)
    transpose_monai = Compose([Transpose([3, 0, 1, 2])])
    ylabe_BCHW = transpose_monai(ylabe_BCHW)
    ypred_BCHW = postImg_monai[0].permute(3, 0, 1, 2)

    # print(ypred_BCHW.shape,ylabe_BCHW.shape)
    list_surf = []
    for i in range(ypred_BCHW.shape[0] - 4):
        if np.sum(ylabe_BCHW[i:i + 2, 1, :, :]) > 0:  # or np.sum(ypred_BCHW[i:i+2,0,:,:])>0:
            surfDice_metric_1Class(y_pred=ypred_BCHW[i:i + 2, :, :, :], y=ylabe_BCHW[i:i + 2, 0:, :, :])
            list_surf.append(surfDice_metric_1Class.aggregate().item())
            # print(list_surf[-1])
            surfDice_metric_1Class.reset()
    return np.mean(list_surf)


# hausdorff_metric(y_pred=postImg_monai, y=val_labels)
# print('hausdorff monai    :',hausdorff_metric.aggregate().item())
# hausdorff_metric.reset()


def test(figures_folder_i,image_size,model, val_loader,device,val_files):
    #Define PostTranforms
    out_channels = 2  # including background
    post_transforms = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True, to_onehot=out_channels, threshold=.1),
            FillHoles(applied_labels=1, connectivity=0),
            RemoveSmallObjects(min_size=5, connectivity=3, independent_channels=True),
            # KeepLargestConnectedComponent(applied_labels=None,is_onehot=True,connectivity=3, num_components=3),
        ]
    )
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)], )

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")

    # Testing the model
    count = 0
    nr_images = 8
    figsize = (18, 12)
    all_metrics = []

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),)
            roi_size = image_size
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            px = val_data["image"].meta["filename_or_obj"][0].split('/')[-2]
            newGTV_name = val_data["image"].meta["filename_or_obj"][0][:-9] + 'aGTV.nii.gz'
            bp = val_data["image"].meta["filename_or_obj"][0].split('%')[-2][-2:]
            val_outPost = [post_transforms(i) for i in decollate_batch(val_outputs)]

            print('new GTV name:',newGTV_name)

            if True:  # AXIAL
                slice_indices = []
                for i in range(val_inputs.shape[4]):
                    if np.sum(val_outputs[0][0, :, :, i]) > 0:
                        slice_indices.append(i)
                if len(slice_indices) < nr_images:
                    slice_indices.append(
                        random.sample(range(1, val_inputs.shape[4]), nr_images - len(slice_indices))[0])
                else:
                    slice_indices = random.sample(slice_indices, k=nr_images)
                instance = random.randint(0, val_inputs.shape[0] - 1)
                j = 1
                for i, idx in enumerate(slice_indices):
                    j = 1 + i
                    fig = plt.figure('Instance = {}'.format(instance), figsize=figsize)
                    plt.subplot(3, nr_images, j), plt.title('AXIAL CT ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, :, :, idx], cmap='gray', vmin=0, vmax=1), plt.axis('off')
                    plt.subplot(3, nr_images, nr_images + j), plt.title('Prediction ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, :, :, idx] + val_outputs[0].detach().cpu()[instance, :, :,
                                                                           idx]), plt.axis('off')
                    plt.subplot(3, nr_images, 2 * nr_images + j), plt.title(
                        'postImg Monai ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, :, :, idx] + val_outPost[0].detach().cpu()[instance, :,
                                                                           :, idx]), plt.axis('off')
                plt.tight_layout(), plt.show()
                if not os.path.exists(os.path.join(figures_folder_i, px)):
                    os.makedirs(os.path.join(figures_folder_i, px))
                plt.savefig(os.path.join(figures_folder_i, px, bp + 'AXIAL final_{}.png'.format(px)))
            if True:  # SAGITAL
                slice_indices = []
                for i in range(val_inputs.shape[3]):
                    if np.sum(val_outputs[0][0, :, i, :]) > 0:
                        slice_indices.append(i)
                if len(slice_indices) < nr_images:
                    slice_indices.append(
                        random.sample(range(1, val_inputs.shape[3]), nr_images - len(slice_indices))[0])
                else:
                    slice_indices = random.sample(slice_indices, k=nr_images)
                instance = random.randint(0, val_inputs.shape[0] - 1)
                j = 1
                for i, idx in enumerate(slice_indices):
                    j = 1 + i
                    fig = plt.figure('Instance = {}'.format(instance), figsize=figsize)
                    plt.subplot(3, nr_images, j), plt.title('SAGITAL CT ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, :, idx, :], cmap='gray', vmin=0, vmax=1), plt.axis('off')
                    plt.subplot(3, nr_images, nr_images + j), plt.title('Prediction ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, :, idx, :] + val_outputs[0].detach().cpu()[instance, :,
                                                                           idx, :]), plt.axis('off')
                    plt.subplot(3, nr_images, 2 * nr_images + j), plt.title(
                        'postImg Monai ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, :, idx, :] + val_outPost[0].detach().cpu()[instance, :,
                                                                           idx, :]), plt.axis('off')
                plt.tight_layout(), plt.show()
                if not os.path.exists(os.path.join(figures_folder_i, px)):
                    os.makedirs(os.path.join(figures_folder_i, px))
                plt.savefig(os.path.join(figures_folder_i, px, bp + 'SAGITAL final_{}.png'.format(px)))
            if True:  # CORONAL
                slice_indices = []
                for i in range(val_inputs.shape[2]):
                    if np.sum(val_outputs[0][0, i, :, :]) > 0:
                        slice_indices.append(i)
                if len(slice_indices) < nr_images:
                    slice_indices.append(
                        random.sample(range(1, val_inputs.shape[2]), nr_images - len(slice_indices))[0])
                else:
                    slice_indices = random.sample(slice_indices, k=nr_images)
                instance = random.randint(0, val_inputs.shape[0] - 1)
                j = 1
                for i, idx in enumerate(slice_indices):
                    j = 1 + i
                    fig = plt.figure('Instance = {}'.format(instance), figsize=figsize)
                    plt.subplot(3, nr_images, j), plt.title('CORONAL CT ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, idx, :, :], cmap='gray', vmin=0, vmax=1), plt.axis('off')
                    plt.subplot(3, nr_images, nr_images + j), plt.title('Prediction ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, idx, :, :] + val_outputs[0].detach().cpu()[instance, idx,
                                                                           :, :]), plt.axis('off')
                    plt.subplot(3, nr_images, 2 * nr_images + j), plt.title(
                        'postImg Monai ({})'.format(idx)), plt.imshow(
                        val_inputs.cpu().numpy()[instance, 0, idx, :, :] + val_outPost[0].detach().cpu()[instance,
                                                                           idx, :, :]), plt.axis('off')
                plt.tight_layout()
                plt.show()
                if not os.path.exists(os.path.join(figures_folder_i, px)):
                    os.makedirs(os.path.join(figures_folder_i, px))
                plt.savefig(os.path.join(figures_folder_i, px, bp + 'CORONAL final_{}.png'.format(px)))

            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            # compute metric for current iteration

            #DICE
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice0 = dice_metric.aggregate().item()
            print('Scores vanilla:', dice0)
            dice_metric.reset()

            dice_metric(y_pred=val_outPost, y=val_labels) #[0][1:2, :, :, :] [0]
            dice1 = dice_metric.aggregate().item()
            print('Scores post    :', dice1)
            dice_metric.reset()

            #SURFDICE
            sdice0 = surfDiceFun_1Class(val_labels, val_outputs)
            sdice1 = surfDiceFun_1Class(val_labels, val_outPost)
            print('SurfFice vanilla 1 class:', sdice0)
            print('SurfFice post 1 class:', sdice1)

            #ALL TOGETHER
            all_metrics.append([val_files[count]['label'].split('/')[-2], dice0, dice1, sdice0, sdice1])
            count += 1

            return all_metrics

def main(figures_folder_i,name_run,pretrained_path,cache,num_workers,root_path):

    all_patientdir = []
    all_patientdir = os.listdir(root_path)
    all_patientdir.sort()
    print(len(all_patientdir),'in',name_run)

    CT_fpaths, lbl_fpaths, lung_fpaths = LookSortFiles(root_path, all_patientdir)


    #Create data dictionat
    data_dicts = [
        {"image": image_name,"lung":lung_name,"label": label_name}
        for image_name,lung_name,label_name in zip(CT_fpaths,lung_fpaths,lbl_fpaths)
    ]
    val_files =  data_dicts[:]
    print('train val len:',0,'-',len(val_files))

    minmin_CT = -1024 #NBIA
    maxmax_CT = 3071 #NBIA

    #Create Compose functions for preprocessing of train and validation
    set_determinism(seed=0)
    image_keys = ["image","lung","label"]
    p = .5 #Data aug transform probability
    size = 96
    image_size = (size,size,size)
    pin_memory = True if num_workers > 0 else False  # Do not change


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

    # Check the images after the preprocessing
    if cache:  # Cache
        #train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
        #train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0,
                              num_workers=int(num_workers // 2))
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=int(num_workers // 2), pin_memory=pin_memory)
    else:
        #train_ds = Dataset(data=train_files, transform=train_transforms)
        #train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)  # ,collate_fn=pad_list_data_collate)

    # Create the model
    spatial_dims = 3
    max_epochs = 250
    in_channels = 1
    out_channels = 2  # including background
    lr = 1e-3  # 1e-4
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

    #metrics, no definition of :
    #NO Loss Function
    #NO Optimizer


    # Load pretrained model
    if pretrained_path is not(None):
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(device)))
        print('Using pretrained weights!')

    all_metrics = test(figures_folder_i ,image_size,model, val_loader,device,val_files)


    #Save csv of metrics
    if True:
        f= open(figures_folder_i+'res_UMCGx2.csv','w', encoding='UTF8')
        writer = csv.writer(f)
        writer.writerow(['Patient','Dice V','Dice P','Sdice V','SDice P'])
        writer.writerows(all_metrics)
        f.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    num_workers = 12

    # peregrine
    # root_path = '/data/p308104/NBIA_Data/NIFTI_NBIA/imagesTr/'
    # root_path = '/data/p308104/Nifti_Imgs_V0/' #UMCG data on peregrine
    #root_path = '/data/p308104/MultipleBP/'
    # local
    # root_path = '/home/umcg/Desktop/NBIA/NBIA_Nifti_v0/'
    # root_path = '/home/umcg/Desktop/Dicom_UMCG/MultipleBP/'
    root_path = '/home/umcg/OneDrive/MultipleBreathingP/'
    # root_path = '/home/umcg/Desktop/AutomaticITV_code/MultipleBreathingP-OneDriveCopy/MultipleBreathingP/'

    #figures_folder_i = '/data/p308104/MultipleBP_Res/'  # Peregrine
    figures_folder_i = '/home/umcg/Desktop/AutomaticITV_code/figures_folder_i/figures_folder_UMCG_v9/DynU_l3_v9/'
    pretrained_path = '/home/umcg/Desktop/AutomaticITV_code/weights/v9/best_DynUnet_V9_UMCG_Loss3.pth'

    cache = False
    lf_select = None # NOT Needed for testing
    print('LF is ', lf_select, 'Db is UMCG', 'DynUnet')
    name_run = "DynUnet" + "LF" + str(lf_select) + "run1"
    main(figures_folder_i,name_run, pretrained_path, cache, num_workers, root_path)

# Changes from NBIA versions are:
# 1- patients file seeker function optimized for umcg db
# 2- lr is higher and eta_min lower, aiming for a broader area on where to look
# 3- WandB project name is now UMCG_V9
# 4- Training with all the images, not anymore the 75-25 ratio
# 5- Max Epochs is 250 now

