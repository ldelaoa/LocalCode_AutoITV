import os
import sys
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
from lungmask import mask as lungmask_fun
from skimage.measure import label, regionprops
from skimage.morphology import dilation,ball,erosion
from monai.utils import first, set_determinism
from monai.transforms import (
    RandFlipd,
    MaskIntensityd,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    FillHoles,
    #RemoveSmallObjects,
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
    ScaleIntensity,
)
from monai.networks.nets import SwinUNETR,DynUNet
from monai.metrics import DiceMetric,SurfaceDiceMetric,SurfaceDistanceMetric,HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch


def CreateLungMasks(root_path, CT_fpaths):
    # Get Lung mask and save it
    CT_path0 = CT_fpaths[0]
    CT_nii = nib.load(CT_path0)
    for ct in CT_fpaths:
        empty_header = nib.Nifti1Header()
        lung_path = ct[:-10] + '_LungMask.nii.gz'
        print('Creating Lung Mask: ', lung_path)
        input_image = sitk.ReadImage(ct, imageIO='NiftiImageIO')
        lungmask = lungmask_fun.apply(input_image)  # default model is U-net(R231)
        lungmask_ni = nib.Nifti1Image(lungmask, CT_nii.affine, empty_header)
        nib.save(lungmask_ni, lung_path)
    return 0

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
                if True:  # 4D Data local
                    if "ct.nii.gz" in f.lower() and not("ave" in f.lower()):
                        CT_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss = False
                    if "_gtv" in f.lower():
                        lbl_fpaths.append(os.path.join(root_path, patient_path, f))
                        gtv_miss = False
                    if "lungmask.nii.gz" in f.lower() and not("ave" in f.lower()):
                        lung_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss = False
            if gtv_miss:
                for f in files:
                    if "_igtv" in f.lower():
                        lbl_fpaths.append(os.path.join(root_path, patient_path, f))
                        gtv_miss = False
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
                    if "ct.nii.gz" in f.lower() and ct_miss:
                        CT_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss = False
                    if 'lungmask.nii.gz' in f.lower() and lung_miss:
                        lung_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss = False
            if gtv_miss and len(files) > 0:
                CT_fpaths.pop()
                ct_miss = True
                lung_fpaths.pop()
                lung_miss = True

    print('ct: ', len(CT_fpaths), 'label: ', len(lbl_fpaths), 'lung: ', len(lung_fpaths))
    if False:
        CreateLungMasks(root_path, CT_fpaths)
        print('Rerun the program')
        exit(1)

    CT_fpaths.sort()
    lbl_fpaths.sort()
    lung_fpaths.sort()

    print(CT_fpaths[-1])
    print(lbl_fpaths[-1])
    print(lung_fpaths[-1])


    if (len(CT_fpaths) != len(lbl_fpaths)) or (len(lbl_fpaths) != len(lung_fpaths)):
        print('Different number of files for each structure')
        exit(1)

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

def test(figures_folder_i,image_size,SelectModel,modelSwin,modelDyn, val_loader,device,val_files):
    #Define PostTranforms

    post_transforms = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True, threshold=0.8),
            ScaleIntensity(minv=0.0, maxv=1.0),
            KeepLargestConnectedComponent(applied_labels=None, is_onehot=False, connectivity=2, num_components=1),
        ]
    )
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, threshold=0.1),
                         ScaleIntensity(minv=0.0, maxv=1.0)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)], )

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    surfDice_metric191 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 191), include_background=False)
    surfDice_metric287 = SurfaceDiceMetric(class_thresholds=np.linspace(3,3,287), include_background=False)
    surfDice_metric383 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 383), include_background=False)

    # Testing the model
    count = 0
    figsize = (18, 8)
    all_metrics = []
    full_volume_disp = False
    metrics=True
    ListTP = []
    ListFN = []
    ListFP = []
    sumTP = 0
    sumFN = 0
    sumFP = 0
    sum_voxTP = 0
    sum_voxFN = 0
    sum_voxFP = 0
    print("Len Valloader: ",len(val_loader))
    ## 0 Swin  - 1 Dyn - 2 SwinDyn
    if SelectModel!=1:
        modelSwin.eval()
    if SelectModel!=0:
        modelDyn.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),)
            roi_size = image_size
            sw_batch_size = 1
            px = val_data["image"].to('cpu').meta["filename_or_obj"][0].split('/')[-2]
            print('Px: ', px)

            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            if SelectModel==0:
                val_outputs_Swin = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelSwin)
                val_outPost_Swin = [post_transforms(i) for i in decollate_batch(val_outputs_Swin)]
                val_outPost = val_outPost_Swin[0]
            if SelectModel==1:
                val_outputs_Dyn = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelDyn)
                val_outPost_Dyn = [post_transforms(i) for i in decollate_batch(val_outputs_Dyn)]
                val_outPost = val_outPost_Dyn[0]
            if SelectModel==2:
                val_outputs_Swin = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelSwin)
                val_outPost_Swin = [post_transforms(i) for i in decollate_batch(val_outputs_Swin)]
                val_outputs_Dyn = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelDyn)
                val_outPost_Dyn = [post_transforms(i) for i in decollate_batch(val_outputs_Dyn)]
                val_outPost = torch.logical_or(val_outPost_Swin[0], val_outPost_Dyn[0])

            image = val_inputs[0].detach().cpu().numpy()
            image = image.squeeze()

            lbl_3dnp = val_labels[0].detach().cpu().numpy()
            lbl_3dnp = lbl_3dnp.squeeze()
            lbl_3dnp = dilation(lbl_3dnp, ball(2))

            label_img = label(lbl_3dnp)
            props = regionprops(label_img)

            print("num de blobs in label: ", len(props))
            for i in range(len(props)):
                r = props[i]
                attrs = [a for a in r.__dir__() if not a.startswith('_')]
                print('label bbox ', r.bbox, "size", len(r.coords))
                lbl_bbox = r.bbox

            out_3dnp = val_outPost[0].detach().cpu().numpy()
            out_3dnp = out_3dnp.squeeze()
            out_3dnp = dilation(out_3dnp, ball(2))

            label_out_3dnp = label(out_3dnp)
            props = regionprops(label_out_3dnp)

            print("num de blobs predicted: ", len(props))
            for n in range(len(props)):
                r = props[n]
                voxTP = 0
                attrs = [a for a in r.__dir__() if not a.startswith('_')]
                print('prediction bbox', r.bbox, "size", len(r.coords))

                patch = np.zeros(out_3dnp.shape)
                for j in range(len(r.coords)):
                    patch[r.coords[j][0], r.coords[j][1], r.coords[j][2]] = 1

                TP = False
                for j in range(len(r.coords)):
                    if (lbl_bbox[0] < r.coords[j][0] < lbl_bbox[3]) and (
                            lbl_bbox[1] < r.coords[j][1] < lbl_bbox[4]) and (
                            lbl_bbox[2] < r.coords[j][2] < lbl_bbox[5]):
                        TP = True
                        voxTP += 1

                # Create different matrixes, one for each blob to send to metrics
                predicted_blobn = np.zeros(out_3dnp.shape)
                predicted_blobn[label_out_3dnp == n + 1] = 1
                predicted_blobn4d = np.expand_dims(predicted_blobn, 0)
                tensor_blobn = torch.from_numpy(predicted_blobn4d)

                lbl_4d = np.expand_dims(lbl_3dnp, 0)
                tensor_label = torch.from_numpy(lbl_4d)

                sumPredicted = np.sum(predicted_blobn)
                sumGroundT = np.sum(lbl_3dnp)
                voxFP = sumPredicted - voxTP
                voxFN = sumGroundT - voxTP

                sum_voxTP += voxTP
                sum_voxFN += voxFN
                sum_voxFP += voxFP

                print('True Positive: ', TP)
                
                # compute metric for current iteration
                if metrics:
                    # DICE
                    dice_metric(y_pred=tensor_blobn, y=tensor_label)  # [0][1:2, :, :, :] [0]
                    dice1 = dice_metric.aggregate().item()
                    print('Scores post    :', dice1)
                    dice_metric.reset()

                    hausdorff_metric(y_pred=tensor_blobn, y=tensor_label)
                    hausd1 = hausdorff_metric.aggregate().item()
                    hausdorff_metric.reset()
                    print('Haus post 1 class:', hausd1)
                    #print("Shape:",tensor_blobn.shape,tensor_blobn[0].shape[1])
                    if tensor_blobn.shape[1]==288:
                        surfDice_metric287(y_pred=tensor_blobn, y=tensor_label)
                        sdice1 = surfDice_metric287.aggregate().item()
                        surfDice_metric287.reset()
                    elif tensor_blobn.shape[1]==192:
                        surfDice_metric191(y_pred=tensor_blobn, y=tensor_label)
                        sdice1 = surfDice_metric191.aggregate().item()
                        surfDice_metric191.reset()
                    else:
                        surfDice_metric383(y_pred=tensor_blobn, y=tensor_label)
                        sdice1 = surfDice_metric383.aggregate().item()
                        surfDice_metric383.reset()
                    print('SurfFice post 1 class:', sdice1)

                FN=False
                FP=False
                if TP:
                    sumTP = sumTP + 1
                    ListTP.append(px)
                if not TP:
                    sumFN = sumFN + 1
                    ListFN.append(px)
                    FN=True
                if len(props) > 1:
                    sumFP = sumFP + 1
                    ListFP.append(px)
                    FP=True


                #ALL TOGETHER
                all_metrics.append([val_files[count]['label'].split('/')[-2], dice1, sdice1,hausd1,TP,FN,FP,voxTP,voxFP,voxFN])
                
                # Plot all images
                if full_volume_disp:
                    count = 0
                    for i in range(lbl_3dnp.shape[2]):
                        if (np.sum(lbl_3dnp[:, :, i], ) > 0) or (np.sum(predicted_blobn[:, :, i], ) > 0):
                            fig = plt.figure('Instance = {}'.format(0), figsize=figsize)
                            ax = fig.add_subplot(121)
                            ax.imshow(np.rot90(image[:, :, i]), cmap='gray'), plt.axis('off')
                            ax = fig.add_subplot(122)
                            ax.imshow(np.rot90(image[:, :, i]), cmap='gray'), plt.axis('off')
                            ax.contour(np.rot90(lbl_3dnp[:, :, i]), colors='yellow')
                            if n ==0:
                                ax.contour(np.rot90(predicted_blobn[:, :, i]), colors='blue')
                            else:
                                ax.contour(np.rot90(predicted_blobn[:, :, i]), colors='red')
                            ax.text(8, 10, 'Yellow Label', style='normal', color='white', fontsize=15)
                            ax.text(8, 25, 'Red/Blue Prediction', style='normal', color='white', fontsize=15)
                            # plt.show()

                            if not os.path.exists(os.path.join(figures_folder_i, px)):
                                os.makedirs(os.path.join(figures_folder_i, px))

                            plt.savefig(os.path.join(figures_folder_i, px, 'FullV_final_V10{}.png'.format(i)))
                            plt.clf()

                            # newGTV_name = val_data["image"].meta["filename_or_obj"][0][:-9] + 'aGTV.nii.gz'
                            # bp = 'NoBpSpec'#val_data["image"].meta["filename_or_obj"][0].split('%')[-2][-2:]
                            # print('new GTV name:',newGTV_name)
            count += 1
        if sumTP ==0 :
            print("Sensitivity: ", 0)
            print("Precision: ", 0)
            print(sumTP, sumFN, sumFP)
        else:
            print("Sensitivity: ", sumTP / (sumTP + sumFN))
            print("Precision: ", sumTP / (sumTP + sumFP))
            print(sumTP, sumFN, sumFP)

        return all_metrics


def main(spec_patient,SelectModel,figures_folder_i,name_run,pretrained_path_Swin,pretrained_path_Dyn,cache,num_workers,root_path):

    all_patientdir = []
    all_patientdir = os.listdir(root_path)
    all_patientdir.sort()
    print(len(all_patientdir),'in',name_run)

    CT_fpaths, lbl_fpaths, lung_fpaths = LookSortFiles(root_path, all_patientdir)

    # SPECIFIC PX Testing:

    if spec_patient is not (None):
        for i in range(len(CT_fpaths)):
            if spec_patient in CT_fpaths[i]:
                print('Specified patient found')
                spec_CT_fpaths = CT_fpaths[i]
                spec_lbl_fpaths = lbl_fpaths[i]
                spec_lung_fpaths = lung_fpaths[i]
                break
        CT_fpaths = []
        lbl_fpaths = []
        lung_fpaths = []
        CT_fpaths.append(spec_CT_fpaths)
        lbl_fpaths.append(spec_lbl_fpaths)
        lung_fpaths.append(spec_lung_fpaths)

    #Create data dictionat
    data_dicts = [
        {"image": image_name,"lung":lung_name,"label": label_name}
        for image_name,lung_name,label_name in zip(CT_fpaths,lung_fpaths,lbl_fpaths)
    ]
    val_files = data_dicts[:]
    print('train val len:',0,'-',len(val_files))

    minmin_CT = -1024 #NBIA
    maxmax_CT = 200 #NBIA

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
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT, b_min=0.0, b_max=1.0, clip=True, ),
            Create_sequences(keys=image_keys),
            CropForegroundd(keys=image_keys, source_key="lung", k_divisible=size),
            MaskIntensityd(keys=["image"], mask_key="lung"),
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

    #Plot Input Volume
    #plotInputVolume(val_loader, device)

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
    if SelectModel==0:
        print("MODEL SWIN")
        modelSwin = SwinUNETR(
            image_size,
            in_channels, out_channels,
            use_checkpoint=True,
            feature_size=48,
            #spatial_dims=spatial_dims
        ).to(device)
        modelDyn = None
    if SelectModel==1:
        print("MODEL Dyn")
        task_id = "06"
        modelDyn = DynUNet(
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
        modelSwin = None
    if SelectModel==2:
        print("MODEL SwinDyn")
        print("MODEL SWIN")
        modelSwin = SwinUNETR(
            image_size,
            in_channels, out_channels,
            use_checkpoint=True,
            feature_size=48,
            # spatial_dims=spatial_dims
        ).to(device)
        task_id = "06"
        modelDyn = DynUNet(
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
    if pretrained_path_Swin is not(None) and SelectModel!=1:
        modelSwin.load_state_dict(torch.load(pretrained_path_Swin, map_location=torch.device(device)))
        print('Using Swin pretrained weights!')
    if pretrained_path_Dyn is not(None) and SelectModel!=0:
        modelDyn.load_state_dict(torch.load(pretrained_path_Dyn, map_location=torch.device(device)))
        print('Using Dyn pretrained weights!')
    all_metrics = test(figures_folder_i ,image_size,SelectModel,modelSwin,modelDyn, val_loader,device,val_files)


    #Save csv of metrics
    if True:
        f= open(figures_folder_i+"Model_"+str(SelectModel)+'res_UMCGx10_SwinDyn_forv11.csv','w', encoding='UTF8')
        writer = csv.writer(f)
        writer.writerow(['Patient','Dice','SDice','HausD','TP','FN','FP','voxTP','voxFP','voxFN'])
        writer.writerows(all_metrics)
        f.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    num_workers = 0

    # root_path = '/data/p308104/Nifti_Imgs_V0/' #UMCG data on peregrine
    #root_path = '/data/p308104/MultipleBP/'
    #root_path = '/home/umcg/OneDrive/MultipleBreathingP/'
    root_path = '/home/umcg/Desktop/AutomaticITV_code/SABR1322_Nifti/'
    preweight_path = '/home/umcg/Desktop/AutomaticITV_code/weights/v12/'
    figures_path = '/home/umcg/Desktop/AutomaticITV_code/figures_folder_i/'

    cache = False
    lf_select = None # NOT Needed for testing
    #spec_patient = '0492498'
    if len(sys.argv)>1:
        spec_patient = sys.argv[1]
        print("Patient spec is:", spec_patient)
    else:
        spec_patient = None
        print("No Spec Patient")

    SelectModel = 0 ## 0 Swin  - 1 Dyn - 2 SwinDyn
    figures_folder_i = figures_path + 'figures_SwinDyn_v10_forv11/'
    pretrained_path_Swin = preweight_path + 'best_SwinUnet_V11_UMCG_TestSet3.pth'
    pretrained_path_Dyn = None
    name_run = "NameRun Swin" + str(SelectModel) + "LF" + str(lf_select) + "run0"
    print(name_run)
    main(spec_patient, SelectModel, figures_folder_i, name_run, pretrained_path_Swin,pretrained_path_Dyn, cache, num_workers, root_path)

    SelectModel = 1  ## 0 Swin  - 1 Dyn - 2 SwinDyn
    figures_folder_i = figures_path + 'figures_SwinDyn_v10_forv11/'
    pretrained_path_Swin = None
    pretrained_path_Dyn = preweight_path + 'best_DynUnet_V11_UMCG_TestSet3.pth'
    name_run = "NameRun Dyn" + str(SelectModel) + "LF" + str(lf_select) + "run0"
    print(name_run)
    main(spec_patient, SelectModel, figures_folder_i, name_run, pretrained_path_Swin, pretrained_path_Dyn, cache,
         num_workers, root_path)

    SelectModel = 2  ## 0 Swin  - 1 Dyn - 2 SwinDyn
    figures_folder_i = figures_path + 'figures_SwinDyn_v10_forv11/'
    pretrained_path_Swin = preweight_path + 'best_SwinUnet_V11_UMCG_TestSet3.pth'
    pretrained_path_Dyn = preweight_path + 'best_DynUnet_V11_UMCG_TestSet3.pth'
    name_run = "NameRun SwinDyn" + str(SelectModel) + "LF" + str(lf_select) + "run0"
    print(name_run)
    main(spec_patient, SelectModel, figures_folder_i, name_run, pretrained_path_Swin, pretrained_path_Dyn, cache,
         num_workers, root_path)

    print("The End")

#To test v6
#rotate images when saving - Pending
#just a line, not an area , on and off - possible with python ? Alessias GUI?
#find FP and FN

#To train v10
#adapt to window level ? - what does that mean ?
#preprocessing to increase the contrast inside the lung - 1000 HU  - Pending

#Extra
#Test Lung delineation

#Test v6 must include dice score of all blobs detected ?
#Not only to the main blob, how to do that ? idk haha
