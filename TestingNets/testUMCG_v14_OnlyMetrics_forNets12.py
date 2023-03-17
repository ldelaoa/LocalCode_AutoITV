
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
from lungmask import mask as lungmask_fun
from skimage.measure import label, regionprops,shannon_entropy
from skimage.morphology import dilation,ball,erosion,remove_small_objects

from monai.utils import first, set_determinism
from monai.transforms import (
    AddChannel,
    EnsureChannelFirst,
    ResizeWithPadOrCropd,
    MaskIntensityd,
    ScaleIntensityd,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    ResizeWithPadOrCrop,
    LoadImaged,
    Orientationd,
    ThresholdIntensity,
    RemoveSmallObjects,
    KeepLargestConnectedComponent,
    RandCropByPosNegLabeld,
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
from monai.networks.nets import UNet,VNet,SwinUNETR,UNETR,DynUNet
from monai.metrics import DiceMetric,SurfaceDiceMetric,HausdorffDistanceMetric,compute_surface_dice
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,pad_list_data_collate


def saveMetrics(path_to_save,file_to_save,all_metric_rows):
    print("Saving Metrics for Rows: ",len(all_metric_rows))
    f = open(path_to_save+file_to_save, 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(['Patient', 'Dice', 'SDice', 'HausD', 'Seg Sens','Seg PRec','TP', 'Area Predic', 'Area Lab'])
    for i in range(len(all_metric_rows)):
        writer.writerows(all_metric_rows[i])
    f.close()
    return 0


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



def LookSortFiles(root_path, all_patientdir):
    CTALL_fpaths =[]
    lungALL_fpaths =[]
    itv_fpaths = []
    gtv_fpaths = []  
    predicted_fpaths_yes = []
    predicted_fpaths_no = []
    for patient_path in all_patientdir:
        ct_miss = 0
        gtv_miss = 0
        itv_miss = 0
        lung_miss = 0
        predicted_miss_yes = 0
        predicted_miss_no =0
        print(patient_path)
        for root, dirs, files in os.walk(root_path + patient_path, topdown=False):
            for f in files:
                if "predicteditv_allbp_yes" in f.lower():
                    predicted_fpaths_yes.append(os.path.join(root_path, patient_path, f))
                    predicted_miss_yes+=1
                if "predicteditv_allbp_no" in f.lower():
                    predicted_fpaths_no.append(os.path.join(root_path, patient_path, f))
                    predicted_miss_no+=1
                if "_gtv" in f.lower():
                    gtv_fpaths.append(os.path.join(root_path, patient_path, f))
                    gtv_miss +=1
                if "pproved" in f.lower():
                    itv_fpaths.append(os.path.join(root_path, patient_path, f))
                    itv_miss +=1
                if "50%" in f.lower() and not("ave" in f.lower()):
                    if "ct" in f.lower():                            
                        CTALL_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss +=1
                    if "lung" in f.lower():
                        lungALL_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss +=1

    print('ct: ',ct_miss,"Lungs: ",lung_miss, "Label: ",itv_miss,"pITV yes",predicted_miss_yes,"pITV no",predicted_miss_no)
    CTALL_fpaths = np.sort(CTALL_fpaths)
    lungALL_fpaths = np.sort(lungALL_fpaths)
    itv_fpaths  = np.sort(itv_fpaths)
    gtv_fpaths = np.sort(gtv_fpaths)
    predicted_fpaths_yes = np.sort(predicted_fpaths_yes)
    predicted_fpaths_no = np.sort(predicted_fpaths_no)
    return CTALL_fpaths, itv_fpaths,gtv_fpaths, lungALL_fpaths, predicted_fpaths_yes,predicted_fpaths_no



def postITV(ITV_tensor,binarize,dilation_bool):
    tensor_blobn=[]
    #Delete outliers of slices
    out_3dnp = ITV_tensor.detach().cpu().numpy()
    out_3dnp = out_3dnp.squeeze()
    if binarize:
        out_3dnp[out_3dnp>=1] = 1
        out_3dnp[out_3dnp<1] = 0
    if dilation_bool:
        out_3dnp = dilation(out_3dnp, ball(3))
    label_out_3dnp = label(out_3dnp)
    props = regionprops(label_out_3dnp)
    print("Num Blobs: ",len(props))
    if len(props)>1:
        for n in range(len(props)):
            r = props[n]
            predicted_blobn = np.zeros(out_3dnp.shape)
            predicted_blobn[label_out_3dnp==n+1]=1
            predicted_blobn = np.expand_dims(predicted_blobn, 0)
            tensor_blobn.append(torch.from_numpy(predicted_blobn))
        return tensor_blobn
    else:
        predicted_blobn = np.expand_dims(out_3dnp, 0)
        tensor_blobn.append(torch.from_numpy(predicted_blobn))
        return tensor_blobn
    



def metrics(Metrics_tensor, tensor_label,px,multiplier):
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    surfDice_metric383 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 96*multiplier-1), include_background=False)

    row_of_metrics = []
    voxTP = 0
    voxFN = 0
    voxFP = 0

    # DICE
    dice_metric(y_pred=Metrics_tensor, y=tensor_label)  # [0][1:2, :, :, :] [0]
    dice1 = dice_metric.aggregate().item()
    print("Dice :'{:0.2f}".format(dice1))
    dice_metric.reset()

    hausdorff_metric(y_pred=Metrics_tensor, y=tensor_label)
    hausd1 = hausdorff_metric.aggregate().item()
    hausdorff_metric.reset()
    print("Hausdorff: {:0.2f}".format(hausd1))

    # print("Shape:",tensor_blobn.shape,tensor_blobn[0].shape[1])
    surfDice_metric383(y_pred=Metrics_tensor, y=tensor_label)
    sdice1 = surfDice_metric383.aggregate().item()
    surfDice_metric383.reset()
    print("Surface dice: {:0.2f}".format(sdice1))

    out_3dnp = Metrics_tensor.detach().cpu().numpy()
    out_3dnp = out_3dnp.squeeze()
    label_out_3dnp = label(out_3dnp)
    props = regionprops(label_out_3dnp)

    lbl_3dnp = tensor_label.detach().cpu().numpy()
    lbl_3dnp = lbl_3dnp.squeeze()
    label_lbl = label(lbl_3dnp)
    regions = regionprops(label_lbl)
    #print("num de blobs in label: ", len(regions))
    for i in range(len(regions)):
        r = regions[i]
        lbl_bbox = r.bbox

    print("num de blobs predicted: ", len(props))
    if len(props)<1:
        row_of_metrics.append([px, dice1, sdice1, hausd1, 0, 0, False, 0, 0])
    else:
        for n in range(len(props)):
            r = props[n]
            #print("Blob Area:", r.area)
            TP = False
            for j in range(len(r.coords)):
                if (r.coords[j][0] > lbl_bbox[0] and r.coords[j][0] < lbl_bbox[3]):
                    if (r.coords[j][1] > lbl_bbox[1] and r.coords[j][1] < lbl_bbox[4]):
                        if (r.coords[j][2] > lbl_bbox[2] and r.coords[j][2] < lbl_bbox[5]):
                            TP = True
                            voxTP += 1
            if TP:
                sumPredicted = np.sum(out_3dnp)
                sumGroundT = np.sum(lbl_3dnp)
                voxFP = abs(sumPredicted - voxTP)
                voxFN = abs(sumGroundT - voxTP)

                SegSens = voxTP / (voxTP + voxFP)
                SegPrec = voxTP / (voxTP + voxFN)
                print("Sensitivity: {:0.2f}".format(voxTP / (voxTP + voxFP)))
                print("Precision: {:0.2f}".format(voxTP / (voxTP + voxFN)))

                row_of_metrics.append([px,dice1,sdice1,hausd1,SegSens,SegPrec,TP,sumPredicted,sumGroundT])
            else:
                row_of_metrics.append([px, dice1, sdice1, hausd1, 0, 0, False, 0, 0])

    return row_of_metrics


def main(root_path, figures_folder_i, name_run):
    all_patientdir = []
    all_patientdir = os.listdir(root_path)
    all_patientdir.sort()
    print(len(all_patientdir), 'in', name_run, all_patientdir)
    metrics_yes = []
    metrics_no = []
    all_patientdir = all_patientdir[:]
    for px_dir in all_patientdir:
        templist = []
        templist.append(px_dir)
        print("Px: ", px_dir)
        temp_metrics_YesPost, temp_metrics_NoPost = evaluationPerPx(templist, root_path, figures_folder_i, name_run)
        metrics_yes.append(temp_metrics_YesPost)
        metrics_no.append(temp_metrics_NoPost)
    _ = saveMetrics(figures_folder_i, 'YesPost_metrics.csv', metrics_yes)
    _ = saveMetrics(figures_folder_i, 'NoPost_metrics.csv', metrics_no)
    return 0


def evaluationPerPx(all_patientdir, root_path, figures_folder_i,name_run):
    all_allBPmetric_rows = []
    all_twoBPmetric_rows = []

    CTALL_fpaths, itv_fpaths, gtv_fpaths, lungALL_fpaths, predicted_fpaths_yes, predicted_fpaths_no = LookSortFiles(root_path, all_patientdir)

    # Create data dictionat
    data_dicts = [
        {"image": image_name, "lung": lung_name, "ITV": itv_name, "pITV_yes": predicted_name_yes,
         "pITV_no": predicted_name_no}
        for image_name, lung_name, itv_name, predicted_name_yes, predicted_name_no in
        zip(CTALL_fpaths, lungALL_fpaths, itv_fpaths, predicted_fpaths_yes, predicted_fpaths_no)
    ]
    val_files = data_dicts[:]
    print('CT val len:', len(val_files))

    # HU are -1000 air , 0 water , usually normal tissue are around 0, top values should be around 100, bones are around 1000
    minmin_CT = -1024
    maxmax_CT = 200
    # Create Compose functions for preprocessing of train and validation
    image_keys = ["image", "lung", "ITV", "pITV_yes", "pITV_no"]
    size = 96
    image_size = (size, size, size)
    multiplier = 2
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "lung", "ITV", "pITV_yes", "pITV_no"]),
            EnsureChannelFirstd(keys=["image", "lung", "ITV"]),
            Orientationd(keys=["image", "ITV", "pITV_yes", "pITV_no"], axcodes="RAS"),
            # Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT, b_min=0.0, b_max=1.0, clip=True, ),
            Create_sequences(keys=["image", "ITV", "lung"]),
            CropForegroundd(keys=["image", "ITV", "lung"], source_key="lung", k_divisible=size),
            ResizeWithPadOrCropd(keys=["image", "ITV", "lung"], spatial_size=(384, 384, 192), method="symmetric"),
            # MaskIntensityd(keys=["image"], mask_key="lung"),
            CropForegroundd(keys=image_keys, source_key="ITV", k_divisible=size * multiplier),
            AsDiscreted(keys=["pITV_yes", "pITV_no"], to_onehot=None, threshold=.01),
            ToTensord(keys=["image", "ITV", "lung", "pITV_yes", "pITV_no"]),
        ]
    )

    # Check the images after the preprocessing

    figsize = (18, 9)
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)
    px = batch_data["image"].to('cpu').meta["filename_or_obj"][0].split('/')[-2]
    image, lung, label_ITV, pITV_yes, pITV_no = (batch_data["image"][0][0], batch_data["lung"][0][0], batch_data["ITV"][0][0], batch_data["pITV_yes"][0][0],batch_data["pITV_no"][0][0])


    tensor_pITV_yes = postITV(pITV_yes, True,True)
    tensor_pITV_no = postITV(pITV_no, True,True)
    tensor_label = postITV(label_ITV, True,False)

    row_of_metrics_yes = []
    row_of_metrics_no = []

    print("With Post:")
    for m in range(len(tensor_pITV_yes)):
        row_of_metrics_yes.append(metrics(tensor_pITV_yes[m], tensor_label[0], px, multiplier))

    print("Without Post")
    for m in range(len(tensor_pITV_no)):
        row_of_metrics_no.append(metrics(tensor_pITV_no[m], tensor_label[0], px, multiplier))

    return row_of_metrics_yes,row_of_metrics_no


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    root_path = '/home/umcg/Desktop/AutomaticITV_code/SABR1322_Nifti_AllBP_V2/'
    figures_path = '/home/umcg/Desktop/AutomaticITV_code/figures_folder_i/'
    figures_folder_i = figures_path + 'figures_SwinDyn_V14_ITV/'

    name_run = "v14_OnlyMetrics_V12"
    main(root_path, figures_folder_i, name_run)




