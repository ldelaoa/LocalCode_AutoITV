
from skimage.filters import threshold_otsu,threshold_multiotsu
import os
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import rotate
from skimage.morphology import dilation,ball,erosion,remove_small_objects

from monai.utils import first, set_determinism
from monai.transforms import (
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    MapTransform,
    ToTensord,
)

from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,pad_list_data_collate
from metrics import metrics_fun,saveMetrics_fun


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



def LookSortFiles_ITV_Eval(root_path, all_patientdir):
    CTALL_fpaths =[]
    lungALL_fpaths =[]
    itv_fpaths = []  
    predicted_fpaths =[]
    itv_miss = 0
    ct_miss =0
    lung_miss=0
    pred_miss=0
    #Look for labels
    for patient_path in all_patientdir:
        for root, dirs, files in os.walk(root_path + patient_path, topdown=False):
            for f in files:
                if '_igtv' in f.lower() and "predicted" not in f.lower():
                    itv_miss+=1
                    itv_fpaths.append(os.path.join(root_path, patient_path, f))
                elif '_itv' in f.lower() and "predicted" not in f.lower():
                    itv_miss+=1
                    itv_fpaths.append(os.path.join(root_path, patient_path, f))
                if '_ct' in f.lower() and "predicted" not in f.lower() and "ave" not in f.lower():
                    ct_miss+=1
                    CTALL_fpaths.append(os.path.join(root_path, patient_path, f))
                if 'lung' in f.lower() and "ave" not in f.lower():
                    lung_miss += 1
                    lungALL_fpaths.append(os.path.join(root_path, patient_path, f))
                if 'predicted12' in f.lower():
                    pred_miss+=1
                    predicted_fpaths.append(os.path.join(root_path, patient_path, f))
    
    for i in range(len(predicted_fpaths)-len(itv_fpaths)):
        itv_fpaths.append(itv_fpaths[0])
    
    CTALL_fpaths = np.sort(CTALL_fpaths)
    lungALL_fpaths = np.sort(lungALL_fpaths)
    itv_fpaths = np.sort(itv_fpaths)
    predicted_fpaths = np.sort(predicted_fpaths)
    print("CT image",ct_miss,"Lung",lung_miss,"ITV",itv_miss,"Pred",pred_miss)
    return CTALL_fpaths,itv_fpaths,lungALL_fpaths,predicted_fpaths


def post(tensor):
    converted = tensor.cpu().detach().numpy()
    dilated = dilation(converted, ball(3))
    return torch.from_numpy(dilated)


def CreateITV(predicted_ITV, tresh, dilation_bool):
    # Creating ITV
    for k in range(len(predicted_ITV)):
        if k == 0:
            ITV_tensor_10BP = predicted_ITV[k]
            ITV_tensor_2BP = predicted_ITV[k]
        elif k == 5:
            ITV_tensor_2BP = torch.add(ITV_tensor_2BP, predicted_ITV[k])
            ITV_tensor_10BP = torch.add(ITV_tensor_10BP, predicted_ITV[k])
        elif k != 0 and k != 5:
            ITV_tensor_10BP = torch.add(ITV_tensor_10BP, predicted_ITV[k])

    # Creating Optimized ITV with otsu thresholding
    maxITV = ITV_tensor_10BP.max()
    otsu_val = threshold_otsu(ITV_tensor_10BP.detach().cpu().numpy())
    otsu_multival = threshold_multiotsu(ITV_tensor_10BP.detach().cpu().numpy(), classes=3)
    print(maxITV.item(), "Otsu: ", otsu_val, "Multival 1: ", otsu_multival[0], " 2: ", otsu_multival[-1])
    if False:
        to_hist = ITV_tensor_10BP.flatten().detach().cpu().numpy()
        plt.hist(to_hist, bins=256, range=(to_hist.min() + .1, to_hist.max()))
        plt.show()
    ITV_tensor_optimized = ITV_tensor_10BP.clone()
    if tresh == "Otsu":
        ITV_tensor_optimized[ITV_tensor_optimized < otsu_val] = 0
        ITV_tensor_optimized[ITV_tensor_optimized >= otsu_val] = 1
    else:
        ITV_tensor_optimized[ITV_tensor_optimized < otsu_multival[-1]] = 0
        ITV_tensor_optimized[ITV_tensor_optimized >= otsu_multival[-1]] = 1

    # Binarizing 2BP and 10BP
    ITV_tensor_10BP[ITV_tensor_10BP > 0] = 1
    ITV_tensor_2BP[ITV_tensor_2BP > 0] = 1
    if dilation_bool:
        ITV_10BP_dil = post(ITV_tensor_10BP)
        ITV_2BP_dil = post(ITV_tensor_2BP)
        ITV_Opti_dil = post(ITV_tensor_optimized)
        return ITV_tensor_10BP, ITV_tensor_2BP, ITV_tensor_optimized, ITV_10BP_dil, ITV_2BP_dil, ITV_Opti_dil
    else:
        return ITV_tensor_10BP, ITV_tensor_2BP, ITV_tensor_optimized


def main(root_path,device,figures_folder_i):
    all_patientdir = os.listdir(root_path)
    all_patientdir.sort()
    full_metrics = []

    for px_dir in all_patientdir:
        templist = []
        templist.append(px_dir)
        print("Px: ", px_dir)
        temp_metrics = evaluationPerPx(templist, root_path)
        full_metrics.append(temp_metrics)

    _ = saveMetrics_fun(figures_folder_i, 'FullMetrics_TP.csv', full_metrics)
    return 0


def evaluationPerPx(all_patientdir, root_path):
    CTALL_fpaths,itv_fpaths, lungALL_fpaths,predicted_fpaths = LookSortFiles_ITV_Eval(root_path, all_patientdir)

    #Create data dictionat
    data_dicts = [
        {"image": image_name,"ITV":itv_name,"lung":lung_name,"Pred":pred_name}
        for image_name,itv_name,lung_name,pred_name in zip(CTALL_fpaths,itv_fpaths,lungALL_fpaths,predicted_fpaths)
    ]
    val_files =data_dicts[:]
    print('CT val len:',len(val_files))

    # HU are -1000 air , 0 water , usually normal tissue are around 0, top values should be around 100, bones are around 1000
    minmin_CT = -1024
    maxmax_CT = 200
    #Create Compose functions for preprocessing of train and validation
    image_keys = ["image","lung","ITV","Pred"]
    print("image_keys",image_keys)
    size = 96
    image_size = (size,size,size)
    multiplier = 1
    val_transforms = Compose(
        [
            LoadImaged(keys=image_keys),
            EnsureChannelFirstd(keys=["image","lung","ITV","Pred"]),
            Orientationd(keys=["image","ITV","Pred"], axcodes="RAS"),
            #Spacingd(keys=["image","label","GTV","Pred"], pixdim=(1,1,1,1),mode=("bilinear","nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT,b_min=0.0, b_max=1.0, clip=True,),
            Create_sequences(keys=["image","lung","ITV"]),
            CropForegroundd(keys=["image","ITV","lung"], source_key="lung",k_divisible=size),
            ResizeWithPadOrCropd(keys=["image","lung","ITV","Pred"],spatial_size=(384, 384, 192), method="symmetric"),
            #MaskIntensityd(keys=["image"], mask_key="lung"),
            #CropForegroundd(keys=image_keys, source_key="lung",k_divisible=size*multiplier),
            #AsDiscreted(keys=["Pred",],to_onehot=None,threshold=.01),
            ToTensord(keys=["image","lung","ITV"]),
        ]
    )


    #Check the images after the preprocessing
    check_ds =Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1,num_workers=0)
    batch_data = first(check_loader)
    px = batch_data["image"].to('cpu').meta["filename_or_obj"][0].split('/')[-2]
    image,lung,itv_label = (batch_data["image"][0][0],batch_data["lung"][0][0],batch_data["ITV"][0][0])

    Pred_list = []
    for batch_data in check_loader:
        Pred_list.append(batch_data["Pred"][0][0])


    ITV_10BP,ITV_2BP,ITV_Opti,ITV_10BP_dil,ITV_2BP_dil,ITV_Opti_dil = CreateITV(Pred_list,"Otsu",True) #"Otsu" or "OtsuMulti"

    metrics_10BP = metrics_fun(ITV_10BP,itv_label,px,1)
    metrics_2BP = metrics_fun(ITV_2BP,itv_label,px,1)
    metrics_Opti = metrics_fun(ITV_Opti,itv_label,px,1)
    metrics_10BP_dil = metrics_fun(ITV_10BP_dil,itv_label,px,1)
    metrics_2BP_dil = metrics_fun(ITV_2BP_dil,itv_label,px,1)
    metrics_Opti_dil = metrics_fun(ITV_Opti_dil,itv_label,px,1)

    rowOfMetrics = [metrics_10BP,metrics_2BP,metrics_Opti,metrics_10BP_dil,metrics_2BP_dil,metrics_Opti_dil]

    return rowOfMetrics


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    root_path = '/home/umcg/Desktop/AutomaticITV_code/SABR1322_Nifti_AllBP_V2/'
    figures_folder_i = '/home/umcg/Desktop/AutomaticITV_code/figures_folder_i/figures_SwinDyn_V12_ITV/'
    main(root_path, device,figures_folder_i)

