import os
import torch
import glob
import numpy as np
from monai.utils import first, set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    MaskIntensityd,
    EnsureType,
    Invertd,
    DivisiblePadd,
    MapTransform,
    ToTensord,
    Transpose,
    ScaleIntensity,
)
from monai.networks.nets import SwinUNETR,DynUNet
from scipy.ndimage import rotate
from monai.data import CacheDataset, DataLoader, Dataset


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


class Create_sequences(MapTransform):
    # class to transpose lung mask
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


def CreateModel_fun(size,device):
    # Create the model
    spatial_dims = 3
    max_epochs = 250
    in_channels = 1
    out_channels = 2  # including background
    lr = 1e-3  # 1e-4
    weight_decay = 1e-5
    image_size = (size,size,size)
    T_0 = 40  # Cosine scheduler

    task_id = "06"
    deep_supr_num = 1  # when is 3 shape of outputs/labels dont match
    patch_size = image_size
    spacing = [1, 1, 1]
    kernels, strides = get_kernels_strides(patch_size, spacing)

    print("MODEL SwinDyn")
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
    return modelSwin,modelDyn


def createVal_Loader_fun(cache,val_files,size):
    num_workers=0
    # HU are -1000 air , 0 water , usually normal tissue are around 0, top values should be around 100, bones are around 1000
    minmin_CT = -1024
    maxmax_CT = 200
    set_determinism(seed=0)
    image_keys = ["image","lung"]
    p = .5 #Data aug transform probability
    pin_memory = True if num_workers > 0 else False  # Do not change

    val_transforms = Compose(
        [
            LoadImaged(keys=image_keys),
            EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=["image"], axcodes="RAS"),
            #Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT,b_min=0.0, b_max=1.0, clip=True,),
            Create_sequences(keys=image_keys),
            CropForegroundd(keys=image_keys, source_key="lung",k_divisible=size),
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
    return val_loader
