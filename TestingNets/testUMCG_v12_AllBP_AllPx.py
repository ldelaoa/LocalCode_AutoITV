
import os
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
from scipy.ndimage import rotate
import nibabel as nib
import SimpleITK as sitk
#from lungtumormask import mask as tumormask
from lungmask import mask as lungmask_fun
from skimage.measure import label, regionprops,shannon_entropy
from skimage.morphology import dilation,ball,erosion,remove_small_objects
import csv
from monai.utils import first, set_determinism
from monai.transforms import (
    ResizeWithPadOrCrop,
    MaskIntensityd,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    FillHoles,
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
from monai.networks.nets import UNet,VNet,SwinUNETR,UNETR,DynUNet
from monai.metrics import DiceMetric,SurfaceDiceMetric,HausdorffDistanceMetric,compute_surface_dice
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,pad_list_data_collate


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


def saveMetrics(path_to_save,file_to_save,all_metric_rows):
    f = open(path_to_save+file_to_save, 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(['Patient', 'Dice', 'SDice', 'HausD', 'Seg Sens','Seg PRec','TP', 'Area Predic', 'Area Lab'])
    writer.writerows(all_metric_rows)
    f.close()
    return 0


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
    CTALL_fpaths =[]
    lungALL_fpaths =[]
    itv_fpaths = []
    gtv_fpaths = []    
    for patient_path in all_patientdir:
        ct_miss = 0
        gtv_miss = 0
        itv_miss = 0
        lung_miss = 0
        print(patient_path)
        for root, dirs, files in os.walk(root_path + patient_path, topdown=False):
            for f in files:
                if "_gtv" in f.lower():
                    gtv_fpaths.append(os.path.join(root_path, patient_path, f))
                    gtv_miss +=1
                if "_igtv" in f.lower() or "itv" in f.lower():
                    itv_fpaths.append(os.path.join(root_path, patient_path, f))
                    itv_miss +=1
                if "%" in f.lower() and not("ave" in f.lower()):
                    if "ct" in f.lower():                            
                        CTALL_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss +=1
                    if "lung" in f.lower():
                        lungALL_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss +=1
            for i in range(len(CTALL_fpaths)-1):
                gtv_fpaths.append(itv_fpaths[-1])
                itv_fpaths.append(itv_fpaths[-1])

    print('ct: ',ct_miss,"Lungs: ",lung_miss,"GTV Miss: ", gtv_miss, "ITV Miss: ",itv_miss)
    CTALL_fpaths = np.sort(CTALL_fpaths)
    lungALL_fpaths = np.sort(lungALL_fpaths)
    itv_fpaths = np.sort(itv_fpaths)
    gtv_fpaths = np.sort(gtv_fpaths)
    return CTALL_fpaths, itv_fpaths,gtv_fpaths, lungALL_fpaths


def evaluationPerPx(all_patientdir, root_path, pretrained_path_Swin, pretrained_path_Dyn, figures_folder_i,name_run):
    all_allBPmetric_rows = []
    all_twoBPmetric_rows = []
    CTALL_fpaths, itv_fpaths, gtv_fpaths, lungALL_fpaths = LookSortFiles(root_path, all_patientdir)

    if len(lungALL_fpaths)==0:
        print("Patient withou Lung Delineations, Will perform delineations")
        CreateLungMasks(root_path,CTALL_fpaths)
        CTALL_fpaths, itv_fpaths, gtv_fpaths, lungALL_fpaths = LookSortFiles(root_path, all_patientdir)

    data_dicts = [
        {"image": image_name, "lung": lung_name, "GTV": gtv_name, "ITV": itv_name}
        for image_name, lung_name, gtv_name, itv_name in zip(CTALL_fpaths, lungALL_fpaths, gtv_fpaths, itv_fpaths)
    ]
    val_files = data_dicts[:]
    print('CT val len:', len(val_files))

    size = 96
    image_size = (size, size, size)

    val_loader = createVal_Loader_fun(cache, val_files, size)
    modelSwin, modelDyn = CreateModel_fun(size)

    # Load pretrained model
    if pretrained_path_Swin is not None:
        modelSwin.load_state_dict(torch.load(pretrained_path_Swin, map_location=torch.device(device)))
        print('Using Swin pretrained weights!')
    if pretrained_path_Dyn is not None:
        modelDyn.load_state_dict(torch.load(pretrained_path_Dyn, map_location=torch.device(device)))
        print('Using Dyn pretrained weights!')

    predicted_ITV, val_data, px, val_inputs = EvalNet_fun(val_loader, modelSwin, modelDyn, image_size)

    # interpolate ITV Tensor
    ITV_tensor, ITV_tensor_2BP, predicted_ITV_interp = interpolateTensor(predicted_ITV)
    # Display tumor trajectory
    _ = TumorTrajectory_fun(predicted_ITV_interp)

    # Create Label Tensor
    lbl_3dnp, tensor_label = CreateLabelTensor_fun(val_data)

    # Evaluate Tensors
    print("All breathing phases:")
    ITV_tensor_post = postITV(ITV_tensor, binarize=True)
    for p in range(len(ITV_tensor_post)):
        print("Stats for blob #", p + 1)
        out_3dnp,metric_rows = metrics(ITV_tensor_post[p], tensor_label,px)
        all_allBPmetric_rows.append(metric_rows)
        path_to_save = os.path.join(figures_folder_i, px)
        saveNifti(np_image=out_3dnp, path_to_save=path_to_save + '/predictedITV_AllBP_' + str(p) + '.Nii')

    print("TWO breathing phases:")
    BP2_ITV_tensor_post = postITV(ITV_tensor_2BP, binarize=True)
    for p in range(len(BP2_ITV_tensor_post)):
        out_3dnp,metric_rows = metrics(BP2_ITV_tensor_post[p], tensor_label,px)
        all_twoBPmetric_rows.append(metric_rows)
        path_to_save = os.path.join(figures_folder_i, px)
        saveNifti(np_image=out_3dnp, path_to_save=path_to_save + '/predictedITV_TwoBP_' + str(p) + '.Nii')

    return all_allBPmetric_rows,all_twoBPmetric_rows


def createVal_Loader_fun(cache,val_files,size):
    num_workers=0
    # HU are -1000 air , 0 water , usually normal tissue are around 0, top values should be around 100, bones are around 1000
    minmin_CT = -1024
    maxmax_CT = 200
    set_determinism(seed=0)
    image_keys = ["image","lung","GTV","ITV"]
    p = .5 #Data aug transform probability
    pin_memory = True if num_workers > 0 else False  # Do not change

    val_transforms = Compose(
        [
            LoadImaged(keys=image_keys),
            EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=["image","GTV","ITV"], axcodes="RAS"),
            #Spacingd(keys=["image","label"], pixdim=(1,1,1),mode=("bilinear","nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=minmin_CT, a_max=maxmax_CT,b_min=0.0, b_max=1.0, clip=True,),
            Create_sequences(keys=image_keys),
            CropForegroundd(keys=image_keys, source_key="lung",k_divisible = size),
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


def CreateModel_fun(size):
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


def EvalNet_fun(val_loader, modelSwin, modelDyn,image_size):
    out_channels = 2  # including background
    post_transforms = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True, threshold=0.9),
            # FillHoles(applied_labels=1, connectivity=0),
            # RemoveSmallObjects(min_size=64, connectivity=3, independent_channels=True),
            ScaleIntensity(minv=0.0, maxv=1.0),
            KeepLargestConnectedComponent(applied_labels=None, is_onehot=False, connectivity=2, num_components=1),
        ]
    )
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, threshold=0.9),
                         ScaleIntensity(minv=0.0, maxv=1.0)])

    # Testing the model
    print("Length val_loader: ", len(val_loader))

    predicted_ITV = []
    best_blob = True
    count = 0
    modelSwin.eval()
    modelDyn.eval()
    with torch.no_grad():
        for val_data in val_loader:

            # val_inputs, val_labels = (
            #    val_data["image"].to(device),
            #    val_data["label"].to(device),)
            val_inputs = val_data["image"].to(device)
            roi_size = image_size
            sw_batch_size = 1
            px = val_data["image"].to('cpu').meta["filename_or_obj"][0].split('/')[-2]
            bp = val_data["image"].to('cpu').meta["filename_or_obj"][0].split('/')[-1].split('=')[-1]
            print('Px: ',px, ' BP: ', bp, " Count: ", count + 1)
            count += 1

            val_outputs_Swin = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelSwin)
            val_outPost_Swin = [post_transforms(i) for i in decollate_batch(val_outputs_Swin)]

            val_outputs_Dyn = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelDyn)
            val_outPost_Dyn = [post_transforms(i) for i in decollate_batch(val_outputs_Dyn)]

            val_outPost = torch.logical_or(val_outPost_Swin[0], val_outPost_Dyn[0])

            if best_blob:
                out_3dnp = val_outPost[0].detach().cpu().numpy()
                out_3dnp = out_3dnp.squeeze()
                out_3dnp = dilation(out_3dnp, ball(2))
                label_out_3dnp = label(out_3dnp)
                props = regionprops(label_out_3dnp, val_inputs[0].detach().cpu().numpy().squeeze())

                print("num de blobs predicted: ", len(props))
                for n in range(len(props)):
                    r = props[n]
                    # print('prediction bbox',r.bbox,"size",len(r.coords))
                    patch = np.zeros(out_3dnp.shape)
                    for j in range(len(r.coords)):
                        patch[r.coords[j][0], r.coords[j][1], r.coords[j][2]] = 1
                    # Create different matrixes, one for each blob to send to metrics
                    predicted_blobn = np.zeros(out_3dnp.shape)
                    predicted_blobn[label_out_3dnp == n + 1] = 1
                    #print("Bounding Box: ", r.bbox)
                    #print("Values: ", r.axis_major_length / r.axis_minor_length, "Feret: ", r.feret_diameter_max)
                    #print("Intensity:", r.intensity_max, r.intensity_min, r.intensity_mean)
                    #print("Entropy:", shannon_entropy(predicted_blobn))

                    if len(props) == 1:
                        bestBlob = np.expand_dims(predicted_blobn, 0)
                        tensor_blobn = torch.from_numpy(bestBlob)
                    elif n == 0:
                        ratio_zero = r.axis_major_length / r.axis_minor_length
                        minint_zero = r.intensity_min
                        entr_zero = shannon_entropy(predicted_blobn)
                        blob_zero = predicted_blobn
                    elif n > 0:
                        ratio_curr = r.axis_major_length / r.axis_minor_length
                        minint_curr = r.intensity_min
                        entr_curr = shannon_entropy(predicted_blobn)
                        if (ratio_curr > 3) or (ratio_zero > 3):
                            print("Selected by ratio")
                            if ratio_curr < ratio_zero:
                                bestBlob = np.expand_dims(predicted_blobn, 0)
                                tensor_blobn = torch.from_numpy(bestBlob)
                            else:
                                bestBlob = np.expand_dims(blob_zero, 0)
                                tensor_blobn = torch.from_numpy(bestBlob)
                        else:
                            if minint_zero < 0.0001:
                                bestBlob = np.expand_dims(predicted_blobn, 0)
                                tensor_blobn = torch.from_numpy(bestBlob)
                            elif minint_curr < 0.0001:
                                bestBlob = np.expand_dims(blob_zero, 0)
                                tensor_blobn = torch.from_numpy(bestBlob)
                            elif entr_curr < entr_zero:
                                bestBlob = np.expand_dims(predicted_blobn, 0)
                                tensor_blobn = torch.from_numpy(bestBlob)
                            else:
                                bestBlob = np.expand_dims(blob_zero, 0)
                                tensor_blobn = torch.from_numpy(bestBlob)

            if best_blob:
                predicted_ITV.append(tensor_blobn)
            else:
                predicted_ITV.append(val_outPost)

    return predicted_ITV,val_data,px,val_inputs


def interpolateTensor(predicted_ITV):
    predicted_ITV_rescale = []

    post_rescale = Compose(
        [ResizeWithPadOrCrop(spatial_size=(384, 384, 192), method="symmetric"), AsDiscrete(threshold=0.1),
         ScaleIntensity(minv=0.0, maxv=1.0)])
    for k in range(len(predicted_ITV)):
        temp_ITV_tensor = post_rescale(predicted_ITV[k])

        predicted_ITV_rescale.append(temp_ITV_tensor)

        if k == 0:
            ITV_tensor_10BP = temp_ITV_tensor
            ITV_tensor_2BP = temp_ITV_tensor
        elif k == 4:
            ITV_tensor_2BP = torch.add(ITV_tensor_2BP, temp_ITV_tensor)
        elif k != 0 and k != 4:
            ITV_tensor_10BP = torch.add(ITV_tensor_10BP, temp_ITV_tensor)

    return ITV_tensor_10BP, ITV_tensor_2BP, predicted_ITV_rescale


def TumorTrajectory_fun(predicted_ITV_interp):
    xx = np.zeros(10,int)
    yy = np.zeros(10,int)
    zz = np.zeros(10,int)
    for l in range(len(predicted_ITV_interp)):
        labeledGTV = label(predicted_ITV_interp[l].detach().cpu().numpy().squeeze())
        propsGTV = regionprops(labeledGTV)
        for m in range(len(propsGTV)):
            #print("GTV #",l," -Centroid: ",propsGTV[m].centroid)
            xx[l],yy[l],zz[l] = propsGTV[m].centroid_local
    ax = plt.figure().add_subplot(projection='3d')
    plt.subplot(1,1,1),plt.plot(xx,yy,zz,label='x,y,z')
    ax.legend()
    plt.show()
    plt.subplot(1,3,1),plt.plot(xx)
    plt.subplot(1,3,2),plt.plot(yy)
    plt.subplot(1,3,3),plt.plot(zz)
    plt.show()
    print('Limits: ',xx.max()-xx.min(),yy.max()-yy.min(),zz.max()-zz.min())
    return 0


def CreateLabelTensor_fun(val_data):
    post_label = Compose([EnsureType(),ResizeWithPadOrCrop(spatial_size=(384,384,192),method="symmetric"),AsDiscrete(threshold=0.5),ScaleIntensity(minv=0.0, maxv=1.0)])
    # Create Label tensors
    # val_GTV,val_ITV = val_data["GTV"].to(device),val_data["ITV"].to(device)
    # val_GTV = [post_label(i) for i in decollate_batch(val_GTV)]
    val_ITV = val_data["ITV"].to(device)
    val_ITV = [post_label(i) for i in decollate_batch(val_ITV)]
    lbl_3dnp = val_ITV[0].detach().cpu().numpy()
    lbl_3dnp = lbl_3dnp.squeeze()
    lbl_4d = np.expand_dims(lbl_3dnp, 0)
    tensor_label = torch.from_numpy(lbl_4d)
    label_img = label(lbl_3dnp)
    regions = regionprops(label_img)
    print("num de blobs in label: ", len(regions))
    for i in range(len(regions)):
        r = regions[i]
        print('label bbox ', r.bbox, "size", len(r.coords))
        lbl_bbox = r.bbox

    print(tensor_label.shape)
    return lbl_3dnp, tensor_label


def metrics(Metrics_tensor, tensor_label,px):
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    surfDice_metric191 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 191), include_background=False)
    surfDice_metric287 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 287), include_background=False)
    surfDice_metric383 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 383), include_background=False)

    row_of_metrics = []
    voxTP = 0
    voxFN = 0
    voxFP = 0

    # DICE
    dice_metric(y_pred=Metrics_tensor, y=tensor_label)  # [0][1:2, :, :, :] [0]
    dice1 = dice_metric.aggregate().item()
    print('Dice :', dice1)
    dice_metric.reset()

    hausdorff_metric(y_pred=Metrics_tensor, y=tensor_label)
    hausd1 = hausdorff_metric.aggregate().item()
    hausdorff_metric.reset()
    print('Hausdorff:', hausd1)

    # print("Shape:",tensor_blobn.shape,tensor_blobn[0].shape[1])
    if Metrics_tensor.shape[1] == 288:
        surfDice_metric287(y_pred=Metrics_tensor, y=tensor_label)
        sdice1 = surfDice_metric287.aggregate().item()
        surfDice_metric287.reset()
    elif Metrics_tensor.shape[1] == 192:
        surfDice_metric191(y_pred=Metrics_tensor, y=tensor_label)
        sdice1 = surfDice_metric191.aggregate().item()
        surfDice_metric191.reset()
    else:
        surfDice_metric383(y_pred=Metrics_tensor, y=tensor_label)
        sdice1 = surfDice_metric383.aggregate().item()
        surfDice_metric383.reset()
    print('Surface dice:', sdice1)

    out_3dnp = Metrics_tensor.detach().cpu().numpy()
    out_3dnp = out_3dnp.squeeze()
    label_out_3dnp = label(out_3dnp)
    props = regionprops(label_out_3dnp)

    lbl_3dnp = tensor_label.detach().cpu().numpy()
    lbl_3dnp = lbl_3dnp.squeeze()
    label_lbl = label(lbl_3dnp)
    regions = regionprops(label_lbl)
    print("num de blobs in label: ", len(regions))
    for i in range(len(regions)):
        r = regions[i]
        lbl_bbox = r.bbox

    print("num de blobs predicted: ", len(props))
    for n in range(len(props)):
        r = props[n]
        print("Blob Area:", r.area)
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
            print("Sensitivity: ", voxTP / (voxTP + voxFP))
            print("Precision: ", voxTP / (voxTP + voxFN))

            row_of_metrics.append([px,dice1,sdice1,hausd1,SegSens,SegPrec,TP,sumPredicted,sumGroundT])

        return out_3dnp,row_of_metrics


def saveNifti(np_image,path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    converted_array = np.array(np_image, dtype=np.float32)
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(converted_array, affine)
    nib.save(nifti_file, path_to_save) # Here you put the path + the extionsion 'nii' or 'nii.gz'
    return 0


def postITV(ITV_tensor,binarize):
    tensor_blobn=[]
    #Delete outliers of slices
    out_3dnp = ITV_tensor.detach().cpu().numpy()
    out_3dnp = out_3dnp.squeeze()
    out_3dnp[:,:,:20] = 0
    out_3dnp[:,:,170:] = 0
    out_3dnp[out_3dnp>=1] = 1
    out_3dnp[out_3dnp<1] = 0
    label_out_3dnp = label(out_3dnp)
    props = regionprops(label_out_3dnp)
    for n in range(len(props)):
        r = props[n]
        predicted_blobn = np.zeros(out_3dnp.shape)
        predicted_blobn[label_out_3dnp==n+1]=1
        #predicted_blobn = dilation(predicted_blobn, ball(3))
        predicted_blobn = np.expand_dims(predicted_blobn, 0)
        tensor_blobn.append(torch.from_numpy(predicted_blobn))
    return tensor_blobn


def main(root_path,pretrained_path_Swin,pretrained_path_Dyn,figures_folder_i,name_run):

    all_patientdir = []
    all_patientdir = os.listdir(root_path)
    all_patientdir.sort()
    print(len(all_patientdir),'in',name_run,all_patientdir)
    metrics_allBP = []
    metrics_twoBP = []
    all_patientdir = all_patientdir[:]
    for px_dir in all_patientdir:
        templist = []
        templist.append(px_dir)
        print("Px: ",px_dir)
        temp_metrics_allBP,temp_metrics_twoBP = evaluationPerPx(templist, root_path, pretrained_path_Swin, pretrained_path_Dyn, figures_folder_i, name_run)
        metrics_allBP.append(temp_metrics_allBP)
        metrics_twoBP.append(temp_metrics_twoBP)
    _ = saveMetrics(figures_folder_i, 'AllBP_metrics.csv', metrics_allBP)
    _ = saveMetrics(figures_folder_i, 'TwoBP_metrics.csv', metrics_twoBP)

    return 0



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = False
    print('device:', device)

    root_path = '/home/umcg/Desktop/AutomaticITV_code/SABR1322_Nifti_AllBP_V2/'

    preweight_path = '/home/umcg/Desktop/AutomaticITV_code/weights/v11/'
    pretrained_path_Swin = preweight_path + 'best_SwinUnet_V11_UMCG_TestSet3.pth'
    pretrained_path_Dyn = preweight_path + 'best_DynUnet_V11_UMCG_TestSet3.pth'

    figures_path = '/home/umcg/Desktop/AutomaticITV_code/figures_folder_i/'
    figures_folder_i = figures_path + 'figures_SwinDyn_V12_ITV/'

    name_run = "TestRun" + "SwynDyn" + "ITV_FullContours" + "_V12"
    print(name_run)

    main(root_path, pretrained_path_Swin, pretrained_path_Dyn, figures_folder_i, name_run)

    print("The end")
