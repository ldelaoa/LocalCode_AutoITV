import torch
from monai.transforms import (
    Compose,
    KeepLargestConnectedComponent,
    AsDiscrete,
    ResizeWithPadOrCrop,
    EnsureType,
    ScaleIntensity,
)


def rescaleITV(predicted_ITV):
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
        elif k == 5:
            ITV_tensor_2BP = torch.add(ITV_tensor_2BP, temp_ITV_tensor)
        elif k != 0 and k != 5:
            ITV_tensor_10BP = torch.add(ITV_tensor_10BP, temp_ITV_tensor)

    return ITV_tensor_10BP, ITV_tensor_2BP, predicted_ITV_rescale