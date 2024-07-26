import os
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from torchvision import transforms
import torch
import albumentations as A

current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录

class SpectrumNormalize(ImageOnlyTransform):
    """Spectrum Normalization
    """
    def __init__(self, always_apply=False, p=1.0):
        super(SpectrumNormalize, self).__init__(always_apply, p)

    def apply(self, image, **params):
        normalized_spectrum = self.extract_spectrum(image)

        return normalized_spectrum

    @staticmethod
    def extract_spectrum(image):
        image_float32 = np.float32(image)
        x = transforms.ToTensor()(image_float32)
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        out = np.transpose(x_freq.abs().numpy(), (1, 2, 0))  # 幅度谱
        out = cv2.normalize(out, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return out


class DoNothing(ImageOnlyTransform):
    """Do nothing"""
    def __init__(self, always_apply=False, p=1.0):
        super(DoNothing, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return image



def create_train_transforms(size=300, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                            is_crop=False,):
    resize_fuc = A.RandomCrop(height=size, width=size) if is_crop else A.LongestMaxSize(max_size=size)
    aug_hard = [
        A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
        A.RandomScale(scale_limit=(-0.5, 0.5), p=0.2),  # 23/11/04 add
        A.HorizontalFlip(),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(p=0.1),
        A.RandomRotate90(),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0) if is_crop else DoNothing(),
        resize_fuc,
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.5),
        A.OneOf([A.CoarseDropout(), A.GridDropout()], p=0.5),
        A.ToGray(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    return A.Compose(aug_hard, additional_targets={'rec_image': 'image'})


def create_val_transforms(size=300, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_crop=False):
    # resize_fuc = A.CenterCrop(height=size, width=size) if is_crop else A.Resize(height=size, width=size)
    resize_fuc = A.CenterCrop(height=size, width=size) if is_crop else A.LongestMaxSize(max_size=size)
    return A.Compose([
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0) if is_crop else DoNothing(),
        resize_fuc,
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'rec_image': 'image'})


def create_sdie_transforms(size=224, phase='train'):
    if phase == 'train':
        aug_list = [
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomCrop(height=size, width=size),
            # A.HorizontalFlip(p=0.2),
            # A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.2),
        ]
    else:
        aug_list = [
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.CenterCrop(height=size, width=size)
        ]
    return A.Compose(aug_list, additional_targets={'rec_image': 'image'})


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = cv2.imread('samples/01.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (128, 128))
    print(image.shape)
    # transform = create_sdie_transforms(size=224, phase='train')
    transform = create_train_transforms(size=512, is_crop=False)
    # transform = create_val_transforms(size=300, is_crop=True)

    data = transform(image=image)
    out = data["image"]






