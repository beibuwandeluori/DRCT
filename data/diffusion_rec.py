from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import torch
from PIL import Image
import numpy as np
import cv2
import random
import os
import glob
from tqdm import tqdm
import albumentations as A
import gc

GenImage_LIST = [
    'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
    'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
    'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan'
]
DRCT_2M_LIST = [
    'ldm-text2im-large-256', 'stable-diffusion-v1-4', 'stable-diffusion-v1-5', 'stable-diffusion-2-1',
    'stable-diffusion-xl-base-1.0', 'stable-diffusion-xl-refiner-1.0', 'sd-turbo', 'sdxl-turbo',
    'lcm-lora-sdv1-5', 'lcm-lora-sdxl',  'sd-controlnet-canny',
    'sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0', 'stable-diffusion-inpainting',
    'stable-diffusion-2-inpainting', 'stable-diffusion-xl-1.0-inpainting-0.1',
]

def create_crop_transforms(height=224, width=224):
    aug_list = [
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=height, width=width)
    ]
    return A.Compose(aug_list)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def find_nearest_multiple(a, multiple=8):
    """
    找到最接近a的multiple倍数，且该倍数大于a
    """
    n = a // multiple
    remainder = a % multiple
    if remainder == 0:
        # 如果a整除multiple，，那么n就是multiple倍数
        return a
    else:
        # 否则，我们需要找到更大的一个multiple的倍数
        return (n + 1) * multiple


def pad_image_to_size(image, target_width=224, target_height=224, fill_value=255):
    """将图像填充为目标宽度和高度，使用指定的填充值（默认为255）"""

    height, width = image.shape[:2]

    if height < target_height:
        pad_height = target_height - height
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
    else:
        pad_top = pad_bottom = 0

    if width < target_width:
        pad_width = target_width - width
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
    else:
        pad_left = pad_right = 0

    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=fill_value
    )

    return padded_image


def center_crop(image, crop_width, crop_height):
    height, width = image.shape[:2]

    # 计算裁剪区域的起始点和终点
    if width > crop_width:
        start_x = (width - crop_width) // 2
        end_x = start_x + crop_width
    else:
        start_x, end_x = 0, width
    if height > crop_height:
        start_y = (height - crop_height) // 2
        end_y = start_y + crop_height
    else:
        start_y, end_y = 0, height

    # 使用数组切片实现 center crop
    cropped_image = image[start_y:end_y, start_x:end_x]
    if cropped_image.shape[0] < crop_height or cropped_image.shape[1] < crop_width:
        cropped_image = pad_image_to_size(cropped_image, target_width=crop_width, target_height=crop_width,
                                          fill_value=255)

    return cropped_image


def stable_diffusion_inpainting(pipe, image, mask_image, prompt, steps=50, height=512, width=512,
                                seed=2023, guidance_scale=7.5):
    set_seed(int(seed))
    image_pil = Image.fromarray(image)
    mask_image_pil = Image.fromarray(mask_image).convert("L")
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    new_image = pipe(prompt=prompt, image=image_pil, mask_image=mask_image_pil,
                     height=height, width=width, num_inference_steps=steps,
                     guidance_scale=guidance_scale).images[0]

    return new_image


def read_image(image_path, max_size=512):
    create_crop_transforms(height=224, width=224)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # crop image
    height, width = image.shape[:2]
    height = height if height < max_size else max_size
    width = width if width < max_size else max_size
    transform = create_crop_transforms(height=height, width=width)
    image = transform(image=image)["image"]
    # 处理8的倍数
    original_shape = image.shape
    new_height = find_nearest_multiple(original_shape[0], multiple=8)
    new_width = find_nearest_multiple(original_shape[1], multiple=8)
    new_image = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    new_image[:original_shape[0], :original_shape[1]] = image

    mask_image = np.zeros_like(image)

    del transform
    del image
    gc.collect()

    return new_image, mask_image, original_shape


def func(image_path, save_path, crop_save_path, step=50, max_size=1024):
    image, mask_image, original_shape = read_image(image_path, max_size)
    # print(image.shape, mask_image.shape, np.unique(mask_image))
    new_image = stable_diffusion_inpainting(pipe, image, mask_image, prompt='', steps=step,
                                            height=image.shape[0],
                                            width=image.shape[1],
                                            seed=2023, guidance_scale=7.5)
    # 恢复原来的尺寸
    new_image = new_image.crop(box=(0, 0, original_shape[1], original_shape[0]))
    # 保存结果
    new_image.save(save_path)
    if not os.path.exists(crop_save_path):
        image = Image.fromarray(image).crop(box=(0, 0, original_shape[1], original_shape[0]))
        image.save(crop_save_path)


if __name__ == '__main__':
    # load stable diffusion models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = '/disk2/chenby/nas_dsw/dataset/AIGC_data'

    sd_model_names = ["runwayml/stable-diffusion-inpainting",
                      "stabilityai/stable-diffusion-2-inpainting",
                      "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                      ]
    index = 0
    sd_model_name = sd_model_names[index]
    if 'xl' in sd_model_name:
        pipe = AutoPipelineForInpainting.from_pretrained(
            sd_model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,  # 关闭安全审查机制
        )
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(device)
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_name,
            # revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,  # 关闭安全审查机制
        )
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(device)
    print(f"Load model successful:{sd_model_name}")


    # Create "SDv1-DR", "SDv2-DR" and "SDXL-DR" from MSCOCO dataset
    step = 50
    phase = 'train'
    model_name = 'real'
    inpainting_dir = {0: 'full_inpainting', 1: 'full_inpainting2', 2: 'full_inpainting_xl'}[index]
    if step != 50:
        inpainting_dir = f'step{step}_{inpainting_dir}'
    image_root = f'{root}/MSCOCO/{phase}2017'
    save_root = f'{root}/DR/MSCOCO/{inpainting_dir}/{phase}2017'
    crop_root = None

    # Create reconstructed images for the DRCT-2M dataset.
    # step = 50
    # phase = 'val'
    # model_index = 0
    # model_name = DRCT_2M_LIST[model_index]
    # inpainting_dir = {0: 'full_inpainting', 1: 'full_inpainting2', 2: 'full_inpainting_xl'}[index]
    # if step != 50:
    #     inpainting_dir = f'step{step}_{inpainting_dir}'
    # image_root = f'{root}/DRCT-2M/{model_name}/{phase}2017'
    # save_root = f'{root}/DR/DRCT-2M/{model_name}/{inpainting_dir}/{phase}2017'
    # crop_root = None

    # Create reconstructed images for the GenImage dataset.
    # step = 50
    # phase = 'train'
    # label = 'ai'
    # inpainting_dir = {0: 'inpainting', 1: 'inpainting2', 2: 'inpainting_xl'}[index]
    # model_index = 0
    # model_name = GenImage_LIST[model_index]
    # image_root = f'{root}/GenImage/{model_name}/{phase}/{label}'
    # save_root = f'{root}/DR/GenImage/{model_name}/{phase}/{label}/{inpainting_dir}'
    # crop_root = f'{root}/DR/GenImage/{model_name}/{phase}/{label}/crop'

    os.makedirs(save_root, exist_ok=True)
    if crop_root is not None:
        os.makedirs(crop_root, exist_ok=True)
    start_index, end_index = 0, 200000
    image_paths = sorted(glob.glob(f"{image_root}/*.*"))[start_index:end_index]
    print(f'start_index:{start_index}, end_index:{end_index}, {len(image_paths)}, image_root:{model_name}')
    failed_num = 0
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path).split('.')[0]
        save_path = os.path.join(save_root, image_name + '.png')
        crop_save_path = os.path.join(crop_root, image_name + '.png')
        if os.path.exists(save_path):
            if (crop_root is not None and os.path.exists(crop_save_path)) or crop_root is None:
                continue
        try:
            func(image_path, save_path, crop_save_path, step=step, max_size=1024)
        except:
            failed_num += 1
            print(f'Failed to generate image in {image_path}.')
    print(f'Inference finished! start_index:{start_index}, end_index:{end_index}, model_id:{model_name}, failed_num:{failed_num}')