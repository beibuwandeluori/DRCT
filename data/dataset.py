import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import os
import torch
import glob
import json
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.filters import gaussian_filter
try:
    from transform import create_train_transforms, create_val_transforms, create_sdie_transforms
except:
    from .transform import create_train_transforms, create_val_transforms, create_sdie_transforms

# AIGC类别映射
CLASS2LABEL_MAPPING = {
    'real': 0,  # 正常图像, MSCOCO, ImageNet等
    'ldm-text2im-large-256': 1,  # 'CompVis/ldm-text2im-large-256': 'Latent Diffusion',  # Latent Diffusion 基础版本
    'stable-diffusion-v1-4': 2,  # 'CompVis/stable-diffusion-v1-4': 'Stable Diffusion',  # 现实版本
    'stable-diffusion-v1-5': 3,  # 'runwayml/stable-diffusion-v1-5': 'Stable Diffusion',  # 现实版本
    'stable-diffusion-2-1': 4,
    'stable-diffusion-xl-base-1.0': 5,
    'stable-diffusion-xl-refiner-1.0': 6,
    'sd-turbo': 7,
    'sdxl-turbo': 8,
    'lcm-lora-sdv1-5': 9,
    'lcm-lora-sdxl': 10,
    'sd-controlnet-canny': 11,
    'sd21-controlnet-canny': 12,
    'controlnet-canny-sdxl-1.0': 13,
    'stable-diffusion-inpainting': 14,
    'stable-diffusion-2-inpainting': 15,
    'stable-diffusion-xl-1.0-inpainting-0.1': 16,
}
LABEL2CLASS_MAPPING = {CLASS2LABEL_MAPPING.get(key): key for key in CLASS2LABEL_MAPPING.keys()}
GenImage_LIST = ['stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
                 'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
                 'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan']

# 抗JPEG压缩后处理测试
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


# 抗缩放后处理测试
def cv2_scale(img, scale):
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    return resized_img


# 保持长宽比resize
def resize_long_size(img, long_size=512):
    scale_percent = long_size / max(img.shape[0], img.shape[1])

    # 计算新的高度和宽度
    new_width = int(img.shape[1] * scale_percent)
    new_height = int(img.shape[0] * scale_percent)

    # 调整大小
    img_resized = cv2.resize(img, (new_width, new_height))

    return img_resized


def read_image(image_path, resize_size=None):
    try:
        image = cv2.imread(image_path)
        if resize_size is not None:
            image = resize_long_size(image, long_size=resize_size)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, True
    except:
        print(f'{image_path} read error!!!')
        return np.zeros(shape=(512, 512, 3), dtype=np.uint8), False


# 同步对应打乱两个数组
def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b


# 把标签转换为one-hot格式
def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


# 数据划分
def split_data(image_paths, labels, val_split=0.1, test_split=0.0, phase='train', seed=2022):
    image_paths, labels = shuffle_two_array(image_paths, labels, seed=seed)
    total_len = len(image_paths)
    if test_split > 0:
        if phase == 'train':
            start_index, end_index = 0, int(total_len * (1 - val_split - test_split))
        elif phase == 'val':
            start_index, end_index = int(total_len * (1 - val_split - test_split)), int(total_len * (1 - test_split))
        else:
            start_index, end_index = int(total_len * (1 - test_split)), total_len
    else:
        if phase == 'train':
            start_index, end_index = 0, int(total_len * (1 - val_split))
        else:
            start_index, end_index = int(total_len * (1 - val_split)), total_len
    # print(f'{phase} start_index-end_index:{start_index}-{end_index}')
    image_paths, labels = image_paths[start_index:end_index], labels[start_index:end_index]

    return image_paths, labels


def split_dir(image_dirs, val_split=0.1, phase='train', seed=2022):
    if phase == 'all':
        return image_dirs
    image_dirs, _ = shuffle_two_array(image_dirs, image_dirs, seed=seed)
    total_len = len(image_dirs)
    if phase == 'train':
        start_index, end_index = 0, int(total_len * (1 - val_split * 2))
    elif phase == 'val':
        start_index, end_index = int(total_len * (1 - val_split * 2)), int(total_len * (1 - val_split))
    else:
        start_index, end_index = int(total_len * (1 - val_split)), total_len
    image_dirs = image_dirs[start_index:end_index]

    return image_dirs


# 获取所有图像文件
def find_images(dir_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']):
    image_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.basename(file).startswith("._"):  # skip files that start with "._"
                continue
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files


# Calculate the DIRE
def calculate_dire(img, sdir_img, is_success=True, input_size=224, phase='train'):
    if not is_success:
        return torch.zeros(size=(3, input_size, input_size), dtype=torch.float32)
    sdie_transforms = create_sdie_transforms(size=input_size, phase=phase)

    data = sdie_transforms(image=img, rec_image=sdir_img)
    img, sdir_img = data['image'], data['rec_image']

    # norm [0,255] -> [-1, 1]
    img = img / 127.5 - 1
    sdir_img = sdir_img / 127.5 - 1
    # absolute error
    sdie = np.abs(img - sdir_img)
    sdie_tensor = torch.from_numpy(np.transpose(np.array(sdie, dtype=np.float32), [2, 0, 1]))

    return sdie_tensor


# 根据文件路径获取图片的类别名字
def get_class_name_by_path(image_path):
    if 'GenImage' in image_path:
        class_names = GenImage_LIST
        class_name = class_names[0]
        for name in class_names[1:]:
            if f'/{name}/' in image_path:
                class_name = name
                break
    else:
        class_name = 'real'
        class_names = list(CLASS2LABEL_MAPPING.keys())
        for name in class_names:
            if f'/{name}/' in image_path:
                class_name = name
                break

    return class_name


def load_DRCT_2M(real_root_path='/disk4/chenby/dataset/MSCOCO',
                 fake_root_path='/disk4/chenby/dataset/AIGC_MSCOCO',
                 fake_indexes='1,2,3,4,5,6', phase='train', val_split=0.1,
                 seed=2022):
    fake_indexes = [int(index) for index in fake_indexes.split(',')]
    if phase != 'test':  # 训练集和验证机按照 9：1 划分
        real_paths = sorted(glob.glob(f"{real_root_path}/train2017/*.*"))
        real_labels = [0 for _ in range(len(real_paths))]
        real_paths, real_labels = split_data(real_paths, real_labels, val_split=val_split, phase=phase, seed=seed)
        fake_paths = []
        fake_labels = []
        for i, index in enumerate(fake_indexes):
            fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/train2017/*.*"))
            fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]
            fake_paths_t, fake_labels_t = split_data(fake_paths_t, fake_labels_t, val_split=val_split, phase=phase,
                                                     seed=seed)
            fake_paths += fake_paths_t
            fake_labels += fake_labels_t
    else:  # 把所有的val2017当最终的测试集
        real_paths = sorted(glob.glob(f"{real_root_path}/val2017/*.*"))
        real_labels = [0 for _ in range(len(real_paths))]
        fake_paths = []
        fake_labels = []
        for i, index in enumerate(fake_indexes):
            fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/val2017/*.*"))
            fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]
            fake_paths += fake_paths_t
            fake_labels += fake_labels_t
    image_paths = real_paths + fake_paths
    labels = real_labels + fake_labels

    # 各个类别数量统计
    class_count_mapping = {cls: 0 for cls in range(len(fake_indexes) + 1)}
    for label in labels:
        class_count_mapping[label] += 1
    class_name_mapping = {0: 'real'}
    for i, fake_index in enumerate(fake_indexes):
        class_name_mapping[i + 1] = LABEL2CLASS_MAPPING[fake_index]
    print(f"{phase}:{class_count_mapping}, total:{len(image_paths)}, class_name_mapping:{class_name_mapping}")

    return image_paths, labels


def load_normal_data(root_path, val_split, seed, phase='train', regex='*.*', test_all=False):
    images_t = sorted(glob.glob(f'{root_path}/{regex}'))
    if not test_all:
        images_t, _ = split_data(images_t, images_t, val_split=val_split, phase=phase, seed=seed)

    captions_t = [' ' for _ in images_t]
    print(f'{root_path}: {len(images_t)}')
    return images_t, captions_t


def load_GenImage(root_path='/disk1/chenby/dataset/AIGC_data/GenImage', phase='train', seed=2023,
                  indexes='1,2,3,4,5,6,7,8', val_split=0.1):

    indexes = [int(i) - 1 for i in indexes.split(',')]
    dir_list = GenImage_LIST
    selected_dir_list = [dir_list[i] for i in indexes]
    real_images, real_labels, fake_images, fake_labels = [], [], [], []
    dir_phase = 'train' if phase != 'test' else 'val'
    for i, selected_dir in enumerate(selected_dir_list):
        real_root = os.path.join(root_path, selected_dir, dir_phase, 'nature')
        fake_root = os.path.join(root_path, selected_dir, dir_phase, 'ai')
        real_images_t = sorted(glob.glob(f'{real_root}/*.*'))
        fake_images_t = sorted(glob.glob(f'{fake_root}/*.*'))
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
            fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
        real_images += real_images_t
        real_labels += [0 for _ in real_images_t]
        fake_images += fake_images_t
        fake_labels += [i + 1 for _ in fake_images_t]
        print(f'phase:{phase}, real:{len(real_images_t)}, fake-{i+1}:{len(fake_images_t)}, selected_dir:{selected_dir}')
    total_images = real_images + fake_images
    # labels = [0 for _ in real_images] + [1 for _ in fake_images]
    labels = real_labels + fake_labels
    print(f'phase:{phase}, real:{len(real_images)}, fake:{len(fake_images)}')

    return total_images, labels


def load_data(real_root_path, fake_root_path,
              phase='train', val_split=0.1, seed=2022, ):
    # load real images
    total_real_images, total_real_captions = [], []
    for real_root in real_root_path.split(','):
        real_images_t, real_captions_t = load_normal_data(real_root, val_split, seed, phase)
        total_real_images += list(real_images_t)
        total_real_captions += list(real_captions_t)
    # load fake images
    total_fake_images, total_fake_captions = [], []
    for fake_root in fake_root_path.split(','):
        fake_images_t, fake_captions_t = load_normal_data(fake_root, val_split, seed, phase)
        total_fake_images += list(fake_images_t)
        total_fake_captions += list(fake_captions_t)
    # 合并
    image_paths = total_real_images + total_fake_images
    labels = [0 for _ in total_real_images] + [1 for _ in total_fake_images]
    print(f'{phase}-total:{len(image_paths)}, real:{len(total_real_images)},fake:{len(total_fake_images)}')

    return image_paths, labels


def load_pair_data(root_path, fake_root_path=None, phase='train', seed=2023, fake_indexes='1',
                   inpainting_dir='full_inpainting'):
    if fake_root_path is None:  # 推理加载代码，或者用于特征提取
        assert len(root_path.split(',')) == 2
        root_path, rec_root_path = root_path.split(',')[:2]
        image_paths = sorted(glob.glob(f"{root_path}/*.*"))
        rec_image_paths = sorted(glob.glob(f"{rec_root_path}/*.*"))
        assert len(image_paths) == len(rec_image_paths)
        total_paths = []
        for image_path, rec_image_path in zip(image_paths, rec_image_paths):
            total_paths.append((image_path, rec_image_path))
        print(f'Pair data-{phase}:{len(total_paths)}.')
        return total_paths
    assert (len(root_path.split(',')) == 2 and len(fake_root_path.split(',')) == 2) or \
           (root_path == fake_root_path and 'GenImage' in root_path)
    if 'MSCOCO' in root_path:
        phase_mapping = {'train': 'train2017', 'val': 'train2017', 'test': 'val2017'}
        real_root, real_rec_root = root_path.split(',')[:2]
        real_root = f'{real_root}/{phase_mapping[phase]}'
        real_rec_root = f'{real_rec_root}/{inpainting_dir}/{phase_mapping[phase]}'
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
        fake_root = f'{fake_root}/{LABEL2CLASS_MAPPING[int(fake_indexes)]}/{phase_mapping[phase]}'
        fake_rec_root = f'{fake_rec_root}/{LABEL2CLASS_MAPPING[int(fake_indexes)]}/{inpainting_dir}/{phase_mapping[phase]}'
        print(f'fake_name:{LABEL2CLASS_MAPPING[int(fake_indexes)]}')
    elif 'DR/GenImage' in root_path:
        phase_mapping = {'train': 'train', 'val': 'train', 'test': 'val'}
        fake_indexes = int(fake_indexes)
        assert 1 <= fake_indexes <= 8 and inpainting_dir in ['inpainting', 'inpainting2', 'inpainting_xl']
        fake_name = GenImage_LIST[fake_indexes-1]
        real_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/crop'
        real_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/{inpainting_dir}'
        fake_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/crop'
        fake_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/{inpainting_dir}'
        print(f'fake_name:{fake_name}')
        # print(real_root, real_rec_root, fake_root, fake_rec_root)
    else:
        real_root, real_rec_root = root_path.split(',')[:2]
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
    image_paths, labels = [], []
    # load real images
    real_image_paths = sorted(glob.glob(f"{real_root}/*.*"))
    real_image_rec_paths = sorted(glob.glob(f"{real_rec_root}/*.*"))
    assert len(real_image_paths) == len(real_image_rec_paths) and len(real_image_paths) > 0
    total_real = len(real_image_paths)
    if phase != 'test':
        real_image_paths, real_image_rec_paths = split_data(real_image_paths, real_image_rec_paths, phase=phase, seed=seed)
    for real_image_path, real_image_rec_path in zip(real_image_paths, real_image_rec_paths):
        image_paths.append((real_image_path, real_image_rec_path))
    # load fake images
    fake_image_paths = sorted(glob.glob(f"{fake_root}/*.*"))
    fake_image_rec_paths = sorted(glob.glob(f"{fake_rec_root}/*.*"))
    assert len(fake_image_paths) == len(fake_image_rec_paths) and len(fake_image_paths) > 0
    total_fake = len(fake_image_paths)
    if phase != 'test':
        fake_image_paths, fake_image_rec_paths  = split_data(fake_image_paths, fake_image_rec_paths, phase=phase, seed=seed)
    for fake_image_path, fake_image_rec_path in zip(fake_image_paths, fake_image_rec_paths):
        image_paths.append((fake_image_path, fake_image_rec_path))
    labels = [0 for _ in range(len(real_image_paths))] + [1 for _ in range(len(fake_image_paths))]
    print(f'Phase:{phase}, real:{len(real_image_paths)}, fake:{len(fake_image_paths)},'
          f'Total real:{total_real}, fake:{total_fake}')

    return image_paths, labels


class AIGCDetectionDataset(Dataset):
    def __init__(self, root_path='/disk4/chenby/dataset/MSCOCO', fake_root_path='/disk4/chenby/dataset/DRCT-2M',
                 fake_indexes='1,2,3,4,5,6', phase='train', is_one_hot=False, seed=2021,
                 transform=None, use_label=True, num_classes=None, regex='*.*',
                 is_dire=False, inpainting_dir='full_inpainting', post_aug_mode=None):
        self.root_path = root_path  # real 图像的根目录
        self.phase = phase
        self.is_one_hot = is_one_hot
        self.num_classes = len(fake_indexes.split(',')) + 1 if num_classes is None else num_classes
        self.transform = transform
        self.use_label = use_label
        self.regex = regex  # 数据过滤的正则表达式
        self.is_dire = is_dire  # training by DIRE
        self.post_aug_mode = post_aug_mode  # 抗后处理测试模式：[blur_1, blur_2, blur_3, blur4, jpeg_30, jpeg_40, ..., jpeg_100]
        self.seed = seed

        if use_label:
            if self.is_dire:
                # load DR data
                self.image_paths, self.labels = load_pair_data(root_path, fake_root_path, phase,
                                                               fake_indexes=fake_indexes,
                                                               inpainting_dir=inpainting_dir)
            elif 'MSCOCO' in root_path and len(fake_root_path.split(',')) == 1:
                self.image_paths, self.labels = load_DRCT_2M(real_root_path=root_path,
                                                             fake_root_path=fake_root_path,
                                                             fake_indexes=fake_indexes, phase=phase, seed=seed)
            elif 'GenImage' in root_path and fake_root_path == '':
                self.image_paths, self.labels = load_GenImage(root_path=root_path, phase=phase, seed=seed,
                                                              indexes=fake_indexes)
            else:
                self.image_paths, self.labels = load_data(real_root_path=root_path, fake_root_path=fake_root_path,
                                                          phase=phase, seed=seed)

            self.labels = [int(label > 0)for label in self.labels] if self.num_classes == 2 else self.labels
        else:
            if len(root_path.split(',')) == 2 and 'DR' in root_path:
                self.is_dire = True
                self.image_paths = load_pair_data(root_path, phase=phase, fake_indexes=fake_indexes,
                                                  inpainting_dir=inpainting_dir)
            else:
                if self.regex == 'all':
                    self.image_paths = sorted(find_images(dir_path=root_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']))
                else:
                    self.image_paths = sorted(glob.glob(f'{root_path}/{self.regex}'))[:]
            print(f'Total predict images:{len(self.image_paths)}, regex:{self.regex}')
        if self.phase == 'test' and self.post_aug_mode is not None:
            print(f"post_aug_mode:{self.post_aug_mode}, {self.post_aug_mode.split('_')[1]}")

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return list(self.labels)

    def __getitem__(self, index):
        if not self.is_dire:
            image_path = self.image_paths[index]
            image, is_success = read_image(image_path)
        else:
            image_path, rec_image_path = self.image_paths[index]
            image, is_success = read_image(image_path)
            rec_image, rec_is_success = read_image(rec_image_path)
            is_success = is_success and rec_is_success
            image = calculate_dire(image, rec_image, phase=self.phase)

        # 测试后处理攻击
        if self.phase == 'test' and self.post_aug_mode is not None:
            if 'jpeg' in self.post_aug_mode:
                compress_val = int(self.post_aug_mode.split('_')[1])
                image = cv2_jpg(image, compress_val)
            elif 'scale' in self.post_aug_mode:
                scale = float(self.post_aug_mode.split('_')[1])
                image = cv2_scale(image, scale)

        label = 0  # default label
        if self.use_label:
            label = self.labels[index] if is_success else 0

        if self.transform is not None and not self.is_dire:
            try:
                if isinstance(self.transform, torchvision.transforms.transforms.Compose):
                    image = self.transform(Image.fromarray(image))
                else:
                    data = self.transform(image=image)
                    image = data["image"]
            except:
                print("transform error!!!")
                image = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
                if isinstance(self.transform, torchvision.transforms.transforms.Compose):
                    image = self.transform(Image.fromarray(image))
                else:
                    data = self.transform(image=image)
                    image = data["image"]
                label = 0

        if not self.use_label:
            return image, image_path.replace(f"{self.root_path}", '')  # os.path.basename(image_path)

        if self.is_one_hot:
            label = one_hot(self.num_classes, label)

        return image, label




