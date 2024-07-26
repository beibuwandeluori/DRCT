import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

from data.dataset import AIGCDetectionDataset
from data.transform import create_train_transforms, create_val_transforms


def merge_tensor(img, label, is_train=True):
    def shuffle_tensor(img, label):
        indices = torch.randperm(img.size(0))
        return img[indices], label[indices]
    if isinstance(img, list) and isinstance(label, list):
        img, label = torch.cat(img, dim=0), torch.cat(label, dim=0)
        if is_train:
            img, label = shuffle_tensor(img, label)
    return img, label


def debug_data_loader():
    input_size = 224
    use_label = True
    is_crop = True
    is_dire = False
    phase = 'train'
    fake_indexes = '2'
    inpainting_dir = 'full_inpainting'
    # DRCT-2M
    # root_path = '/disk1/chenby/dataset/AIGC_data/DRCT_data/MSCOCO'
    # fake_root_path = '/disk1/chenby/dataset/AIGC_data/DRCT_data/DRCT-2M'

    # GenImage
    # root_path = '/disk1/chenby/dataset/AIGC_data/DRCT_data/GenImage'
    # fake_root_path = ''

    # DIRE of DRCT-2M
    # inpainting_dir = 'full_inpainting'
    # is_dire = True
    # root_dir = '/disk1/chenby/dataset/AIGC_data/DRCT_data'
    # root_path = f'{root_dir}/MSCOCO,{root_dir}/DR/MSCOCO'
    # fake_root_path = f'{root_dir}/DRCT-2M,{root_dir}/DR/DRCT-2M'

    # DIRE of GenImage
    # fake_indexes = '1'
    # is_dire = True
    # inpainting_dir = 'inpainting'
    # root_dir = '/disk1/chenby/dataset/AIGC_data/DRCT_data'
    # root_path = f'{root_dir}/DR/GenImage'
    # fake_root_path = f'{root_dir}/DR/GenImage'

    # DRCT of DRCT-2M
    # root_dir = '/disk1/chenby/dataset/AIGC_data/DRCT_data'
    # root_path = f'{root_dir}/MSCOCO/train2017'
    # fake_root_path = f'{root_dir}/DR/MSCOCO/full_inpainting/train2017,' \
    #                  f'{root_dir}/DRCT-2M/stable-diffusion-v1-4/train2017,' \
    #                  f'{root_dir}/DR/DRCT-2M/stable-diffusion-v1-4/full_inpainting/train2017'

    # DRCT of GenImage
    fake_indexes = '1'
    is_dire = False
    root_dir = '/disk1/chenby/dataset/AIGC_data/DRCT_data'
    root_path = f'{root_dir}/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/nature'
    fake_root_path = f'{root_dir}/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature/inpainting,' \
                     f'{root_dir}/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/ai,' \
                     f'{root_dir}/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai/inpainting'

    transform = create_train_transforms(size=input_size, is_crop=is_crop)
    xdl = AIGCDetectionDataset(root_path=root_path, fake_root_path=fake_root_path, phase=phase, use_label=use_label,
                               transform=transform, fake_indexes=fake_indexes, is_dire=is_dire,
                               inpainting_dir=inpainting_dir,
                               )
    train_loader = DataLoader(xdl, batch_size=32, shuffle=True, num_workers=8)
    print('length:', len(xdl), f'iter:{len(train_loader)}')
    start = time.time()
    times = 10
    for i, data in enumerate(train_loader):
        if use_label:
            (img, label) = data
            if isinstance(img, list) and not isinstance(label, list):
                print(i, img[0].size(), img[1].size(), label.size(), label)
            else:
                img, label = merge_tensor(img, label, is_train=True)
                print(i, img.size(), label.size(), label)
        else:
            img, image_name = data
            if isinstance(img, list):
                print(f'{i + 1}/{len(train_loader)}', img[0].size(), img[1].size(), len(image_name))
            else:
                print(f'{i + 1}/{len(train_loader)}', img.size(), len(image_name))
        if i == times - 1:
            break
    end = time.time()
    QPS = times / (end - start)
    print(f"run_time:{end - start}s, QPS:{QPS}")


def show_image():
    import cv2
    from data.dataset import read_image
    image_path = 'data/samples/01.png'
    img, _ = read_image(image_path)
    is_success, img_bytes = cv2.imencode(".png", img)  # .jpg有损，.png无损
    img_bytes = img_bytes.tobytes()

    # 将 bytes 类型的图片数据恢复为 OpenCV 图像
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print(img.shape)
    diff = np.abs(img - img_decoded)
    print(np.mean(diff), np.max(diff))


def test_post_aug():
    from data.dataset import read_image, cv2_jpg
    import matplotlib.pyplot as plt
    image_path = 'data/samples/01.png'
    img, _ = read_image(image_path)
    img_aug = cv2_jpg(img, compress_val=100)
    diff = np.abs(img - img_aug).astype(np.uint8)
    # print(np.mean(diff), np.max(diff))

    # 创建一个新的figure
    plt.figure(figsize=(15, 5))

    # 画出原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    # 画出后处理的图像
    plt.subplot(1, 3, 2)
    plt.imshow(img_aug)
    plt.title('Image after Post Aug.')
    # 画出后处理前后的图像差异
    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.title('Diff after Post Aug.')
    # 显示图像
    plt.show()


if __name__ == '__main__':
    debug_data_loader()
