import argparse
import warnings
import sys
import os


def get_parser():
    parser = argparse.ArgumentParser(description="AIGCDetection @cby Training")
    parser.add_argument("--model_name", default='efficientnet-b0', help="Setting the model name", type=str)
    parser.add_argument("--embedding_size", default=None, help="Setting the embedding_size", type=int)
    parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
    parser.add_argument('--freeze_extractor', action='store_true', help='Whether to freeze extractor?')
    parser.add_argument("--model_path", default=None, help="Setting the model path", type=str)
    parser.add_argument('--no_strict', action='store_true', help='Whether to load model without strict?')
    parser.add_argument("--root_path", default='/disk4/chenby/dataset/MSCOCO',
                        help="Setting the root path for dataset loader", type=str)
    parser.add_argument("--fake_root_path", default='/disk4/chenby/dataset/DRCT-2M',
                        help="Setting the fake root path for dataset loader", type=str)
    parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
    parser.add_argument("--regex", default='*.*', help="Setting the regex for dataset loader", type=str)
    parser.add_argument('--test_all', action='store_true', help='Whether to test_all?')
    parser.add_argument('--post_aug_mode', default=None, help='Stetting the post aug mode during test phase.')
    parser.add_argument('--save_txt', default=None, help='Stetting the save_txt path.')
    parser.add_argument("--fake_indexes", default='1',
                        help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
    parser.add_argument("--dataset_name", default='MSCOCO', help="Setting the dataset name", type=str)
    parser.add_argument("--device_id", default='0',
                        help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')
    parser.add_argument("--batch_size", default=64, help="Setting the batch size", type=int)
    parser.add_argument("--epoch_start", default=0, help="Setting the epoch start", type=int)
    parser.add_argument("--num_epochs", default=50, help="Setting the num epochs", type=int)
    parser.add_argument("--num_workers", default=4, help="Setting the num workers", type=int)
    parser.add_argument('--is_warmup', action='store_true', help='Whether to using lr warmup')
    parser.add_argument("--lr", default=1e-3, help="Setting the learning rate", type=float)
    parser.add_argument("--save_flag", default='', help="Setting the save flag", type=str)
    parser.add_argument("--sampler_mode", default='', help="Setting the sampler mode", type=str)
    parser.add_argument('--is_test', action='store_true', help='Whether to predict the test set?')
    parser.add_argument('--is_amp', action='store_true', help='Whether to using amp autocast(使用混合精度加速)?')
    parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
    parser.add_argument("--threshold", default=0.5, help="Setting the valid or testing threshold.", type=float)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


warnings.filterwarnings("ignore")
sys.path.append('..')
args = get_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

import torch
import torch.nn as nn
import torch.optim as optim
from catalyst.data import BalanceClassSampler
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
import pytorch_warmup as warmup

from utils.utils import Logger, AverageMeter, Test_time_agumentation, calculate_fnr
from network.models import get_models
from data.dataset import AIGCDetectionDataset, CLASS2LABEL_MAPPING, GenImage_LIST
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

# 9 times
def TTA(model_, img, activation=nn.Softmax(dim=1)):
    # original 1
    outputs = activation(model_(img))
    tta = Test_time_agumentation()
    # 水平翻转 + 垂直翻转 2
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += activation(model_(flip_img))
    # 2*3=6
    for flip_img in [img, flip_imgs[0]]:
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            outputs += activation(model_(rot_flip_img))

    outputs /= 9

    return outputs


def eval_model(model, epoch, eval_loader, is_save=True, is_tta=False, threshold=0.5, save_txt=None):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)
    labels = []
    outputs = []
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            img, label = merge_tensor(img, label, is_train=False)
            if i > 0 and i % 1 == 0:
                eval_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg, accuracies.avg))
            img, label = img.cuda(), label.cuda()
            if not is_tta:
                y_pred = model(img)
                y_pred = nn.Softmax(dim=1)(y_pred)
            else:
                y_pred = TTA(model, img, activation=nn.Softmax(dim=1))
            outputs.append(1 - y_pred[:, 0])
            labels.append(label)
            loss = criterion(y_pred, label)
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / img.size(0)

            losses.update(loss.item(), img.size(0))
            accuracies.update(acc, img.size(0))

    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    labels[labels > 0] = 1
    auc = roc_auc_score(labels, outputs)
    recall = recall_score(labels, outputs > threshold)
    precision = precision_score(labels, outputs > threshold)
    binary_acc = accuracy_score(labels, outputs > threshold)
    f1 = f1_score(labels, outputs > threshold)
    fnr = calculate_fnr(labels, outputs > threshold)
    print(f'AUC:{auc}-Recall:{recall}-Precision:{precision}-BinaryAccuracy:{binary_acc}, f1: {f1}, fnr:{fnr}')
    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(accuracies.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    acc_avg = accuracies.avg
    # 垃圾回收
    del outputs, labels, losses, accuracies
    gc.collect()

    if save_txt is not None:
        return binary_acc, auc, recall, precision, f1, fnr
    return acc_avg


def train_model(model, criterion, optimizer, epoch, scaler=None):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)
    for i, (x, label) in enumerate(training_process):
        x, label = merge_tensor(x, label, is_train=True)
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]['lr']
        if i > 0 and i % 1 == 0:
            training_process.set_description(
                "Epoch: %d, LR: %.8f, Loss: %.4f, Acc: %.4f" % (
                    epoch, current_lr, losses.avg, accuracies.avg))

        x = x.cuda()
        label = label.cuda()
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the model
        if scaler is None:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, label)
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / x.size(0)

            losses.update(loss.item(), x.size(0))  # loss -> loss.item(), 减少内存泄漏风险
            accuracies.update(acc, x.size(0))

            loss.backward()
            optimizer.step()
        else:
            with autocast():
                y_pred = model(x)
                # Compute and print loss
                loss = criterion(y_pred, label)
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / x.size(0)

            losses.update(loss.item(), x.size(0))
            accuracies.update(acc, x.size(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if args.is_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
    if not args.is_warmup:
        scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(accuracies.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    # 垃圾回收
    del losses, accuracies
    gc.collect()


# python train.py --device_id=0 --model_name=efficientnet-b0 --input_size=224 --batch_size=48 --fake_indexes=1 --is_amp --save_flag=
if __name__ == '__main__':
    batch_size = args.batch_size * torch.cuda.device_count()
    writeFile = f"../output/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
                f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"
    store_name = writeFile.replace('/logs', '/weights')
    print(f'Using gpus:{args.device_id},batch size:{batch_size},gpu_count:{torch.cuda.device_count()},num_classes:{args.num_classes}')
    # Load model
    model = get_models(model_name=args.model_name, num_classes=args.num_classes,
                       freeze_extractor=args.freeze_extractor, embedding_size=args.embedding_size)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=not args.no_strict)
        print('Model found in {}'.format(args.model_path))
    else:
        print('No model found, initializing random model.')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)
    is_train = not args.is_test
    if is_train:
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        # setting data loader
        xdl = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='train',
                                   num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                   transform=create_train_transforms(size=args.input_size, is_crop=args.is_crop)
                                   )
        sampler = BalanceClassSampler(labels=xdl.get_labels(), mode=args.sampler_mode) if args.sampler_mode != '' else None  # "upsampling"
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=sampler is None, num_workers=args.num_workers, sampler=sampler)
        train_dataset_len = len(xdl)

        xdl_eval = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='val',
                                        num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        )
        eval_loader = DataLoader(xdl_eval, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        if not args.is_warmup:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        else:
            num_steps = train_dataset_len * args.num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        best_acc = 0.5 if args.epoch_start == 0 else eval_model(model, args.epoch_start - 1, eval_loader, is_save=False)
        for epoch in range(args.epoch_start, args.num_epochs):
            train_model(model, criterion, optimizer, epoch, scaler=GradScaler() if args.is_amp else None)
            if epoch % 1 == 0 or epoch == args.num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader)
                if best_acc < acc:
                    best_acc = acc
                    save_path = '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc)
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
            print(f'Current best acc:{best_acc}')
        last_save_path = '{}/last_acc{:.4f}.pth'.format(store_name, acc)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_save_path)
        else:
            torch.save(model.state_dict(), last_save_path)
    else:
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes,
                                        phase='test', num_classes=args.num_classes, is_dire=args.is_dire,
                                        post_aug_mode=args.post_aug_mode, regex=args.regex, inpainting_dir=args.inpainting_dir,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        )
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        out_metrics = eval_model(model, epoch_start, test_loader, is_save=False, is_tta=False,
                                 threshold=args.threshold, save_txt=args.save_txt)
        print('Total time:', time.time() - start)
        # 保存测试结果
        if args.save_txt is not None:
            os.makedirs(os.path.dirname(args.save_txt), exist_ok=True)
            acc, auc, recall, precision, f1, fnr = out_metrics
            with open(args.save_txt, 'a') as file:
                if args.dataset_name == 'GenImage':
                    class_name = GenImage_LIST[int(args.fake_indexes)-1]
                else:
                    class_name = list(CLASS2LABEL_MAPPING.keys())[int(args.fake_indexes)]
                result_str = f'model_path:{args.model_path}, post_aug_mode:{args.post_aug_mode}, class_name:{class_name}\n' \
                             f'acc:{acc:.4f}, auc:{auc:.4f}, recall:{recall:.4f}, precision:{precision:.4f}, ' \
                             f'f1:{f1:.4f}, fnr: {fnr}\n'
                file.write(result_str)
            print(f'The result was saved in {args.save_txt}')

