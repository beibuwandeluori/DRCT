import torch
import torch.nn as nn
from torch.nn import init
import timm
from transformers import CLIPModel
import clip

try:
    from .f3net import F3Net
    from .resnet_gram import get_GramNet
except:
    from f3net import F3Net
    from resnet_gram import get_GramNet


# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel // 3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


# TODO 加载训练后的权值时需要设置strict=False
class CLIPVisual(nn.Module):
    def __init__(self, model_name, num_classes=2, freeze_extractor=False):
        super(CLIPVisual, self).__init__()
        model = CLIPModel.from_pretrained(model_name)
        self.visual_model = model.vision_model
        if freeze_extractor:
            self.freeze(self.visual_model)
        self.fc = nn.Linear(in_features=model.vision_embed_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.visual_model(x)
        x = self.fc(x[1])

        return x

    # 冻结网络层
    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


class CLIPModelV2(nn.Module):
    CHANNELS = {
        "RN50": 1024,  #
        "ViT-B/32": 512,
        "ViT-L/14": 768
    }

    def __init__(self, name='clip-RN50', num_classes=2, freeze_extractor=False):
        super(CLIPModelV2, self).__init__()
        name = name.replace('clip-', '').replace('L-', 'L/').replace('B-', 'B/')
        # self.preprecess will not be used during training, which is handled in Dataset class
        self.model, self.preprocess = clip.load(name, device="cpu")
        # 冻结特征提取器
        if freeze_extractor:
            self.freeze(self.model)
            print(f'Freezing the feature extractors!')

        self.fc = nn.Linear(self.CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

    # 冻结网络层
    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


class ContrastiveModels(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, embedding_size=1024,
                 freeze_extractor=False):
        super(ContrastiveModels, self).__init__()
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.model = get_models(model_name=model_name, pretrained=pretrained, num_classes=embedding_size,
                                freeze_extractor=freeze_extractor)
        # self.default_cfg = self.model.default_cfg
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x, return_feature=False):
        feature = self.model(x)
        y_pred = self.fc(feature)
        if return_feature:
            return y_pred, feature

        return y_pred

    def extract_feature(self, x):
        feature = self.model(x)

        return feature


def get_efficientnet_ns(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2, start_down=True):
    """
     # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    :param model_name:
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    if not start_down:
        net.conv_stem.stride = (1, 1)
    n_features = net.classifier.in_features
    net.classifier = nn.Linear(n_features, num_classes)

    return net


def get_swin_transformers(model_name='swin_base_patch4_window7_224', pretrained=True, num_classes=2):
    """
    :param model_name: swin_base_patch4_window12_384   swin_base_patch4_window7_224 swin_base_patch4_window7_224_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.in_features
    net.head = nn.Linear(n_features, num_classes)

    return net


def get_convnext(model_name='convnext_base_in22k', pretrained=True, num_classes=2, in_channel=3):
    """
    :param model_name: convnext_base_384_in22ft1k, convnext_base_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    if in_channel != 3:
        first_conv_weight = net.stem[0].weight
        first_out_channels = net.stem[0].out_channels
        first_conv = nn.Conv2d(in_channel, first_out_channels, kernel_size=4, stride=4)
        first_conv.weight = init_imagenet_weight(first_conv_weight, input_channel=in_channel)
        net.stem[0] = first_conv

    return net


def get_resnet(model_name='resnet200d', pretrained=True, num_classes=2):
    """
    :param model_name: resnet200d, input_size=512, resnet50
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.fc.in_features
    net.fc = nn.Linear(n_features, num_classes)

    return net


def get_clip_visual_model(model_name="openai/clip-vit-base-patch32", num_classes=2, pretrained=True,
                          freeze_extractor=False):
    if 'openai/clip' in model_name:
        model = CLIPVisual(model_name=model_name, num_classes=num_classes)
    else:
        # 'clip-' + 'name', clip-RN50, clip-ViT-L/14
        model = CLIPModelV2(name=model_name, num_classes=num_classes, freeze_extractor=freeze_extractor)

    return model


def get_models(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2,
               in_channel=3, freeze_extractor=False, embedding_size=None):
    if embedding_size is not None and isinstance(embedding_size, int) and embedding_size > 0:
        model = ContrastiveModels(model_name, num_classes, pretrained, embedding_size, freeze_extractor)
    elif 'efficientnet' in model_name:
        model = get_efficientnet_ns(model_name, pretrained, num_classes)
    elif 'convnext' in model_name:
        model = get_convnext(model_name, pretrained, num_classes, in_channel=in_channel)
    elif 'swin' in model_name:
        model = get_swin_transformers(model_name, pretrained, num_classes)
    elif 'clip' in model_name:
        model = get_clip_visual_model(model_name, num_classes, freeze_extractor=freeze_extractor)  # 输入尺寸必须为224
    elif 'swin' in model_name:
        model = get_swin_transformers(model_name, pretrained=pretrained, num_classes=num_classes)
    elif 'gram' in model_name:  # gram_resnet18
        model = get_GramNet(model_name.replace('gram_', ''))
    elif 'resnet' in model_name:
        model = get_resnet(model_name, pretrained, num_classes)
    elif model_name == 'f3net':
        model = F3Net(num_classes=num_classes, img_width=299, img_height=299, pretrained=pretrained)
    else:
        raise NotImplementedError(model_name)

    return model


if __name__ == '__main__':
    import time
    image_size = 224
    model = get_models(model_name='clip-ViT-L-14', num_classes=2, pretrained=False,
                       embedding_size=512)  # clip-ViT-L-14
    print(model)
    # print(model.default_cfg)
    model = model.to(torch.device('cpu'))
    img = torch.randn(1, 3, image_size, image_size)  # your high resolution picture
    start = time.time()
    times = 1
    for _ in range(times):
        out = model(img)
        if isinstance(out, tuple):
            print([o.shape for o in out])
        else:
            print(out.shape)
    print((time.time()-start)/times)

    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s, device='cpu'))
    pass
