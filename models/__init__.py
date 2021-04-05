from __future__ import absolute_import

# from .DGFANet import *
# from .DGFANetCustom import *
# from .DGFANetLambda import *
# from .DGFANetLambda_v2 import *
from .EfficientMetaNet import *
from models import geffnet
__factory = {
    'FeatExtractor': FeatExtractor,
    'FeatEmbedder': FeatEmbedder,
    'DepthEstmator': DepthEstmator,
}


def names():
    return sorted(__factory.keys())


def create(name, pretrain = True, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    eff_name = 'efficientnet-b0'
    if name == "Eff_FeatExtractor":
        if pretrain:
            # return FeatExtractor.from_pretrained(eff_name,in_channels=3)#dct mode
            return FeatExtractor.from_pretrained(eff_name,in_channels=6)#rgb mode
        else:
            return FeatExtractor.from_name(eff_name,in_channels=6)

    elif name == "Eff_FeatEmbedder":
        if pretrain:
            return FeatEmbedder.from_pretrained(eff_name)#,num_classes=1)
        else:
            return FeatEmbedder.from_name(eff_name)#arcface, multi class
            # return FeatEmbedder.from_name(eff_name,num_classes=1)

    elif name == "Eff_DepthEstmator":
        if eff_name == 'efficientnet-b4':
            return DepthEstmator(48)
        elif eff_name == 'efficientnet-b0':
            return DepthEstmator(32)
    
    elif name == "Eff_lite":
        if pretrain:
            model = geffnet.efficientnet_lite0(pretrained=True)
        else:
            model = geffnet.efficientnet_lite0(pretrained=False)
        return model

    elif name == "Eff_lite_FeatExtractor":
        if pretrain:
            model = geffnet.efficientnet_lite0_extr(pretrained=True)
        else:
            model = geffnet.efficientnet_lite0_extr(pretrained=False)
        return model

    elif name == "Eff_lite_FeatEmbedder":
        if pretrain:
            model = geffnet.efficientnet_lite0_embd(pretrained=True)
        else:
            model = geffnet.efficientnet_lite0_embd(pretrained=False)
        return model

    elif name == "Eff_b4_lite_FeatExtractor":
        if pretrain:
            model = geffnet.efficientnet_lite4_extr(pretrained=False)
        else:
            model = geffnet.efficientnet_lite4_extr(pretrained=False)
        return model

    elif name == "Eff_b4_lite_FeatEmbedder":
        if pretrain:
            model = geffnet.efficientnet_lite4_embd(pretrained=False)
        else:
            model = geffnet.efficientnet_lite4_embd(pretrained=False)
        return model

    elif name not in __factory:
        raise KeyError("Unknown model:", name)

    else:
        return __factory[name](*args, **kwargs)
