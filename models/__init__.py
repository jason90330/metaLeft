from __future__ import absolute_import

# from .DGFANet import *
# from .DGFANetCustom import *
# from .DGFANetLambda import *
# from .DGFANetLambda_v2 import *
from .EfficientMetaNet import *

__factory = {
    'FeatExtractor': FeatExtractor,
    'FeatEmbedder': FeatEmbedder,
    'DepthEstmator': DepthEstmator,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
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
        # return FeatExtractor.from_name(eff_name,in_channels=6)
        # return FeatExtractor.from_pretrained(eff_name,in_channels=6)#rgb mode
        return FeatExtractor.from_pretrained(eff_name,in_channels=3)#dct mode

    elif name == "Eff_FeatEmbedder":
        # return FeatEmbedder.from_name(eff_name)#arcface, multi class
        return FeatEmbedder.from_pretrained(eff_name)#,num_classes=1)
        # return FeatEmbedder.from_name(eff_name,num_classes=1)

    elif name == "Eff_DepthEstmator":
        if eff_name == 'efficientnet-b4':
            return DepthEstmator(48)
        elif eff_name == 'efficientnet-b0':
            return DepthEstmator(32)

    elif name not in __factory:
        raise KeyError("Unknown model:", name)

    else:
        return __factory[name](*args, **kwargs)
