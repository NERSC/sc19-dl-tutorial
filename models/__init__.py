"""
Keras example model factory functions.
"""

def get_model(name, **model_args):
    if name == 'cnn':
        from .cnn import build_model
        return build_model(**model_args)
    elif name == 'resnet_small':
        from .resnet import ResNetSmall
        return ResNetSmall(**model_args)
    elif name == 'resnet50':
        from .resnet import ResNet50
        return ResNet50(**model_args)
    else:
        raise ValueError('Model %s unknown' % name)
