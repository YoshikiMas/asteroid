from .base_models import BaseEncoderMaskerDecoder


class FullSubNet(BaseEncoderMaskerDecoder):
    """Abstract

    Args:
        name (type): description.
    """

    masknet_class = NotImplemented