import torch
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import from_torch_complex, to_torch_complex
from .base_models import BaseEncoderMaskerDecoder
from ..masknn.fullsubnet_masker import FullSubMaskNet

class FullSubNet(BaseEncoderMaskerDecoder):
    """Abstract

    Args:
        name (type): description.
    """

    masknet_class = FullSubMaskNet

    def __init__(
        self, *args, stft_n_filters=512, stft_kernel_size=512, stft_stride=256, sample_rate=16000, **masknet_kwargs
    ):
        # 確認用
        # print(stft_n_filters, stft_kernel_size, stft_stride, sample_rate)  # 512, 512, 256, 16000
        # print(args)  # ()
        # print(masknet_kwargs)  # conf.ymlのmasknetと同じ (辞書)

        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=stft_n_filters,
            kernel_size=stft_kernel_size,
            stride=stft_stride,
            sample_rate=sample_rate,
        )
        masker = FullSubMaskNet(**masknet_kwargs)

        super().__init__(encoder, masker, decoder)  # encoder_activation=None

    def forward_encoder(self, wav):
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor (complex), of shape (batch, freq, time).
        """
        tf_rep = self.encoder(wav)
        return to_torch_complex(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): (Amplitude) Time-frequency representation in (batch,
                freq, time).

        Returns:
            torch.Tensor: (Complex) Estimated mask in (batch, freq, time) shape.
        """
        return self.masker(torch.abs(tf_rep))

    def apply_masks(self, tf_rep, est_masks):
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): (Complex) Time-frequency representation in (batch, freq, time) shape.
            est_masks (torch.Tensor): (Complex) Estimated mask in (batch, freq, time) shape.

        Returns:
            torch.Tensor: (Complex) Masked time-frequency representations  in (batch, freq, time) shape.
        """
        masked_tf_rep = est_masks * tf_rep
        return from_torch_complex(masked_tf_rep)