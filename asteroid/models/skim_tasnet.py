from asteroid_filterbanks import make_enc_dec
from ..masknn import SkiM
from .base_models import BaseEncoderMaskerDecoder


class SkiMTasNet(BaseEncoderMaskerDecoder):
    """SKiM separation model
    """

    def __init__(
        self,
        n_src,
        enc_size,
        seg_input_size,
        seg_hidden_size,
        mem_hidden_size,
        dropout=0.0,
        seg_bidirectional=False,
        mem_bidirectional=False,
        seg_rnn_type="LSTM",
        seg_norm_type="cLN",
        mem_rnn_type="LSTM",
        mem_norm_type="cLN",
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        mask_act="relu",
        fb_name="free",
        kernel_size=16,
        n_filters=64,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if enc_size is not None:
            assert enc_size == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {enc_size}"
            )
        else:
            enc_size = n_feats

        # Update in_chan
        masker = SkiM(
            n_src=n_src,
            enc_size=enc_size,
            seg_input_size=seg_input_size,
            seg_hidden_size=seg_hidden_size,
            mem_hidden_size=mem_hidden_size,
            dropout=dropout,
            seg_bidirectional=seg_bidirectional,
            mem_bidirectional=mem_bidirectional,
            seg_rnn_type=seg_rnn_type,
            seg_norm_type=seg_norm_type,
            mem_rnn_type=mem_rnn_type,
            mem_norm_type=mem_norm_type,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            mask_act=mask_act,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
