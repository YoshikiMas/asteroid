"""
An implementation of SkiM model described in
"SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation"
(https://arxiv.org/abs/2201.10800)

This script is based on the SkiM implementation in ESPNet
and modified for compatibility with Asteroid
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.functional import fold, unfold
import numpy as np

from ..utils import has_arg
from . import activations, norms


class CausalConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, bias=True):
        super().__init__()
        self.pad_size = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad_size, 0))
        x = self.conv(x)
        return x

class SkiMRNN(nn.Module):
    """Module for a RNN block for SKiM
    """

    def __init__(
        self,
        rnn_type,
        norm_type,
        input_size,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
    ):
        super(SkiMRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"], rnn_type
        assert norm_type in ["cLN", "cgLN", "gLN"], norm_type

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_direction = int(bidirectional) + 1
        self.linear_size = self.num_direction * self.hidden_size
        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.linear_size, self.input_size)
        self.norm = norms.get(norm_type)(self.input_size)

    def forward(self, w, hc):
        B, T, H = w.shape
        assert H == self.input_size, H

        if hc is None:
            h = torch.zeros(self.num_direction, B, self.hidden_size).to(w.device)
            c = torch.zeros(self.num_direction, B, self.hidden_size).to(w.device)
        else:
            h, c = hc

        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.

        z, (h, c) = self.rnn(w, (h, c))
        z = self.linear(self.drop(z))
        z = z.transpose(1, 2).contiguous()  # [B, H, T]
        z = self.norm(z).transpose(1, 2).contiguous()  # [B, T, H]
        w = w + z
        return w, (h, c)


class SegRNN(nn.Module):
    """the SegRNN of SkiM
    """

    def __init__(
        self,
        seg_input_size,
        seg_hidden_size,
        dropout=0.0,
        seg_bidirectional=False,
        seg_rnn_type="LSTM",
        seg_norm_type="cLN",
    ):
        super().__init__()

        self.seg_hidden_size = seg_hidden_size
        self.num_direction = int(seg_bidirectional) + 1

        self.rnn_block = SkiMRNN(
        rnn_type=seg_rnn_type,
        norm_type=seg_norm_type,
        input_size=seg_input_size,
        hidden_size=seg_hidden_size,
        dropout=dropout,
        bidirectional=seg_bidirectional,
        )

    def forward(self, w, hc):
        B, T, H = w.size()
        w, hc = self.rnn_block(w, hc)
        return w, hc


class MemRNN(nn.Module):
    """MemRNN of SkiMs
    """

    def __init__(
        self,
        seg_hidden_size,
        mem_hidden_size,
        dropout=0.0,
        seg_bidirectional=False,
        mem_bidirectional=False,
        mem_rnn_type="LSTM",
        mem_norm_type="cLN",
    ):
        super().__init__()

        self.mem_input_size = (int(seg_bidirectional) + 1) * seg_hidden_size
        self.mem_hidden_size = mem_hidden_size
        self.seg_bidirectional = seg_bidirectional
        self.mem_bidirectional = mem_bidirectional

        self.h_rnn_block = SkiMRNN(
        rnn_type=mem_rnn_type,
        norm_type=mem_norm_type,
        input_size=self.mem_input_size,
        hidden_size=self.mem_hidden_size,
        dropout=dropout,
        bidirectional=mem_bidirectional,
        )
        self.c_rnn_block = SkiMRNN(
        rnn_type=mem_rnn_type,
        norm_type=mem_norm_type,
        input_size=self.mem_input_size,
        hidden_size=self.mem_hidden_size,
        dropout=dropout,
        bidirectional=mem_bidirectional,
        )

    def forward(self, hc, S):
        # hc = (h, c), tuple of hidden and cell states from SegLSTM
        # shape of h and c: (D, B*S, H)
        # S: number of segments
        h, c = hc
        D, BS, H = h.shape
        B = BS // S

        h = h.transpose(1, 0).contiguous().view(B, S, D*H)
        c = c.transpose(1, 0).contiguous().view(B, S, D*H)
        h, _ = self.h_rnn_block(h, None)
        c, _ = self.c_rnn_block(c, None)

        h = h.view(B*S, D, H).transpose(1, 0).contiguous()
        c = c.view(B*S, D, H).transpose(1, 0).contiguous()
        ret_val = (h, c)

        if not self.seg_bidirectional:
            causal_ret_val = []
            for x in ret_val:
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val


class SkiMBlock(nn.Module):
    """Skipping-Memory RNN Block
    Note:
        Forward of this block does not change the size of input and output
    """

    def __init__(
        self,
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
    ):
        super().__init__()

        self.seg_input_size = seg_input_size
        self.chunk_size = chunk_size
        self.hop_size = hop_size if hop_size is not None else chunk_size
        self.n_repeats = n_repeats

        self.seg_models = nn.ModuleList([])
        for i in range(n_repeats):
            self.seg_models.append(
                SegRNN(
                    seg_input_size,
                    seg_hidden_size,
                    dropout,
                    seg_bidirectional,
                    seg_rnn_type,
                    seg_norm_type,
                )
            )

        self.mem_models = nn.ModuleList([])
        for i in range(n_repeats-1):
            self.mem_models.append(
                MemRNN(
                    seg_hidden_size,
                    mem_hidden_size,
                    dropout,
                    seg_bidirectional,
                    mem_bidirectional,
                    mem_rnn_type,
                    mem_norm_type,
                )
            )

    def forward(self, x):
        B, H, T = x.size()
        assert H == self.seg_input_size

        # Construct chunks
        x = unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        S = x.shape[-1]  # number of chunks
        x = x.reshape(B, H, self.chunk_size, S)
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, S (number of chunks), chunk_size, H]

        # Main SkiM processing
        x = x.view(B*S, self.chunk_size, H).contiguous()
        hc = None
        for i in range(self.n_repeats-1):
            x, hc = self.seg_models[i](x, hc)
            hc = self.mem_models[i](hc, S)

        x, _ = self.seg_models[-1](x, hc)
        x = x.reshape(B, S, self.chunk_size, H)

        # Reconstruct from chunks
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, H, chunk_size, S (number of chunks)]
        x = fold(
            x.reshape(B, self.chunk_size*H, S),
            (T, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        return x[..., 0]

    def _padfeature(self, x):
        B, H, T = x.size()
        rest = self.chunk_size - T % self.chunk_size
        if rest > 0:
            x = F.pad(x, (0, rest))
        return x, rest



class SkiM(nn.Module):
    """Skipping-Memory Network
    Reference
        [1] "SkiM: Skipping Memory LSTM for Low-Latency Real-Time
        Continuous Speech Separation", Chenda Li, Lei Yang, Weiqin Wang,
        and Yanmin Qian. (https://arxiv.org/abs/2201.10800)
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
    ):
        super(SkiM, self).__init__()

        self.n_src = n_src
        self.enc_size = enc_size
        self.seg_input_size = seg_input_size
        self.seg_hidden_size = seg_hidden_size
        self.mem_hidden_size = mem_hidden_size
        self.dropout = dropout
        self.seg_bidirectional = seg_bidirectional
        self.mem_bidirectional = mem_bidirectional
        self.seg_rnn_type = seg_rnn_type
        self.seg_norm_type = seg_norm_type
        self.mem_rnn_type = mem_rnn_type
        self.mem_norm_type = mem_norm_type
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.mask_act = mask_act

        # Convolution before SkiM
        layer_norm = norms.get(seg_norm_type)(enc_size)
        bottleneck_conv = nn.Conv1d(enc_size, seg_input_size, 1)
        self.bottleneck = nn.Sequential(
            layer_norm,
            bottleneck_conv
        )

        # SkiM
        self.skim = SkiMBlock(
            seg_input_size,
            seg_hidden_size,
            mem_hidden_size,
            dropout,
            seg_bidirectional,
            mem_bidirectional,
            seg_rnn_type,
            seg_norm_type,
            mem_rnn_type,
            mem_norm_type,
            chunk_size,
            hop_size,
            n_repeats,
        )
        # Masking in 3D space
        self.conv_sep = nn.Sequential(
            nn.PReLU(),
            CausalConv1D(seg_input_size, seg_input_size*n_src, n_src),  # n_src
            nn.PReLU()
        )
        # Speaker-wise gated convolution and mask computation
        self.gatedconv_spkwise_sigmoid = nn.Sequential(
            nn.Conv1d(seg_input_size, seg_input_size, 1),
            nn.Tanh()
        )
        self.gatedconv_spkwise_tanh = nn.Sequential(
            nn.Conv1d(seg_input_size, seg_input_size, 1),
            nn.Sigmoid()
        )

        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, "dim"):
            # For softmax, feed the source dimension.
            output_act = mask_nl_class(dim=1)
        else:
            output_act = mask_nl_class()

        self.mask_net = nn.Sequential(
            nn.Conv1d(seg_input_size, enc_size, 1),
            output_act,
        )

    def forward(self, mixture_w):
        """Forward
        """
        B, D, T = mixture_w.size()
        assert D == self.enc_size

        w = self.bottleneck(mixture_w)
        w = self.skim(w)
        w = self.conv_sep(w)  # [B, H*n_src, T]
        w = w.reshape(B*self.n_src, self.seg_input_size, -1)
        w = self.gatedconv_spkwise_sigmoid(w) * self.gatedconv_spkwise_tanh(w)

        # Compute mask
        est_mask = self.mask_net(w)
        est_mask = est_mask.view(B, self.n_src, self.enc_size, T)
        return est_mask

    def get_config(self):
        config = {
            "n_src": self.n_src,
            "enc_size": self.enc_size,
            "seg_input_size": self.seg_input_size,
            "seg_hidden_size": self.seg_hidden_size,
            "mem_hidden_size": self.mem_hidden_size,
            "dropout": self.dropout,
            "seg_bidirectional": self.seg_bidirectional,
            "mem_bidirectional": self.mem_bidirectional,
            "seg_rnn_type": self.seg_rnn_type,
            "seg_norm_type": self.seg_norm_type,
            "mem_rnn_type": self.mem_rnn_type,
            "mem_norm_type": self.mem_norm_type,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "mask_act": self.mask_act,
        }
        return config