# Copyright (c) 2020 郝翔  Xiang Hao
# Released under the MIT license
# https://github.com/Audio-WestlakeU/FullSubNet/blob/main/LICENSE

import torch
import torch.nn as nn
from torch.nn import functional


class FullSubMaskNet(nn.Module):
    def __init__(
        self,
        num_freqs,
        look_ahead,
        sequence_model,
        fb_num_neighbors,
        sb_num_neighbors,
        fb_output_activate_function,
        sb_output_activate_function,
        fb_model_hidden_size,
        sb_model_hidden_size,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
        weight_init=True,
    ):
        """FullSubNet model (cIRM mask).

        Args:
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        # super().__init__()  # audio_zen.model.base_model.BaseModelを継承し__init__関数を呼び出す．
        super(FullSubMaskNet, self).__init__()  # TODO: 上記に対応しているか確認
        
        assert sequence_model in (
            "GRU",
            "LSTM",
        ), f"{self.__class__.__name__} only support GRU and LSTM."
        
        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function,
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert (
            num_channels == 1
        ), f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model
        fb_input = self.norm(noisy_mag).reshape(
            batch_size, num_channels * num_freqs, num_frames
        )
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold fullband model's output, [B, N=F, C, F_f, T]. N is the number of sub-band units
        fb_output_unfolded = self.freq_unfold(fb_output, num_neighbors=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(
            batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames
        )

        # Unfold noisy spectrogram, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.freq_unfold(noisy_mag, num_neighbors=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(
            batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames
        )

        # Concatenation, [B, F, (F_s + F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation.
        if batch_size > 1:
            sb_input = drop_band(
                sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band
            )  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames,
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = (
            sb_mask.reshape(batch_size, num_freqs, 2, num_frames)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        output = sb_mask[:, :, :, self.look_ahead :]
        return output

    # FullSubMaskNetクラス内でself.〇〇で呼ばれている関数 (audio_zen.model.base_model.BaseModel):
    # norm_wrapper, weight_init, freq_unfold, apply (例外: nn.Module)
    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        elif norm_type == "forgetting_norm":
            norm = self.forgetting_norm
        else:
            raise NotImplementedError(
                "You must set up a type of Norm. "
                "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc."
            )
        return norm

    # norm_wrapper内でself.〇〇で呼ばれている関数 (audio_zen.model.base_model.BaseModel): 
    # offline_laplace_norm, cumulative_laplace_norm, offline_gaussian_norm, cumulative_layer_norm, forgetting_norm
    @staticmethod
    def offline_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # utterance-level mu
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = input / (mu + 1e-5)

        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)

        normed = (input - mu) / (std + 1e-5)

        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (
            cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum
        ) / entry_count + cumulative_mean.pow(
            2
        )  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def forgetting_norm(input, sample_length=192):
        """
        Using the mean value of the near frames to normalization

        Args:
            input: feature
            sample_length: length of the training sample, used for calculating smooth factor

        Returns:
            normed feature

        Shapes:
            input: [B, C, F, T]
            sample_length_in_training: 192
        """
        assert input.ndim == 4
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

        eps = 1e-10
        mu = 0
        alpha = (sample_length - 1) / (sample_length + 1)

        mu_list = []
        for frame_idx in range(num_frames):
            if frame_idx < sample_length:
                alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(
                    input[:, :, frame_idx], dim=1
                ).reshape(
                    batch_size, 1
                )  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(
                    batch_size, 1
                )  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        output = input / (mu + eps)

        output = output.reshape(batch_size, num_channels, num_freqs, num_frames)
        return output

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
    
    @staticmethod
    def freq_unfold(input, num_neighbors):
        """Split the overlapped subband units along the frequency axis.

        Args:
            input: four-dimension input with the shape [B, C, F, T]
            num_neighbors: number of neighbors in each side for each subband unit.

        Returns:
            Overlapped sub-band units specified as [B, N, C, F_s, T], where `F_s` represents the frequency axis of
            each sub-band unit and `N` is the number of sub-band unit, e.g., [2, 161, 1, 19, 100].
        """
        assert input.dim() == 4, f"The dim of the input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbors <= 0:  # No change to the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)  # [B * C, 1, F, T]
        sub_band_unit_size = num_neighbors * 2 + 1

        # Pad the top and bottom of the original spectrogram
        output = functional.pad(output, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B * C, 1, F, T]

        # Unfold the spectrogram into sub-band units
        # [B * C, 1, F, T] => [B * C, sub_band_unit_size, num_frames, N], N is equal to the number of frequencies.
        output = functional.unfold(output, kernel_size=(sub_band_unit_size, num_frames))  # move on the F and T axes
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dimension of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, C, F_s, T]

        return output

# from audio_zen.model.module.sequence_model import SequenceModel
class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        bidirectional,
        sequence_model="GRU",
        output_activate_function="Tanh",
    ):
        """
        Wrapper of conventional sequence models (LSTM or GRU)

        Args:
            input_size: input size.
            output_size: when projection_size> 0, the linear layer is used for projection. Otherwise, no linear layer.
            hidden_size: hidden size.
            num_layers:  number of layers.
            bidirectional: whether to use bidirectional RNN.
            sequence_model: LSTM | GRU.
            output_activate_function: Tanh | ReLU | ReLU6 | LeakyReLU | PReLU | None.
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "SRU":
            pass
            # self.sequence_model = CustomSRU(
            #     input_size=input_size,
            #     hidden_size=hidden_size,
            #     num_layers=num_layers,
            #     bidirectional=bidirectional,
            #     highway_bias=-2
            # )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if int(output_size):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )

        self.output_activate_function = output_activate_function
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3, f"The shape of input is {x.shape}."
        self.sequence_model.flatten_parameters()

        x = x.permute(0, 2, 1)  # [B, F, T] => [B, T, F]
        o, _ = self.sequence_model(x)

        if self.output_size:
            o = self.fc_output_layer(o)

        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1)  # [B, T, F] => [B, F, T]
        return o

# from audio_zen.acoustics.feature import drop_band
def drop_band(input, num_groups=2):
    """Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert (
        batch_size > num_groups
    ), f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., : (num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(
            group_idx, batch_size, num_groups, device=input.device
        )
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(
            selected_samples, dim=2, index=freqs_indices
        )  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


if __name__ == "__main__":
    with torch.no_grad():
        noisy_mag = torch.rand(1, 1, 257, 63)
        model = FullSubMaskNet(
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fb_num_neighbors=0,
            sb_num_neighbors=15,
            fb_output_activate_function="ReLU",
            sb_output_activate_function=False,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
            weight_init=False,
        )
        print(model(noisy_mag).shape)