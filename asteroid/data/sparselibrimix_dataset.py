import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import glob
import shutil
import zipfile

from .wham_dataset import wham_noise_license


class SparseLibriMix(Dataset):
    """
    Args:
        base_dir (str) : The path of out_dir in create_sparse.sh
        task (str) : ``'sep_clean'`` or ``'sep_noisy'``
        sample_rate (int) : 8000 or 16000
        n_src (int) : 2 or 3

    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
        [2] https://github.com/popcornell/SparseLibriMix
    """
    dataset_name = "SparseLibriMix"

    def __init__(
        self, base_dir, task="sep_clean", sample_rate=16000, n_src=2, segment=None
    ):
        # Get list of audio paths
        mode = task.replace("sep", "mix")
        self.mix_dir = os.path.join(base_dir, "wav"+str(sample_rate), mode)

        self.source_dirs = []
        for k in range(n_src):
            self.source_dirs.append(os.path.join(base_dir, "wav"+str(sample_rate), f"s{k+1}"))

        utts = glob.glob(os.path.join(self.mix_dir, "*.wav"))
        print(self.mix_dir, len(utts))
        self.utt_names = sorted([os.path.basename(p) for p in utts])
        assert len(self.utt_names) == 500, f"The number of utterances should be 500"

        assert segment is None, "The segment size should be None for evaluation"
        self.segment = None 
        self.seg_len = None
        self.n_src = n_src

    def __len__(self):
        return len(self.utt_names)

    def __getitem__(self, idx, start=0, stop=None):
        # Get mixture path
        utt_name = self.utt_names[idx]
        mixture_path = os.path.join(self.mix_dir, utt_name)
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)

        # If task is enh_both then the source is the clean mixture
        sources_list = []
        for source_dir in self.source_dirs:
            source_path = mixture_path = os.path.join(source_dir, utt_name)
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)

        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)

        # Convert sources to tensor
        sources = np.vstack(sources_list)
        sources = torch.from_numpy(sources)

        return mixture, sources

    @classmethod
    def loaders_from_mini(cls, batch_size=4, **kwargs):
        pass

    @classmethod
    def mini_from_download(cls, **kwargs):
        pass

    @staticmethod
    def mini_download():
        pass

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self._dataset_name()
        infos["task"] = self.task
        if self.task == "sep_clean":
            data_license = [librispeech_license]
        else:
            data_license = [librispeech_license, wham_noise_license]
        infos["licenses"] = data_license
        return infos

    def _dataset_name(self):
        """Differentiate between 2 and 3 sources."""
        return f"SparseLibri{self.n_src}Mix"


librispeech_license = dict(
    title="LibriSpeech ASR corpus",
    title_link="http://www.openslr.org/12",
    author="Vassil Panayotov",
    author_link="https://github.com/vdp",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
