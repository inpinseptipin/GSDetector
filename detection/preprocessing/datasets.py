import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms.v2 as TVT
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchaudio.functional import resample

from .utils import crop_or_pad


class GunshotDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        label: int,
        sample_rate: int = 44100,
        clip_length: int = 2,
        spec_dims: tuple[int] = (256, 256),
        power: int = 2,
        wav_transforms=None,
        spec_transforms=None,
    ) -> None:
        if wav_transforms is None:
            wav_transforms = []
        if spec_transforms is None:
            spec_transforms = []
        self.label = torch.FloatTensor([label])
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.spec_dims = spec_dims
        self.wav_transforms = wav_transforms
        self.spec_transforms = spec_transforms
        self.spectrograms = []
        self.filepaths = []

        n_samples = self.sample_rate * self.clip_length
        n_ffts = (spec_dims[1] * 2) - 1
        hop_length = max(1, int((n_samples - n_ffts) / (spec_dims[0] - 1)) + 2)

        self.to_specgram = T.Spectrogram(
            n_fft=n_ffts, hop_length=hop_length, power=power
        )

        self.power_to_dB = T.AmplitudeToDB(stype="power")

        for sub_dir in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, sub_dir)):
                for fname in os.listdir(os.path.join(root_dir, sub_dir)):
                    if fname.endswith(".wav"):
                        filepath = os.path.join(root_dir, sub_dir, fname)
                        self.filepaths.append(filepath)
                        waveform, orig_freq = torchaudio.load(filepath)
                        waveform = resample(
                            waveform, orig_freq=orig_freq, new_freq=self.sample_rate
                        )
                        waveform = crop_or_pad(
                            waveform, self.sample_rate, self.clip_length
                        )

                        for transform in self.wav_transforms:
                            waveform = transform(waveform)
                        spec = self.power_to_dB(self.to_specgram(waveform))
                        for transform in self.spec_transforms:
                            spec = transform(spec)
                        assert (
                            spec.shape[1:] == self.spec_dims
                        ), f"Expected spec shape {self.spec_dims}, got {spec.shape[1:]}"
                        self.spectrograms.append(spec)

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return (
            self.spectrograms[index],
            self.label,
        )  # label: 0 for negative, 1 for gunshot


class LightweightDataset(Dataset):
    """
    A memory-efficient version of the `GunshotDataset` class for handling homogenous audio samples from a single source.
        - Instead of loading all the spectrograms into memory upon initialization, this class loads them on-the-fly when accessed.
        - This significantly decreases the memory usage but requires that the spectrograms have already been computed and saved to `root_dir` as `.pt` files.

    Args:
        root_dir (str): The root directory of the dataset containing spectrogram tensors (`.pt` files).
        transforms (list, optional): List of data transformations to apply. Defaults to an empty list.

    Attributes:
        filepaths (list): List of filepaths for the dataset files.
        label (torch.Tensor): The label for the dataset.
        transforms (list): List of data transformations to apply.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the tuple (spec, label) at the specified index.

    """

    def __init__(
        self, root_dir: os.PathLike, transforms: List[nn.Module] = None
    ) -> None:
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.filepaths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(".pt")
        ]
        self.label = torch.tensor(
            int(root_dir.split("_")[-1]), dtype=torch.float32
        ).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        spec = torch.load(self.filepaths[index])
        for transform in self.transforms:
            spec = transform(spec)

        return spec, self.label


class WaveformDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 44100,
        clip_length: int = 2,
        transforms=None,
    ) -> None:
        if transforms is None:
            transforms = []
        self.label = torch.tensor(
            int(root_dir.split("_")[-1]), dtype=torch.float32
        ).unsqueeze(0)
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.transforms = transforms
        self.filepaths = []

        for sub_dir in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, sub_dir)):
                for fname in os.listdir(os.path.join(root_dir, sub_dir)):
                    if fname.endswith(".wav"):
                        filepath = os.path.join(root_dir, sub_dir, fname)
                        waveform, orig_freq = torchaudio.load(filepath)
                        if orig_freq >= self.sample_rate:
                            self.filepaths.append(filepath)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        waveform, orig_freq = torchaudio.load(self.filepaths[index])
        waveform = resample(waveform, orig_freq=orig_freq, new_freq=self.sample_rate)
        waveform = crop_or_pad(waveform, self.sample_rate, self.clip_length)
        for transform in self.transforms:
            waveform = transform(waveform)

        return waveform, self.label  # label: 0 for negative, 1 for gunshot


class NormalizedDataset(Dataset):
    def __init__(self, subset: Subset, normalize: nn.Module = None) -> None:
        self.subset = subset
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        spec, label = self.subset[index]
        if self.normalize:
            spec = self.normalize(spec)
        return spec, label


def _test():
    cadre = GunshotDataset(root_dir="data/positive/cadre", label=1, sample_rate=44100)
    ncbi = GunshotDataset(root_dir="data/positive/ncbi", label=1, sample_rate=44100)
    paper = GunshotDataset(root_dir="data/positive/paper", label=1, sample_rate=22500)
    positives = ConcatDataset([cadre, ncbi, paper])
    negatives = GunshotDataset(root_dir="data/negative/", label=0, sample_rate=22500)

    print("Positives:", len(positives))
    for i in range(0, len(positives), 1000):
        spec, label = positives[i]
        plt.imshow(spec[0])
        print(label)
        plt.show()

    print("Negatives:", len(negatives))
    for i in range(0, len(negatives), 1000):
        spec, label = negatives[i]
        plt.imshow(spec[0])
        print(label)
        plt.show()


# _test()
