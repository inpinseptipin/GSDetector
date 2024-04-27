import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt


def crop_or_pad(waveform, sample_rate=44100, desired_length=2) -> torch.Tensor:
    """
    Crop or pad the waveform to the desired length.

    Args:
        `waveform (torch.Tensor): Input waveform.
        sample_rate (int): Sample rate of the waveform (default: 44100).
        desired_length (int): Desired length of the waveform in seconds (default: 2).

    Returns:
        torch.Tensor: Cropped or padded waveform.
    """
    # Ensure mono
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    desired_samples = sample_rate * desired_length
    
    if waveform.shape[1] > desired_samples:
        # Find the peak amplitude
        peak_index = torch.argmax(torch.abs(waveform))
        
        # Calculate start and end samples
        half_length_samples = desired_samples // 2
        start_sample = peak_index - half_length_samples
        end_sample = peak_index + half_length_samples

        # Adjust if outside bounds
        if start_sample < 0:
            start_sample = 0
            end_sample = desired_length * sample_rate
        if end_sample > waveform.size(1):
            end_sample = waveform.size(1)
            start_sample = waveform.size(1) - desired_length * sample_rate

        # Crop the waveform
        new_waveform = waveform[:, start_sample:end_sample]
    
    else:
        # Calculate padding
        pad_length = desired_length * sample_rate - waveform.size(1)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left

        # Apply padding
        new_waveform = F.pad(waveform, (pad_left, pad_right), 'reflect')
    
    assert new_waveform.shape[1] == desired_samples

    return new_waveform


class GunshotDataset(Dataset):
    """
    A custom dataset class for handling homogenous audio samples from a single source.
        - Loads `.wav` files from the specified root directory, computes the spectrograms, and applies the specified transformations.
        - *EXTREMELY* memory-intensive, not be suitable for large datasets.

    Args:
        root_dir (str): The root directory of the dataset.
        label (int): The label for the dataset (0 for negative, 1 for gunshot).
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.
        clip_length (int, optional): The desired length of each audio clip in seconds. Defaults to 2.
        spec_dims (tuple[int], optional): The dimensions of the spectrogram. Defaults to (256, 256).
        power (int, optional): The power parameter for the spectrogram computation. Defaults to 2.
        wav_transforms (list, optional): List of waveform transformations to apply. Defaults to an empty list.
        spec_transforms (list, optional): List of spectrogram transformations to apply. Defaults to an empty list.

    Attributes:
        label (torch.Tensor): The label tensor for the dataset.
        sample_rate (int): The sample rate of the audio.
        clip_length (int): The desired length of each audio clip in seconds.
        spec_dims (tuple[int]): The dimensions of the spectrogram.
        wav_transforms (list): List of waveform transformations to apply.
        spec_transforms (list): List of spectrogram transformations to apply.
        spectrograms (list): List of spectrograms.
        filepaths (list): List of filepaths.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the item at the specified index.

    """

    def __init__(
        self,
        root_dir: str,
        label: int,
        sample_rate: int = 44100,
        clip_length: int = 2,
        spec_dims: tuple[int] = (256, 256),
        power: int = 2,
        wav_transforms=[],
        spec_transforms=[]
    ) -> None:
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

        self.to_specgram = T.Spectrogram(n_fft=n_ffts,
                                         hop_length=hop_length,
                                         power=power)
        
        self.power_to_dB = T.AmplitudeToDB(stype='power')
        
        for sub_dir in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, sub_dir)):
                for fname in os.listdir(os.path.join(root_dir, sub_dir)):
                    if fname.endswith('.wav'):
                        filepath = os.path.join(root_dir, sub_dir, fname)
                        self.filepaths.append(filepath)
                        waveform, orig_freq = torchaudio.load(filepath)
                        waveform = F.resample(waveform, orig_freq=orig_freq, new_freq=self.sample_rate)
                        waveform = crop_or_pad(waveform, self.sample_rate, self.clip_length)
                        
                        for transform in self.wav_transforms:
                            waveform = transform(waveform)
                        spec = self.power_to_dB(self.to_specgram(waveform))
                        for transform in self.spec_transforms:
                            spec = transform(spec)
                        assert spec.shape[1:] == self.spec_dims, f"Expected spec shape {self.spec_dims}, got {spec.shape[1:]}"
                        self.spectrograms.append(spec)
                        
    def __len__(self) -> int:
        return len(self.spectrograms)
        
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spectrograms[index], self.label  # label: 0 for negative, 1 for gunshot
    
    
    
class LightweightGSDataset(Dataset):
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
        __getitem__(index): Returns the item at the specified index.

    """

    def __init__(self, root_dir: str, transforms=[]) -> None:
        self.filepaths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.pt')]
        self.label = torch.tensor(int(root_dir.split('_')[-1]), dtype=torch.float32).unsqueeze(0)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.load(self.filepaths[index])
        for transform in self.transforms:
            spec = transform(spec)

        return spec, self.label
    
    
    
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