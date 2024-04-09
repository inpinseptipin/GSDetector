import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch import nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from utils import crop_or_pad


class GunshotDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        label: int,
        sample_rate: int = 44100,
        clip_length: int = 2,
        spec_dims: tuple[int] = (256, 256),
        power: int = 2,
        transforms = []
    ) -> None:
        self.label = torch.FloatTensor([label])
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.spec_dims = spec_dims
        self.transforms = transforms
        self.waveforms = []
        self.filepaths = []
        
        for sub_dir in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, sub_dir)):
                for fname in os.listdir(os.path.join(root_dir, sub_dir)):
                    if fname.endswith('.wav'):
                        filepath = os.path.join(root_dir, sub_dir, fname)
                        waveform, orig_freq = torchaudio.load(filepath)
                        waveform = F.resample(
                            waveform,
                            orig_freq=orig_freq,
                            new_freq=self.sample_rate,
                            lowpass_filter_width=18,
                        )
                        waveform = crop_or_pad(waveform, self.sample_rate, self.clip_length)
                        self.waveforms.append(waveform)
                        self.filepaths.append(filepath)
            
        n_samples = self.sample_rate * self.clip_length
        n_ffts = (spec_dims[0] * 2) - 1
        hop_length = max(1, int((n_samples - n_ffts) / (spec_dims[1] - 1)) + 2)

        self.to_specgram = T.Spectrogram(n_fft=n_ffts,
                                         hop_length=hop_length,
                                         power=power)
        
        self.power_to_dB = T.AmplitudeToDB(stype='power')

    def __len__(self) -> int:
        return len(self.waveforms)
        
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        waveform = self.waveforms[index]
        filepath = self.filepaths[index]
        for transform in self.transforms:
            waveform = transform(waveform)
        spec = self.to_specgram(waveform)
        spec_dB = self.power_to_dB(spec)
        # if spec_dB.shape[1:] != self.spec_dims:
        #     spec_dB = spec_dB[:self.spec_dims[0], :self.spec_dims[1]]
        assert spec_dB.shape[1:] == self.spec_dims, f"Spectrogram shape {spec_dB.shape[1:]} ≠ desired shape {self.spec_dims}"
        
        return spec_dB, self.label  # label: 0 for negative, 1 for gunshot
    
def test():
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
    

# test()