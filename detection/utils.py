import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

def crop_or_pad(waveform, sample_rate=44100, desired_length=2):
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


def train_one_epoch(loader, model, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    losses = []
    
    with tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', total=len(loader)) as tepoch:
        for inputs, labels in tepoch:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Compute prediction and loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad(True)
            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=np.mean(losses))
    
    return np.mean(losses)
            

@torch.no_grad()
def evaluate(loader, model, criterion, device):
    model.eval()
    losses = []
    correct = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        correct += ((outputs > 0.5).to(int) == labels).sum().item()
    
    avg_loss = np.mean(losses)
    accuracy = (correct / len(loader.dataset)) * 100
    
    return avg_loss, accuracy










# class PaperDataset(Dataset):
#     def __init__(
#         self,
#         root_dir: str,
#         sample_rate: int = 22500,
#         clip_length: int = 2,
#         spec_dims: tuple[int] = (256, 256),
#         power: int = 2,
#         transforms = []
#     ) -> None:
        
#         filepaths = []
#         self.labels = []
#         self.sample_rate = sample_rate
#         self.clip_length = clip_length
#         self.transforms = transforms
#         self.waveforms = []
        
#         # Process samples
#         for sub_dir in os.listdir(root_dir):
#             if os.path.isdir(os.path.join(root_dir, sub_dir)):
#                 for fname in os.listdir(os.path.join(root_dir, sub_dir)):
#                     if fname.endswith('.wav'):
#                         waveform, orig_freq = torchaudio.load(os.path.join(root_dir, sub_dir, fname))
#                         waveform = F.resample(waveform, orig_freq=orig_freq, new_freq=self.sample_rate)
#                         waveform = crop_around_peak(waveform, self.sample_rate, self.clip_length)
#                         self.waveforms.append(waveform)
#                         self.labels.append(1 if sub_dir == 'positive' else 0)
        
#         n_samples = self.sample_rate * self.clip_length
#         n_ffts = (spec_dims[1] * 2) - 1
#         hop_length = max(1, int((n_samples - n_ffts) / (spec_dims[0] - 1)) + 2)
        
#         self.to_specgram = T.Spectrogram(
#             n_fft=n_ffts,
#             hop_length=hop_length,
#             power=power
#         )
        
#         self.power_to_dB = T.AmplitudeToDB(stype='power')
        
#     def __len__(self) -> int:
#         return len(self.waveforms)
        
#     def __getitem__(self, index) -> tuple[torch.Tensor, int]:
#         waveform = self.waveforms[index]
#         label = self.labels[index]
#         for transform in self.transforms:
#             waveform = transform(waveform)
#         spec = self.to_specgram(waveform)
#         spec_dB = self.power_to_dB(spec)
        
#         return spec_dB, label
    
    
# class NCBIDataset(Dataset):
#     def __init__(
#         self,
#         root_dir: str,
#         sample_rate: int = 44100,
#         clip_length: int = 2,
#         spec_dims: tuple[int] = (256, 256),
#         power: int = 2,
#         transforms = []
#     ) -> None:
#         self.sample_rate = sample_rate
#         self.clip_length = clip_length
#         self.transforms = transforms
        
#         self.waveforms = []
#         for sub_dir in os.listdir(root_dir):
#             if os.path.isdir(os.path.join(root_dir, sub_dir)):
#                 for fname in os.listdir(os.path.join(root_dir, sub_dir)):
#                     if fname.endswith('.wav'):
#                         waveform, orig_freq = torchaudio.load(os.path.join(root_dir, sub_dir, fname))
#                         waveform = F.resample(waveform, orig_freq=orig_freq, new_freq=self.sample_rate)
#                         waveform = crop_around_peak(waveform, self.sample_rate, self.clip_length)
#                         self.waveforms.append(waveform)
            
#         n_samples = self.sample_rate * self.clip_length
#         n_ffts = (spec_dims[1] * 2) - 1
#         hop_length = max(1, int((n_samples - n_ffts) / (spec_dims[0] - 1)) + 2)

#         self.to_specgram = T.Spectrogram(n_fft=n_ffts,
#                                          hop_length=hop_length,
#                                          power=power)
        
#         self.power_to_dB = T.AmplitudeToDB(stype='power')

        
#     def __len__(self) -> int:
#         return len(self.waveforms)
        
#     def __getitem__(self, index) -> tuple[torch.Tensor, int]:
#         waveform = self.waveforms[index]
#         for transform in self.transforms:
#             waveform = transform(waveform)
#         spec = self.to_specgram(waveform)
#         spec_dB = self.power_to_dB(spec)
        
#         return spec_dB, 1  # class = 1 for gunshot
        