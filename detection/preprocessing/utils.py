import torch
import torch.nn.functional as F
from torch import Tensor


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
        new_waveform = F.pad(waveform, (pad_left, pad_right), "reflect")

    assert new_waveform.shape[1] == desired_samples

    return new_waveform


# def ensure_mono(waveform):
#     if waveform.size(0) > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)
#     return waveform


# def adjust_bounds(start_sample, end_sample, waveform, desired_length, sample_rate):
#     if start_sample < 0:
#         start_sample = 0
#         end_sample = desired_length * sample_rate
#     if end_sample > waveform.size(1):
#         end_sample = waveform.size(1)
#         start_sample = waveform.size(1) - desired_length * sample_rate
#     return start_sample, end_sample


# def crop_waveform(waveform, desired_samples, sample_rate):
#     peak_index = torch.argmax(torch.abs(waveform))
#     half_length_samples = desired_samples // 2
#     start_sample = peak_index - half_length_samples
#     end_sample = peak_index + half_length_samples
#     start_sample, end_sample = adjust_bounds(
#         start_sample, end_sample, waveform, desired_samples, sample_rate
#     )
#     return waveform[:, start_sample:end_sample]


# def pad_waveform(waveform, desired_samples, sample_rate, desired_length):
#     pad_length = desired_length * sample_rate - waveform.size(1)
#     pad_left = pad_length // 2
#     pad_right = pad_length - pad_left
#     return F.pad(waveform, (pad_left, pad_right), "reflect")


# def crop_or_pad(waveform, sample_rate=44100, desired_length=2) -> Tensor:
#     """
#     Crop or pad the waveform to the desired length.

#     Args:
#         `waveform` (torch.Tensor): Input waveform.
#         `sample_rate` (int): Sample rate of the waveform (default: 44100).
#         `desired_length` (int): Desired length of the waveform in seconds (default: 2).

#     Returns:
#         torch.Tensor: Cropped or padded waveform.
#     """
#     waveform = ensure_mono(waveform)
#     desired_samples = sample_rate * desired_length

#     if waveform.shape[1] > desired_samples:
#         return crop_waveform(waveform, desired_samples, sample_rate)
#     else:
#         return pad_waveform(waveform, desired_samples, sample_rate)
