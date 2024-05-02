import csv
import os
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.transforms.v2 import Normalize

from .datasets import LightweightDataset, NormalizedDataset, WaveformDataset


def calculate_stats(dataset: Dataset) -> Tuple[Tensor, Tensor]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    nimages = 0
    mean = 0.0
    std = 0.0
    for batch, _ in loader:
        # Rearrange to [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        nimages += batch.size(0)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    mean /= nimages
    std /= nimages

    return mean, std


def get_loaders(
    train_pct: float = 0.7,
    batch_size: int = 256,
    normalize: bool = True,
    transforms: List[nn.Module] = None,
) -> Tuple[Dict[str, DataLoader], Tensor]:
    # sourcery skip: use-fstring-for-concatenation

    data_dir = os.path.join(os.getcwd(), "data/waveforms/")

    # Dataset Initialization
    cadre_1 = WaveformDataset(data_dir + "cadre_1")
    ncbi_1 = WaveformDataset(data_dir + "ncbi_1")
    kaggle_1 = WaveformDataset(data_dir + "kaggle_1")
    urban_1 = WaveformDataset(data_dir + "urban_1")
    esc50_0 = WaveformDataset(data_dir + "esc50_0")
    campus_0 = WaveformDataset(data_dir + "campus_0")
    urban_0 = WaveformDataset(data_dir + "urban_0")

    positives = ConcatDataset([cadre_1, ncbi_1, kaggle_1, urban_1])
    negatives = ConcatDataset([esc50_0, campus_0, urban_0])
    pos_weight = torch.tensor(len(negatives) / len(positives))

    full_dataset = ConcatDataset([positives, negatives])
    total_indices = torch.randperm(len(full_dataset))
    shuffled_dataset = Subset(full_dataset, total_indices)

    # Train/Val/Test Split
    total_size = len(shuffled_dataset)
    train_size = int(train_pct * total_size)
    remaining_size = total_size - train_size
    test_size = remaining_size // 2
    valid_size = remaining_size - test_size
    train_dataset, valid_dataset, test_dataset = random_split(
        shuffled_dataset, [train_size, valid_size, test_size]
    )
    spacing = 20
    print(f"{'Training Samples':<{spacing}} {len(train_dataset)}")
    print(f"{'Validation Samples':<{spacing}} {len(valid_dataset)}")
    print(f"{'Testing Samples':<{spacing}} {len(test_dataset)}")

    # Normalization
    if normalize:
        mean, std = calculate_stats(train_dataset)
        print(
            f"{'Before Normalization:':<{spacing}} mean = {mean.item():.2f} | std = {std.item():.2f}"
        )
        norm = Normalize(mean, std)
        train_dataset = NormalizedDataset(train_dataset, norm)
        valid_dataset = NormalizedDataset(valid_dataset, norm)
        test_dataset = NormalizedDataset(test_dataset, norm)

        # Write mean and std values to CSV file
        with open(data_dir + "norm_stats.csv", "a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Seed", "Mean", "Std"])
            writer.writerow([torch.initial_seed(), mean.item(), std.item()])

        # Calculate mean and std after normalization
        mean, std = calculate_stats(train_dataset)
        print(
            f"{'After Normalization:':<{spacing}} mean =  {mean.item():.3f} | std = {std.item():.3f}"
        )

    # Build dictionary of DataLoaders
    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        ),
        "valid": DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        ),
        "test": DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        ),
    }

    return loaders, pos_weight
