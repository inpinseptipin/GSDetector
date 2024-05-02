import contextlib
import timeit
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn import metrics
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils import benchmark
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def inference_loop(
    model: nn.Module,
    data: Tensor,
    device: torch.device,
) -> float:
    with contextlib.suppress(AttributeError):
        model, data = model.to(device), data.to(device)
    # print(f"Benchmarking on {str(device).upper()}...", end=" ", flush=True)
    with tqdm(
        desc=f"{str(device).upper()} Benchmarking",
        unit=" infers",
    ) as pbar:
        result = benchmark.Timer(
            stmt="model(data)",
            globals={"model": model, "data": data},
            # num_threads=4,
        ).blocked_autorange(min_run_time=5, callback=lambda n, t: pbar.update(n))
    pbar.close()
    # print("Done!")

    return result.median


@torch.no_grad()
def get_inference_rates(
    model: nn.Module,
    input_shape: Tuple[int] = (1, 1, 88200),
    gpu: torch.device = None,
) -> Tuple[float, float | str]:
    gpu = gpu if gpu != "cpu" else None
    with contextlib.suppress(AttributeError):
        model.eval()
    data = torch.rand(input_shape, requires_grad=False)
    cpu_rate = inference_loop(model, data, torch.device("cpu"))
    gpu_rate = inference_loop(model, data, gpu) if gpu else "N/A"

    return cpu_rate, gpu_rate


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    y_prob = []
    y_true = []
    total_loss = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        y_prob.extend(outputs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    # Convert lists to NumPy arrays for metrics computations
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = y_prob.round()

    # Calculate Metrics
    avg_loss = total_loss / len(loader.dataset)
    accuracy = metrics.accuracy_score(y_true, y_prob.round())
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    roc_auc = metrics.roc_auc_score(y_true, y_prob)

    # Print Results
    spacing = 25
    print("Testing Results:\n")
    print(f"{'Test Accuracy':<{spacing-int(accuracy>=0.1)}} {accuracy*100:.6f} %")
    print(f"{'Mean Test Loss':<{spacing}} {avg_loss:.6f}")
    print(f"{'Precision':<{spacing}} {precision:.6f}")
    print(f"{'Recall':<{spacing}} {recall:.6f}")
    print(f"{'F1 Score':<{spacing}} {f1:.6f}")
    print(f"{'False-Positive Rate':<{spacing-int(fpr>=0.1)}} {fpr*100:.6f} %")
    print(f"{'False-Negative Rate':<{spacing-int(fnr>=0.1)}} {fnr*100:.6f} %")
    print(f"{'ROC AUC':<{spacing}} {roc_auc:.6f}\n")

    # Benchmarking
    cpu_rate, gpu_rate = get_inference_rates(model, gpu=device)
    print(f"\n{'CPU Inference Rate':<{spacing}} {cpu_rate*1e6:.4f} µs/inference")
    if gpu_rate != "N/A":
        print(f"{'GPU Inference Rate':<{spacing}} {gpu_rate*1e6:.4f} µs/inference")

    results = {
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "cpu_rate": cpu_rate,
        "gpu_rate": gpu_rate,
    }

    return results, y_true, y_prob
