import timeit
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F


@torch.no_grad()
def get_inference_rate(model, data=torch.rand(1, 1, 256, 256), iters=1000) -> float:
    """
    Calculate the inference rate (frames per second) of the model.

    Args:
        model (torch.nn.Module): Model to be evaluated.
        data (torch.Tensor): Input data for inference (default: random tensor).
        iters (int): Number of iterations for timing (default: 1000).

    Returns:
        float: Inference rate in frames per second.
    """
    model.eval()
    data.requires_grad_(False)
    time_per = 0
    time_per = timeit.timeit(lambda: model(data), number=iters) / iters
    fps = 1/time_per
    
    return fps


def compile_and_save(
    model: torch.nn.Module,
    picklepath: str,
    data: torch.Tensor=torch.rand(1, 1, 256, 256),
    save_onnx: bool=True,
    input_names: list[str]=["Spectrogram"],
    output_names: list[str]=["Probability of Gunshot"]
):
    """
    Compile and save the model.

    Args:
        model (torch.nn.Module): Model to be compiled and saved.
        picklepath (str): Path to the model's pickle file.
        data (torch.Tensor): Input data for tracing (default: random tensor).
        save_onnx (bool): Whether to save the model in ONNX format (default: True).
        input_names (list[str]): List of input names for ONNX model (default: ["Spectrogram"]).
        output_names (list[str]): List of output names for ONNX model (default: ["Probability of Gunshot"]).
    """
    model.load_state_dict(torch.load(picklepath))
    model.eval()
    jitpath = picklepath.replace('pickle', 'compiled') + 'c'
    traced_module = torch.jit.trace(model, data)
    traced_module.save(jitpath)
    
    if save_onnx:
        torch.onnx.export(
            model=model,
            args=data,
            f=jitpath.replace('compiled', 'onnx').replace('.ptc', '.onnx'),
            input_names=input_names,
            output_names=output_names
        )