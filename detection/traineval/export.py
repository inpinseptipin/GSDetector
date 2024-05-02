from typing import Tuple

import torch


@torch.no_grad()
def export_traced(
    model: torch.nn.Module,
    picklepath: str,
    input_shape: Tuple[int] = (1, 1, 256, 256),
    device: torch.device = None,
) -> None:
    device = device or torch.device("cpu")
    data = torch.rand(input_shape, device=device)
    model.load_state_dict(torch.load(picklepath))
    model = model.to(device)
    model.eval()
    savepath = picklepath.replace("pickle", "torchscript")
    traced_module = torch.jit.trace(model, data)
    traced_module.save(savepath)


@torch.no_grad()
def export_onnx(
    model: torch.nn.Module,
    picklepath: str,
    input_shape: Tuple[int] = (1, 1, 256, 256),
    device: torch.device = None,
    input_names: list[str] = None,
    output_names: list[str] = None,
) -> None:
    device = device or torch.device("cpu")
    data = torch.rand(input_shape, device=device)
    input_names = input_names or ["Spectrogram"]
    output_names = output_names or ["Probability of Gunshot"]
    savepath = picklepath.replace("pickle", "onnx").replace(".pt", ".onnx")
    model.load_state_dict(torch.load(picklepath))
    model = model.to(device)
    model.eval()
    torch.onnx.export(
        model=model,
        args=data,
        f=savepath,
        input_names=input_names,
        output_names=output_names,
    )

    return savepath


@torch.no_grad()
def export_AOTcompiled(
    model: torch.nn.Module,
    picklepath: str,
    input_shape: Tuple[int] = (1, 1, 256, 256),
) -> None:  # sourcery skip: inline-immediately-returned-variable
    example_inputs = (torch.rand(input_shape, device="cpu"),)
    model.load_state_dict(torch.load(picklepath))
    model = model.to("cpu")
    model.eval()

    savepath = (
        picklepath.replace("pickle", "AOTcompiled").split(".")[:-1][0] + "/model.so"
    )
    so_path = torch._export.aot_compile(
        model, example_inputs, options={"aot_inductor.output_path": savepath}
    )

    return so_path
