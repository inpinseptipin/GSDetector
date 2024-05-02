import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import SpecAugment

from models.enhanced import EnhancedCNN
from models.resnet import build_resnet18, build_resnet34
from models.simple import SimpleCNN, SimplePipeline
from preprocessing.dataloaders import get_loaders
from traineval.evaluation import evaluate, get_inference_rates
from traineval.export import export_AOTcompiled, export_onnx, export_traced
from traineval.training import train


def main():
    torch.manual_seed(13)

    # Hyperparameters
    TRAIN_PCT = 0.7
    BATCH_SIZE = 256
    NUM_EPOCHS = 1
    NORMALIZE = False
    INIT_LR = 1e-4
    MIN_LR = 0.0
    EPS = 1e-8
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Model Initialization
    model = SimplePipeline()
    model.to(DEVICE)
    savepath = os.path.join(
        os.getcwd(), f"models/trained_models/pickle/{model.__name__}.pt"
    )

    print("\n-------------------- PREPROCESSING --------------------\n")

    # Data Augmentation
    transforms = None

    # Create DataLoaders
    loaders, pos_weight = get_loaders(
        train_pct=TRAIN_PCT,
        batch_size=BATCH_SIZE,
        normalize=NORMALIZE,
        transforms=transforms,
    )

    # Optimizer, Loss Function, and Learning Rate Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INIT_LR, eps=EPS, weight_decay=WEIGHT_DECAY
    )
    # weighted_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    pos_weight = torch.tensor(1.0)
    weighted_criterion = torch.nn.BCELoss()
    equal_criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR
    )

    # Initialize Tensorboard SummaryWriter
    log_dir = os.path.join(os.getcwd(), "traineval/")
    writer = SummaryWriter(
        f"runs/{model.__name__}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    print("\n---------------------- TRAINING -----------------------\n")
    print("Device:", DEVICE)

    train_losses, train_accuracies, val_losses, val_accuracies = train(
        model,
        loaders["train"],
        loaders["valid"],
        NUM_EPOCHS,
        weighted_criterion,
        equal_criterion,
        optimizer,
        scheduler,
        PATIENCE,
        DEVICE,
        savepath,
        writer,
    )

    # Load the best model
    model.load_state_dict(torch.load(savepath))

    print("\n--------------------- EVALUATION ----------------------\n")
    # Evaluate the model on test data
    results, y_true, y_prob = evaluate(model, loaders["test"], equal_criterion, DEVICE)

    # Add info to TensorBoard
    example_input = torch.rand(1, 1, 88200, device=DEVICE)
    # writer.add_image('Spectrogram Example', example_spec.squeeze(), dataformats='HW')
    writer.add_graph(model, example_input)
    writer.add_pr_curve("Precision-Recall Curve", y_true, y_prob)
    writer.add_hparams(
        {
            "model": model.__name__,
            "optimizer": optimizer.__class__.__name__,
            "criterion": equal_criterion.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "train_pct": TRAIN_PCT,
            "pos_weight": pos_weight.item(),
            "normalize": True,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "init_lr": INIT_LR,
            "min_lr": MIN_LR,
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "seed": torch.initial_seed(),
        },
        results,
    )
    writer.flush()
    writer.close()

    print("\n---------------------- EXPORTING ----------------------\n")
    # Save the model in ONNX format
    onnx_path = export_onnx(model, savepath, (1, 1, 88200))
    print("ONNX Model exported to:", onnx_path)

    # Compile and save the model to shared library
    so_path = export_AOTcompiled(model, savepath, (1, 1, 88200))
    print("AOT Model exported to:", so_path, "\n")

    # Verify the compiled model
    aot_model = torch._export.aot_load(so_path, "cpu")

    # Test inference rate of compiled model
    cpu_rate, _ = get_inference_rates(aot_model)
    print(f"CPU Inference Rate (AOT Compiled): {cpu_rate*1e6:.4f} µs/inference\n")


if __name__ == "__main__":
    main()
