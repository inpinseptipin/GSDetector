from datetime import datetime
from typing import List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_val_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    weighted_criterion: torch.nn.Module,
    equal_criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    writer: SummaryWriter,
) -> Tuple[float, float, float, float]:

    # Training Loop
    model.train()
    train_loss = 0
    train_correct = 0
    global_step = epoch * len(train_loader)
    with tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{num_epochs} | LR = {scheduler.get_last_lr()[0]:.3e}",
        total=len(train_loader),
        unit=" batch",
    ) as tepoch:
        for inputs, labels in tepoch:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = weighted_criterion(outputs, labels)
            optimizer.zero_grad(True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.round() == labels).sum().item()
            writer.add_scalar("loss/train", loss.item(), global_step)
            tepoch.set_postfix(loss=loss.item())
            global_step += 1

    train_avg_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)
    writer.add_scalar("accuracy/train", train_accuracy, global_step)

    # Validation Loop
    model.eval()
    valid_loss = 0
    valid_correct = 0
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = equal_criterion(outputs, labels)
        valid_loss += loss.item() * inputs.size(0)
        valid_correct += (outputs.round() == labels).sum().item()

    valid_avg_loss = valid_loss / len(valid_loader.dataset)
    valid_accuracy = valid_correct / len(valid_loader.dataset)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
    writer.add_scalar("loss/val", valid_avg_loss, global_step)
    writer.add_scalar("accuracy/val", valid_accuracy, global_step)

    return train_avg_loss, train_accuracy, valid_avg_loss, valid_accuracy


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    weighted_criterion: torch.nn.Module,
    equal_criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    patience: int,
    device: torch.device,
    savepath: str,
    writer: torch.utils.tensorboard.SummaryWriter,
) -> Tuple[torch.nn.Module, List[float], List[float], List[float], List[float]]:

    # Stat logs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_loss = float("inf")
    layout = {
        "Metrics": {
            "Loss": ["Multiline", ["loss/train", "loss/val"]],
            "Accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
        }
    }
    writer.add_custom_scalars(layout)

    # Training
    for epoch in range(num_epochs):
        train_loss, train_acc, val_loss, val_acc = train_val_one_epoch(
            model,
            train_loader,
            valid_loader,
            weighted_criterion,
            equal_criterion,
            optimizer,
            scheduler,
            device,
            epoch,
            num_epochs,
            writer,
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(
            f"Validation: Accuracy = {val_acc*100:.4f}, Mean loss = {val_loss:.8f}",
            end=" ",
        )
        scheduler.step()
        writer.flush()
        if val_loss <= best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), savepath)
            print("---------> CHECKPOINT SAVED")
        elif epoch - best_epoch >= patience:
            print(f"\n\nEARLY STOP ---> No improvement seen for {patience} epochs.")
            break
        else:
            print()

    return train_losses, train_accuracies, val_losses, val_accuracies
