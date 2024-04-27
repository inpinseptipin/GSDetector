from datetime import date, datetime
import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from sklearn import metrics
from tqdm.auto import tqdm

from datasets import GunshotDataset, LightweightGSDataset
from utils import compile_and_save
from models import *


def train_val_one_epoch(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    num_epochs,
    writer
):
    """
    Train the model for one epoch then validate.

    Args:
        model (torch.nn.Module): Model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        valid_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to be used for training and validation.
        epoch (int): Current epoch.
        num_epochs (int): Total number of epochs.
        writer: Summary writer for logging.

    Returns:
        float: Training loss.
        float: Training accuracy.
        float: Validation loss.
        float: Validation accuracy.
    """
    model.train()
    train_loss = 0
    train_correct = 0
    global_step = epoch * len(train_loader)
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} | LR = {scheduler.get_last_lr()[0]:.2E}', unit='batch', total=len(train_loader)) as tepoch:
        for step, (inputs, labels) in enumerate(tepoch):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.sigmoid().round() == labels).sum().item()
            writer.add_scalar('loss/train', loss.item(), global_step)
            tepoch.set_postfix(loss=loss.item())
            global_step += 1
            
    train_avg_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)
    writer.add_scalar('accuracy/train', train_accuracy, global_step)
    
    model.eval()
    valid_loss = 0
    valid_correct = 0
    
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        valid_loss += criterion(outputs, labels).item() * inputs.size(0)
        valid_correct += (outputs.sigmoid().round() == labels).sum().item()
    
    valid_avg_loss = valid_loss / len(valid_loader.dataset)
    valid_accuracy = valid_correct / len(valid_loader.dataset)
    writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
    writer.add_scalar('loss/val', valid_avg_loss, global_step)
    writer.add_scalar('accuracy/val', valid_accuracy, global_step)
    
    return train_avg_loss, train_accuracy, valid_avg_loss, valid_accuracy


@torch.no_grad()
def evaluate(loader, model, criterion, device):
    """
    Evaluate the model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): Data loader.
        model (torch.nn.Module): Model to be evaluated.
        criterion: Loss function.
        device: Device to be used for evaluation.

    Returns:
        float: Average loss.
        float: Accuracy.
        np.ndarray: Predicted probabilities.
        np.ndarray: True labels.
    """
    model.eval()
    y_prob = []
    y_true = []
    total_loss = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item() * inputs.size(0)
        
        y_prob.extend(outputs.sigmoid().cpu().numpy())
        y_true.extend(labels.cpu().numpy())
    
    # Convert lists to NumPy arrays for metric computation
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    # Metrics calculation
    avg_loss = total_loss / len(loader.dataset)
    accuracy = metrics.accuracy_score(y_true, y_prob.round())  # Calculate accuracy
    
    return avg_loss, accuracy, y_prob, y_true


def main():
    
    # Hyperparameters
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    INIT_LR = 1e-4
    WEIGHT_DECAY = 1e-8
    PATIENCE = 30
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Device:", DEVICE)
    
    # Model Initialization
    model_name = 'EnhancedCNN_CosineAnnealingLR'
    model = EnhancedCNN()
    model.to(DEVICE)

    # Dataset Initialization
    cadre_1 = LightweightGSDataset(root_dir="data/processed/cadre_1")
    ncbi_1 = LightweightGSDataset(root_dir="data/processed/ncbi_1")
    kaggle_1 = LightweightGSDataset(root_dir="data/processed/kaggle_1")
    urban_1 = LightweightGSDataset(root_dir="data/processed/urban_1")
    esc50_0 = LightweightGSDataset(root_dir="data/processed/esc50_0")
    campus_0 = LightweightGSDataset(root_dir="data/processed/campus_0")
    urban_0 = LightweightGSDataset(root_dir="data/processed/urban_0")

    positives = ConcatDataset([cadre_1, ncbi_1, kaggle_1, urban_1])
    negatives = ConcatDataset([esc50_0, campus_0, urban_0])
    pos_weight = torch.tensor(len(negatives) / len(positives))

    full_dataset = ConcatDataset([positives, negatives])
    total_indices = torch.randperm(len(full_dataset))
    shuffled_dataset = torch.utils.data.Subset(full_dataset, total_indices)

    # Train/Val/Test Split
    total_size = len(shuffled_dataset)
    train_size = int(0.70 * total_size)
    remaining_size = total_size - train_size
    test_size = remaining_size // 2
    valid_size = remaining_size - test_size
    train_dataset, valid_dataset, test_dataset = random_split(shuffled_dataset, [train_size, valid_size, test_size])

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0)

    # Objects for Training
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_loss = float('inf')
    savepath = f"trained_models/pickle/{model_name}.pt"
    writer = SummaryWriter('runs/' + model_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    layout = {'Metrics':{'Loss': ['Multiline',['loss/train', 'loss/val']],
                         'Accuracy': ['Multiline',['accuracy/train', 'accuracy/val']]}}
    writer.add_custom_scalars(layout)
    # writer.add_custom_scalars_multilinechart(['loss/train', 'loss/val'], title='Loss')
    # writer.add_custom_scalars_multilinechart(['accuracy/train', 'accuracy/val'], title='Accuracy')

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, val_loss, val_acc = train_val_one_epoch(
            model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            scheduler,
            DEVICE,
            epoch,
            NUM_EPOCHS,
            writer
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation: Accuracy = {val_acc*100:.4f}, Avg loss = {val_loss:.8f}", end=" ")
        scheduler.step()
        writer.flush()
        if val_loss <= best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), savepath)
            print("-----> New best model saved.")
        elif epoch - best_epoch >= PATIENCE:
            print(f"\n\nNo improvement for {PATIENCE} Epochs - stopped early.")
            break
        else:
            print()
    

    # Load the best model
    model.load_state_dict(torch.load(savepath))
    
    # Evaluate on the test set
    test_loss, test_acc, y_prob, y_true = evaluate(test_loader, model, criterion, DEVICE)
    y_pred = y_prob.round()
    print(f"\nTest Accuracy = {test_acc*100:.4f} %")
    print(f"Average Test Loss = {test_loss:.6f}")

    # Calculate FPR and FNR
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print(f"False-Positive Rate: {fpr*100:.3f} %")
    print(f"False-Negative Rate: {fnr*100:.3f} %")
    
    # Calculate ROC curve and AUC
    roc_auc = metrics.roc_auc_score(y_true, y_prob)
    print(f"ROC AUC: {roc_auc:.6f}")

    # Calculate precision, recall, and F1 score
    print("\nTESTING RESULTS: ")
    print(metrics.classification_report(y_true, y_pred, digits=6, target_names=['negative samples', 'positive samples']))
    
    # Compile and save the model
    model.cpu()
    compile_and_save(model, savepath, save_onnx=True)
    
    # Add info to TensorBoard
    example_spec, _ = test_dataset[0]
    # writer.add_image('Spectrogram Example', example_spec.squeeze(), dataformats='HW')
    writer.add_graph(model, example_spec.unsqueeze(0))
    writer.add_pr_curve('Precision-Recall Curve', y_true, y_prob)
    writer.flush()
    writer.close()
    


if __name__ == "__main__":
    main()