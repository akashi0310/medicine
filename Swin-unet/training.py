from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from .model import SwinUNet
from .dataset import Synapse_dataset
from torch.utils.data import DataLoader
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNet(224, 224, 1, 32, 1, 3, 4).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(SwinUNet.parameters(), lr=1e-4, weight_decay=1e-5)

BATCH_SIZE = 12

train = Synapse_dataset(data_root = "/kaggle/input/training-data-for-lung-canceraugmented/augmented_scans" , label_root = "/kaggle/input/training-data-for-lung-canceraugmented/augmented_masks")
valid = Synapse_dataset(data_root = "/kaggle/input/lung2data/valid_2/origin_2", label_root = "/kaggle/input/lung2data/valid_2/mask_2")
test = Synapse_dataset(data_root = '/kaggle/input/full-lungsegmentation/test_2/origin_2', label_root = "/kaggle/input/full-lungsegmentation/test_2/mask_2")

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE,shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE,shuffle=True)

#This is for the unaugmented dataset as I was too lazy to merge the augmented and unaugmented dataset
train_non = Synapse_dataset(data_root = "/kaggle/input/lung2data/train_2/origin_2" , label_root = "/kaggle/input/lung2data/train_2/mask_2")
train_non_loader = torch.utils.data.DataLoader(train_non, batch_size=BATCH_SIZE,shuffle=True)
def train_epoch(model, dataloader):
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        out = model(batch["image"].to(DEVICE))  
        target = batch["label"].unsqueeze(1).to(DEVICE).float()  
        loss = loss_fn(out, target)  
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate_epoch(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            out = model(batch["image"].to(DEVICE))
            target = batch["label"].unsqueeze(1).to(DEVICE).float()  
            loss = loss_fn(out, target)  
            losses.append(loss.item())
    return np.mean(losses)
    

def train_on_unagumented(model, dataloader):
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc="Training on unaugmented", leave=False):
        optimizer.zero_grad()
        out = model(batch["image"].to(DEVICE))
        target = batch["label"].unsqueeze(1).to(DEVICE).float() 
        loss = loss_fn(out, target)  
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    losses.append(loss.item())
    return losses

def train(model, epochs, min_epochs, early_stop_count):
    best_valid_loss = float('inf')
    EARLY_STOP = early_stop_count
    for ep in range(epochs):
        print(f"Epoch {ep + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader)
        train_non = train_on_unagumented(model,train_non_loader)
        valid_loss = validate_epoch(model, test_loader)
        print(f'Epoch: {ep + 1}: train_loss={train_loss:.5f}, valid_loss={np.mean(valid_loss):.5f}')
        if np.mean(valid_loss) < best_valid_loss:
            best_valid_loss = np.mean(valid_loss)
            torch.save(model.state_dict(), "best_model_dice.pth")
            print("Best model saved!")
            if ep >= min_epochs:
                EARLY_STOP = early_stop_count 

        if ep >= min_epochs:
            if np.mean(valid_loss) >= best_valid_loss:
                EARLY_STOP -= 1
                print(f"Early stopping counter: {EARLY_STOP}/{early_stop_count}")
                if EARLY_STOP <= 0:
                    print("Early stopping triggered.")
                    return train_loss, valid_loss
            else:
                pass

    return train_loss, valid_loss

if __name__ == "__main__":
    train(model, 150, 100, 10)