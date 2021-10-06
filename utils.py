import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
from datasets import BreastPhantom
from PIL import Image
from tqdm import tqdm
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=12,
    pin_memory=True,
):

    train_ds = PhantomDataset(
        image_path=train_dir,
        mask_path=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PhantomDataset(
        image_path=val_dir,
        mask_path=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            #y = y.to(device).unsqueeze(1) 
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, folder='content/predictions', device=DEVICE):
    model.eval()
    for idx, (X, y) in enumerate(loader):
        X = X.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        #print(y.unsqueeze(1).shape)
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()    

   
