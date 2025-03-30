import os
from PIL import Image
import numpy as np
import torch
from .model import SwinUNet
from .dataset import Synapse_dataset
from torch.utils.data import DataLoader
import os

def compute_rmse(y_pred, y_true):
    y_pred = y_pred.float()
    y_true = y_true.float()
    mse = torch.mean((y_pred - y_true) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

def compute_dice_coefficient(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = torch.sum(y_pred * y_true)
    dice = (2. * intersection) / (torch.sum(y_pred) + torch.sum(y_true))
    return dice.item()

def compute_iou(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    iou = intersection / (union)
    return iou.item()

def compute_precision_recall_f1(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    tp = torch.sum(y_pred * y_true)
    fp = torch.sum(y_pred * (1 - y_true))
    fn = torch.sum((1 - y_pred) * y_true)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision.item(), recall.item(), f1_score.item()

def evaluate_model(model, data_loader, device):
    model.eval()
    total_rmse = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            masks = batch["label"].to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.cpu()
            masks = masks.cpu()
            for idx, (image, mask, pred) in enumerate(zip(images, masks, outputs)):
                total_rmse += compute_rmse(pred, mask)
                total_dice += compute_dice_coefficient(pred, mask)
                total_iou += compute_iou(pred, mask)
                precision, recall, f1_score = compute_precision_recall_f1(pred, mask)
                total_precision += precision
                total_recall += recall
                total_f1 += f1_score
            num_samples += images.size(0)
    mean_rmse = total_rmse / num_samples
    mean_dice = total_dice / num_samples
    mean_iou = total_iou / num_samples
    mean_precision = total_precision / num_samples
    mean_recall = total_recall / num_samples
    mean_f1 = total_f1 / num_samples
    return {
        "RMSE": mean_rmse,
        "Dice": mean_dice,
        "IoU": mean_iou,
        "Precision": mean_precision,
        "Recall": mean_recall,
        "F1-Score": mean_f1
    }

def save_inference_results_from_dataloader(data_loader, model, device, save_dir):
    original_dir = os.path.join(save_dir, "original_images")
    predicted_dir = os.path.join(save_dir, "predicted_masks")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)

    model.eval()
    saved_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs).cpu() 
            images = images.cpu()

            for image, pred in zip(images, outputs):
                image_np = image.squeeze().numpy()
                image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
                original_path = os.path.join(original_dir, f"sample_{saved_samples}_original.png")
                Image.fromarray(image_np).save(original_path)
                pred_mask_np = (pred > 0.5).squeeze().numpy() * 255
                predicted_path = os.path.join(predicted_dir, f"sample_{saved_samples}_predicted.png")
                Image.fromarray(pred_mask_np.astype(np.uint8)).save(predicted_path)
                print(f"Saved sample {saved_samples}")
                saved_samples += 1

if __name__ == "__main__":
    test = Synapse_dataset(data_root="/kaggle/input/lung2data/test_2/origin_2", label_root="/kaggle/input/lung2data/test_2/mask_2")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test, batch_size=1, shuffle=True)
    swin_model = SwinUNet(224, 224, 1, 32, 1, 3, 4).to(DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_results = evaluate_model(swin_model, test_loader, DEVICE)
    print(eval_results)
    save_directory = "/kaggle/working/"
    save_inference_results_from_dataloader(test_loader, swin_model, device, save_directory)