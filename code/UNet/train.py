"""
UNet Training Script with SEBlock

- Load images and masks
- Data augmentation and basic processing
- Define Myself Module (UNet++ with SEBlock)
- Define loss function , optimizer
- Training 
- Save the best model
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , Dataset
from torch import optim
from unet_model import UNet
from PIL import Image
import numpy as np


class PotteryDataset(Dataset) :
    """
    Custom dataset for loading images and masks
    """
    def __init__(self , image_dir , mask_dir , transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir)]
        self.mask_transform = transforms.Compose([
            transforms.Resize([256, 256], interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    def __len__(self) :
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir , img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load images and masks
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Apply transform if True
        if self.transform :
            image = self.transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()
        return image , mask

def train_one_epoch(model , dataloader , criterion , optimizer , device) :
    """
    Run one training epoch 
    Return : avg loss
    """
    model.train()
    running_loss = 0.0
    for images , masks in dataloader :
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        # Gain output
        outputs = model(images)
        # Gain loss
        loss = criterion(outputs , masks)
        loss.backward()
        optimizer.step()
        # Criterion return the avg loss
        # Multiply batch_size to get the total loss
        running_loss += loss.item() * images.size(0) # images : [batch_size , channels , H , W]
    # Avg
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Use device : {device} \n")
    image_dir = r"D:\kaggle陶片拼接\data\unet_labels\imgs"
    mask_dir = r"D:\kaggle陶片拼接\data\unet_labels\labels"
    
    transform = transforms.Compose([
        transforms.Resize([256 , 256]) , 
        transforms.ToTensor()
    ])
    
    dataset = PotteryDataset(image_dir , mask_dir , transform)
    dataloader = DataLoader(dataset , batch_size=8 , shuffle=True , num_workers=0)
    
    model = UNet(n_channels=3 , n_classes=1 , bilinear=True).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters() , lr=1e-4)
    
    # Train
    epochs = 250
    best_loss = float('inf')
    
    save_path = r"D:\kaggle陶片拼接\code\UNet\best_model\best_unet_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
    final_path = r"D:\kaggle陶片拼接\code\UNet\best_model\final_unet_model.pth"
    torch.save(model.state_dict(), final_path)
if __name__ == "__main__" :
    main()
    
