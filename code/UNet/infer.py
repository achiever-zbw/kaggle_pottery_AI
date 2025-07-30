"""
UNet Inference Script

- Load train model
- Predict masks on unlabeled images
- Save predicted masks as PNG 
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , Dataset
from PIL import Image
import numpy as np
from unet_model import UNet
import matplotlib.pyplot as plt

class PredictDataset(Dataset) :
    """Dataset precessing class for predict data
    """
    def __init__(self , image_dir , transform = None) :
        super().__init__()
        self.image_dir = image_dir
        self.image_names = [f for f in os.listdir(image_dir)]
        self.transform = transform
        
    def __len__(self) :
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_dir , img_name)
        
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size
    
        if self.transform :
            image = self.transform(image)
        
        return image , img_name , orig_size
    
def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Use device : {device} \n")
    
    img_dir = r"D:\kaggle_pottery\data\h690\sherd_images"
    save_dir = r"D:\kaggle_pottery\data\h690\sherd_images_masks"
    model_path = r"D:\kaggle_pottery\code\UNet\best_model\final_unet_model.pth"
    os.makedirs(save_dir , exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize([256 , 256]) , 
        transforms.ToTensor()
    ])
    
    dataset = PredictDataset(img_dir, transform)
    dataloder = DataLoader(dataset , batch_size=1 , shuffle=True)
    
    # Init UNet model
    model = UNet(n_channels=3 , n_classes=1 , bilinear=True).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad() :
        for img , img_name , orig_size in dataloder :
            img = img.to(device)
            output = model(img)
            output = torch.sigmoid(output)
            output = output.squeeze().cpu().numpy()
            output = (output > 0.5).astype(np.uint8) * 255
            
            # Resize back to original size
            pred_mask = Image.fromarray(output)
            pred_mask = pred_mask.resize(orig_size , resample=Image.NEAREST)

            save_path = os.path.join(save_dir, os.path.splitext(img_name[0])[0] + "_pred.png")
            pred_mask.save(save_path)
            print(f"Saved: {save_path}")
            
if __name__ == '__main__' :
    main()
    print("\n 所有图片预测完毕 \n")