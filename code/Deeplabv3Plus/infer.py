import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
import segmentation_models_pytorch as smp

class PredictDataset(Dataset):
    """Self_define dataset for inference , only load images"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size  # (width, height)

        if self.transform:
            image = self.transform(image)

        return image, img_name, orig_size

def predict(model, dataloader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for images, img_names, orig_sizes in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)      
            preds = (probs > 0.5).float()           

            for pred, img_name, orig_size in zip(preds, img_names, orig_sizes):
                pred_mask = pred.squeeze().cpu().numpy() * 255  # [H,W]
                pred_mask = pred_mask.astype(np.uint8)
                pred_mask = Image.fromarray(pred_mask)

                if isinstance(orig_size, torch.Tensor):
                    orig_size = tuple(orig_size.tolist())
                if not (isinstance(orig_size, tuple) and len(orig_size) == 2):
                    orig_size = (256, 256) 
                pred_mask = pred_mask.resize(orig_size, resample=Image.NEAREST)
                save_path = os.path.join(save_dir, os.path.splitext(img_name)[0] + "_pred.png")
                pred_mask.save(save_path)

                print(f"Saved prediction mask: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_dir = r"D:\kaggle_pottery\data\h690\sherd_images"
    save_dir = r"D:\kaggle_pottery\data\h690\sherd_images_masks"

    # Ensure input size matches training -- 256
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    dataset = PredictDataset(img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=3,
        classes=1
    )

    model.load_state_dict(torch.load(r"D:\kaggle_pottery\code\Deeplabv3Plus\best_model.pth", map_location=device))
    model.to(device)

    predict(model, dataloader, device, save_dir)

if __name__ == "__main__":
    main()
    print(f"\n 总图片一共 : {PredictDataset.__len__} 张 \n")
