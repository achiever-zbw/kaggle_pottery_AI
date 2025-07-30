from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as transfroms
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms

class PotteryDataset(Dataset):
    """Self_define dataset for pottery images ans masks"""
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        # transform for the binary mask
        self.mask_transform = transforms.Compose([
            transforms.Resize((256,256), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load and process image and masks
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)
        # Convert mask to binary (0 / 1)
        mask = (mask > 0).float() 

        return image, mask
    


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Using device: {device} \n")

    image_dir = r"D:\kaggle_pottery\data\unet_labels\imgs"
    mask_dir = r"D:\kaggle_pottery\data\unet_labels\labels"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = PotteryDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Use DeepLabV3+ as Model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",       
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50
    best_loss = float('inf')
    save_path = r"D:\kaggle_pottery\code\DeepLabV3Plus\best_model.pth"

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()
