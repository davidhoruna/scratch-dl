import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from configs.train_config import TrainConfig
from models.unet import UNet
from datasets.custom_dataset import SegmentationDataset

def train(config: TrainConfig):
    # Seed and device setup
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    # Dataset and dataloaders
    train_dataset = SegmentationDataset(config.data_dir / "images", config.data_dir / "masks", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # Model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(device)

    # Loss + optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config.num_epochs} | Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), config.save_model_path)

if __name__ == "__main__":
    config = TrainConfig()
    train(config)
