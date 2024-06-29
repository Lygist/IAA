import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from IAA.aesthetic_dataset import AestheticDataset
from IAA.aesthetic_model import AestheticResNet50


def train_model(epochs, csv_file, img_dir, output_model_path):
    print("Starting training process...")

    # Data loading
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = AestheticDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Total number of images: {len(dataset)}")
    print(f"Training set size: {train_size}, Validation set size: {val_size}")

    # Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AestheticResNet50(pretrained=True)
    model = model.to(device)
    print("Model initialized and moved to cuda")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Loss function and optimizer initialized")

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} started")
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            try:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0
            except Exception as e:
                print(f"Error in training loop: {e}. Skipping this batch.")

        # Save model checkpoint
        torch.save(model.state_dict(), output_model_path)
        print(f"Model checkpoint saved at {output_model_path}")

    print("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train aesthetic score prediction model")
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--csv_file', type=str, required=True, help='path to the csv file with image names and labels')
    parser.add_argument('--img_dir', type=str, required=True, help='directory with all the images')
    parser.add_argument('--output_model_path', type=str, required=True, help='path to save the trained model')
    args = parser.parse_args()

    train_model(args.epochs, args.csv_file, args.img_dir, args.output_model_path)
