import torch
from torchvision import transforms
from PIL import Image
from IAA.aesthetic_model import AestheticResNet50
import argparse


def evaluate_image(image_path, model_path):
    # Load image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Load model
    model = AestheticResNet50()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Prediction
    with torch.no_grad():
        output = model(image)

    score = output.item()
    print(f"Aesthetic score for the image is: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Aesthetic Score')
    parser.add_argument('--image_path', type=str, required=True, help='path to the image')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    args = parser.parse_args()

    evaluate_image(args.image_path, args.model_path)
