import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = getattr(models, checkpoint['arch'])(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model
    except FileNotFoundError:
        print(f"Checkpoint file not found: {filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

def process_image(image_path):
    """
    Process an image file to be used as input for the model.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        torch.Tensor: Processed image tensor
    """
    try:
        img = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(img)
        return img_tensor
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        exit(1)
    except Exception as e:
        print(f"Error processing image: {e}")
        exit(1)


def predict(image_path, model, topk, device):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
        image_path (str): Path to the input image
        model (torchvision.models): Trained model
        topk (int): Number of top classes to return
        device (torch.device): Device to run inference on
    
    Returns:
        tuple: Top probabilities and classes
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        img = process_image(image_path).unsqueeze(0).to(device)  # Add batch dimension
        output = model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_indices = probabilities.topk(topk)
    
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes


def load_category_names(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Category names file not found: {filepath}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {filepath}")
        exit(1)

def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = load_checkpoint(args.checkpoint)
    
    print("Predicting...")
    probs, classes = predict(args.image_path, model, args.top_k, device)
    
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        class_names = [cat_to_name.get(str(cls), "Unknown") for cls in classes]
    else:
        class_names = classes
    
    print("\nTop predictions:")
    for i, (prob, name) in enumerate(zip(probs, class_names), 1):
        print(f"{i}. {name}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
