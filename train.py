import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Train a new network on a data set")
    parser.add_argument('data_directory', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', choices=['vgg13', 'densenet121'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")
    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in ['train', 'valid', 'test']}
    
    return dataloaders, image_datasets

def build_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        in_features = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
    else:
        raise ValueError('Unsupported model architecture')
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        validation_loss, accuracy = validate_model(model, dataloaders['valid'], criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {validation_loss:.3f}.. "
              f"Validation accuracy: {accuracy:.3f}")

def validate_model(model, dataloader, criterion, device):
    model.eval()
    validation_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            validation_loss += batch_loss.item()
            
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return validation_loss/len(dataloader), accuracy/len(dataloader)

def test_model(model, dataloader, device):
    model.eval()
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Test accuracy: {accuracy/len(dataloader):.3f}")

def save_checkpoint(model, optimizer, save_dir, arch, class_to_idx, epochs):
    checkpoint = {
        'arch': arch,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'classifier': model.classifier
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print("Model checkpoint saved successfully.")

def main():
    args = get_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dataloaders, image_datasets = load_data(args.data_directory)
    except FileNotFoundError:
        print(f"Data directory not found: {args.data_directory}")
        return
    
    model = build_model(args.arch, args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    print("Training the model...")
    train_model(model, dataloaders, criterion, optimizer, device, args.epochs)
    
    print("Testing the model...")
    test_model(model, dataloaders['test'], device)
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_checkpoint(model, optimizer, args.save_dir, args.arch, model.class_to_idx, args.epochs)

if __name__ == "__main__":
    main()
