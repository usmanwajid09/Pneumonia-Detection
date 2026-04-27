import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_config, get_model
from data_loader import create_sample_data_structure, prepare_dataloaders

def train():
    config = load_config('config.yaml')
    data_dir = config['paths']['data_dir']
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Loading data...")
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val') # Use val if exists, else test
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, 'test')
        
    train_paths, train_labels = create_sample_data_structure(train_dir)
    val_paths, val_labels = create_sample_data_structure(val_dir)
    
    print(f"Found {len(train_paths)} training images and {len(val_paths)} validation images.")
    
    # We use batch size from config, but limit num_workers to 0 to avoid multiprocessing issues on windows in notebooks
    config['data']['num_workers'] = 0 
    
    loaders = prepare_dataloaders(
        config,
        train_paths, train_labels,
        val_paths, val_labels
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model(config)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    num_epochs = config['training']['num_epochs']
    
    print(f"Starting training for {num_epochs} epoch(s)...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        epoch_loss = running_loss / len(train_dataset := train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels, _ in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_dataset := val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs} Results:")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")
        
    # Save the trained weights
    weights_path = os.path.join(checkpoint_dir, 'model_weights.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"Training complete. Weights saved to {weights_path}")

if __name__ == '__main__':
    train()
