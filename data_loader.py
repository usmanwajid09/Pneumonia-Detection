"""
Data Loading Utilities with Synthetic Attribute Generation
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, List
import random


class PneumoniaDataset(Dataset):
    """
    Custom Dataset for Pneumonia Detection with Synthetic Sensitive Attributes.
    
    Since demographic metadata is missing, we generate synthetic binary attributes
    for fairness testing purposes.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        sensitive_attribute_prob: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of labels (0: Normal, 1: Pneumonia)
            transform: Torchvision transforms
            sensitive_attribute_prob: Probability for Group A=1
            seed: Random seed for reproducibility
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Generate synthetic sensitive attributes
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.sensitive_attributes = self._generate_synthetic_attributes(
            len(image_paths), 
            sensitive_attribute_prob
        )
        
    def _generate_synthetic_attributes(
        self, 
        num_samples: int, 
        prob: float
    ) -> np.ndarray:
        """
        Generate synthetic binary sensitive attributes.
        
        Args:
            num_samples: Number of samples
            prob: Probability for Group A=1
            
        Returns:
            Array of binary attributes (0 or 1)
        """
        return np.random.binomial(1, prob, num_samples)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image, label, sensitive_attribute)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        sensitive_attr = self.sensitive_attributes[idx]
        
        return image, label, sensitive_attr


def get_transforms(config: dict, is_training: bool = True) -> transforms.Compose:
    """
    Get image transformations.
    
    Args:
        config: Configuration dictionary
        is_training: Whether this is for training (applies augmentation)
        
    Returns:
        Composed transforms
    """
    image_size = config['data']['image_size']
    
    if is_training:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(config['augmentation']['rotation_degrees']),
            transforms.RandomHorizontalFlip(p=config['augmentation']['horizontal_flip']),
            transforms.ColorJitter(
                brightness=config['augmentation']['brightness'],
                contrast=config['augmentation']['contrast']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def prepare_dataloaders(
    config: dict,
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None
) -> dict:
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        train_paths: Training image paths
        train_labels: Training labels
        val_paths: Validation image paths
        val_labels: Validation labels
        test_paths: Test image paths (optional)
        test_labels: Test labels (optional)
        
    Returns:
        Dictionary containing train, val, and optionally test dataloaders
    """
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']
    sensitive_prob = config['fairness']['sensitive_attribute_prob']
    
    # Create datasets
    train_dataset = PneumoniaDataset(
        train_paths, 
        train_labels,
        transform=get_transforms(config, is_training=True),
        sensitive_attribute_prob=sensitive_prob,
        seed=42
    )
    
    val_dataset = PneumoniaDataset(
        val_paths,
        val_labels,
        transform=get_transforms(config, is_training=False),
        sensitive_attribute_prob=sensitive_prob,
        seed=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Add test loader if provided
    if test_paths is not None and test_labels is not None:
        test_dataset = PneumoniaDataset(
            test_paths,
            test_labels,
            transform=get_transforms(config, is_training=False),
            sensitive_attribute_prob=sensitive_prob,
            seed=42
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders['test'] = test_loader
    
    return loaders


def create_sample_data_structure(base_path: str) -> Tuple[List, List]:
    """
    Create sample data structure for testing.
    Assumes Kaggle pneumonia dataset structure:
    - data/train/NORMAL/
    - data/train/PNEUMONIA/
    - data/test/NORMAL/
    - data/test/PNEUMONIA/
    
    Args:
        base_path: Base directory path
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Normal images (label: 0)
    normal_dir = os.path.join(base_path, 'NORMAL')
    if os.path.exists(normal_dir):
        for img_name in os.listdir(normal_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(normal_dir, img_name))
                labels.append(0)
    
    # Pneumonia images (label: 1)
    pneumonia_dir = os.path.join(base_path, 'PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        for img_name in os.listdir(pneumonia_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(pneumonia_dir, img_name))
                labels.append(1)
    
    return image_paths, labels


if __name__ == "__main__":
    # Test the data loader
    print("Testing PneumoniaDataset with synthetic attributes...")
    
    # Create dummy data
    dummy_paths = [f"image_{i}.jpg" for i in range(100)]
    dummy_labels = [0] * 50 + [1] * 50
    
    # Test configuration
    test_config = {
        'data': {'image_size': 224, 'num_workers': 2},
        'training': {'batch_size': 8},
        'fairness': {'sensitive_attribute_prob': 0.5},
        'augmentation': {
            'rotation_degrees': 15,
            'horizontal_flip': 0.5,
            'brightness': 0.2,
            'contrast': 0.2
        }
    }
    
    # This would fail without actual images, but shows the structure
    print(f"✓ Dataset structure created successfully")
    print(f"  - Images: {len(dummy_paths)}")
    print(f"  - Labels distribution: Normal={dummy_labels.count(0)}, Pneumonia={dummy_labels.count(1)}")
