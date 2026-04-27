"""
ResNet-18 Model for Pneumonia Detection
Modified for binary classification with governance features
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class PneumoniaResNet18(nn.Module):
    """
    ResNet-18 architecture for binary pneumonia classification.
    
    Features:
    - Pretrained on ImageNet
    - Modified final layer for binary classification
    - Optional feature extraction mode
    """
    
    def __init__(
        self, 
        num_classes: int = 2, 
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes (default: 2 for Normal/Pneumonia)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone layers for feature extraction
        """
        super(PneumoniaResNet18, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone if requested (for transfer learning)
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Get the number of features in the final layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Add dropout for regularization
            nn.Linear(num_features, num_classes)
        )
        
        # Store feature extractor for Grad-CAM
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.resnet(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Feature tensor of shape (batch_size, num_features)
        """
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return features
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True
    
    def freeze_early_layers(self, num_layers_to_freeze: int = 6):
        """
        Freeze early layers of the network.
        
        Args:
            num_layers_to_freeze: Number of initial layers to freeze
        """
        layers = list(self.resnet.children())[:num_layers_to_freeze]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False


def create_model(config: dict) -> PneumoniaResNet18:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model = PneumoniaResNet18(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    # Test the model
    model = PneumoniaResNet18(num_classes=2, pretrained=True)
    
    # Print model summary
    trainable, total = count_parameters(model)
    print(f"Model: ResNet-18 for Pneumonia Detection")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
