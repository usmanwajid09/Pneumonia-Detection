"""
Robustness Testing Module
Tests model robustness against Gaussian noise and other perturbations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm


class RobustnessTester:
    """
    Test model robustness against various perturbations.
    
    Key Tests:
    - Gaussian Noise
    - AUC Drop calculation
    - Performance degradation analysis
    """
    
    def __init__(self, noise_levels: List[float] = None):
        """
        Initialize robustness tester.
        
        Args:
            noise_levels: List of noise standard deviations to test
        """
        self.noise_levels = noise_levels or [0.0, 0.05, 0.1, 0.15, 0.2]
        self.test_results = {}
    
    def add_gaussian_noise(
        self, 
        images: torch.Tensor, 
        noise_std: float
    ) -> torch.Tensor:
        """
        Add Gaussian noise to images.
        
        Args:
            images: Input tensor of shape (batch_size, C, H, W)
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Noisy images tensor
        """
        if noise_std == 0.0:
            return images
        
        noise = torch.randn_like(images) * noise_std
        noisy_images = images + noise
        
        # Clip to valid range (assuming normalized images)
        # For ImageNet normalization, this keeps values reasonable
        noisy_images = torch.clamp(noisy_images, -3, 3)
        
        return noisy_images
    
    def evaluate_with_noise(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        noise_std: float
    ) -> Dict[str, float]:
        """
        Evaluate model performance with added Gaussian noise.
        
        Args:
            model: Trained model
            data_loader: DataLoader
            device: Device to run on
            noise_std: Noise standard deviation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(data_loader, desc=f"Noise σ={noise_std:.2f}", leave=False):
                # Add noise to images
                noisy_images = self.add_gaussian_noise(images, noise_std)
                noisy_images = noisy_images.to(device)
                
                # Get predictions
                outputs = model(noisy_images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        # Calculate AUC if we have both classes
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probabilities)
        else:
            auc = 0.0
        
        return {
            'noise_std': noise_std,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'num_samples': len(labels)
        }
    
    def test_robustness(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict:
        """
        Perform comprehensive robustness testing.
        
        Args:
            model: Trained model
            data_loader: DataLoader
            device: Device to run on
            
        Returns:
            Dictionary containing robustness test results
        """
        print("\n" + "="*70)
        print("ROBUSTNESS TESTING")
        print("="*70)
        
        results = []
        
        for noise_level in self.noise_levels:
            metrics = self.evaluate_with_noise(
                model, data_loader, device, noise_level
            )
            results.append(metrics)
            
            print(f"Noise σ={noise_level:.2f}: "
                  f"Accuracy={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}, "
                  f"AUC={metrics['auc']:.4f}")
        
        # Calculate robustness metrics
        clean_auc = results[0]['auc']
        max_noise_auc = results[-1]['auc']
        auc_drop = clean_auc - max_noise_auc
        
        # Calculate average performance drop
        clean_acc = results[0]['accuracy']
        avg_noisy_acc = np.mean([r['accuracy'] for r in results[1:]])
        avg_acc_drop = clean_acc - avg_noisy_acc
        
        # Robustness score (1 - normalized AUC drop)
        robustness_score = 1 - auc_drop
        
        self.test_results = {
            'detailed_results': results,
            'clean_auc': clean_auc,
            'max_noise_auc': max_noise_auc,
            'auc_drop': auc_drop,
            'avg_accuracy_drop': avg_acc_drop,
            'robustness_score': robustness_score
        }
        
        print("\n" + "-"*70)
        print(f"📊 ROBUSTNESS SUMMARY:")
        print(f"  • Clean AUC: {clean_auc:.4f}")
        print(f"  • Max Noise AUC (σ={self.noise_levels[-1]}): {max_noise_auc:.4f}")
        print(f"  • AUC Drop: {auc_drop:.4f}")
        print(f"  • Average Accuracy Drop: {avg_acc_drop:.4f}")
        print(f"  • Robustness Score: {robustness_score:.4f}")
        print("="*70 + "\n")
        
        return self.test_results
    
    def visualize_robustness(self, save_path: str = None):
        """
        Create visualizations of robustness test results.
        
        Args:
            save_path: Path to save the figure
        """
        if not self.test_results:
            print("No test results available. Run test_robustness() first.")
            return
        
        results = self.test_results['detailed_results']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Robustness Testing Results', fontsize=16, fontweight='bold')
        
        noise_levels = [r['noise_std'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        aucs = [r['auc'] for r in results]
        
        # 1. Accuracy vs Noise
        ax1 = axes[0]
        ax1.plot(noise_levels, accuracies, marker='o', linewidth=2, markersize=8, label='Accuracy')
        ax1.axhline(y=accuracies[0], color='r', linestyle='--', alpha=0.5, label='Clean Performance')
        ax1.set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs Gaussian Noise', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1])
        
        # 2. F1-Score vs Noise
        ax2 = axes[1]
        ax2.plot(noise_levels, f1_scores, marker='s', linewidth=2, markersize=8, 
                color='green', label='F1-Score')
        ax2.axhline(y=f1_scores[0], color='r', linestyle='--', alpha=0.5, label='Clean Performance')
        ax2.set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_title('F1-Score vs Gaussian Noise', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1])
        
        # 3. AUC vs Noise
        ax3 = axes[2]
        ax3.plot(noise_levels, aucs, marker='^', linewidth=2, markersize=8, 
                color='purple', label='AUC')
        ax3.axhline(y=aucs[0], color='r', linestyle='--', alpha=0.5, label='Clean Performance')
        ax3.fill_between(noise_levels, aucs, aucs[0], alpha=0.2, color='red')
        ax3.set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
        ax3.set_ylabel('AUC', fontsize=12)
        ax3.set_title('AUC vs Gaussian Noise (Drop Highlighted)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim([0, 1])
        
        # Add AUC drop annotation
        auc_drop = self.test_results['auc_drop']
        ax3.text(0.5, 0.5, f'AUC Drop: {auc_drop:.4f}', 
                transform=ax3.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Robustness visualization saved to: {save_path}")
        
        plt.show()
    
    def visualize_noisy_samples(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_samples: int = 5,
        save_path: str = None
    ):
        """
        Visualize sample images with different noise levels and predictions.
        
        Args:
            model: Trained model
            data_loader: DataLoader
            device: Device to run on
            num_samples: Number of samples to visualize
            save_path: Path to save the figure
        """
        model.eval()
        
        # Get a batch of images
        images, labels, _ = next(iter(data_loader))
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        fig, axes = plt.subplots(num_samples, len(self.noise_levels), 
                                figsize=(3*len(self.noise_levels), 3*num_samples))
        
        fig.suptitle('Model Predictions on Noisy Images', fontsize=16, fontweight='bold')
        
        class_names = ['Normal', 'Pneumonia']
        
        with torch.no_grad():
            for i in range(num_samples):
                img = images[i:i+1]
                label = labels[i].item()
                
                for j, noise_std in enumerate(self.noise_levels):
                    # Add noise
                    noisy_img = self.add_gaussian_noise(img, noise_std).to(device)
                    
                    # Get prediction
                    output = model(noisy_img)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(output, dim=1).item()
                    confidence = probs[0, pred].item()
                    
                    # Denormalize for display
                    display_img = noisy_img.cpu().squeeze(0) * std + mean
                    display_img = torch.clamp(display_img, 0, 1)
                    display_img = display_img.permute(1, 2, 0).numpy()
                    
                    # Plot
                    ax = axes[i, j] if num_samples > 1 else axes[j]
                    ax.imshow(display_img)
                    ax.axis('off')
                    
                    # Add title
                    color = 'green' if pred == label else 'red'
                    title = f"σ={noise_std:.2f}\nPred: {class_names[pred]} ({confidence:.2f})"
                    if j == 0:
                        title = f"True: {class_names[label]}\n" + title
                    
                    ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Noisy samples visualization saved to: {save_path}")
        
        plt.show()
    
    def generate_robustness_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive robustness report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if not self.test_results:
            return "No test results available. Run test_robustness() first."
        
        report = []
        report.append("=" * 70)
        report.append("ROBUSTNESS TEST REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall metrics
        report.append("📊 OVERALL ROBUSTNESS METRICS:")
        report.append(f"  • Clean AUC: {self.test_results['clean_auc']:.4f}")
        report.append(f"  • Maximum Noise AUC (σ={self.noise_levels[-1]}): {self.test_results['max_noise_auc']:.4f}")
        report.append(f"  • AUC Drop: {self.test_results['auc_drop']:.4f}")
        report.append(f"  • Average Accuracy Drop: {self.test_results['avg_accuracy_drop']:.4f}")
        report.append(f"  • Robustness Score: {self.test_results['robustness_score']:.4f}")
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS BY NOISE LEVEL:")
        for result in self.test_results['detailed_results']:
            report.append(f"\n  Noise σ = {result['noise_std']:.2f}:")
            report.append(f"    • Accuracy: {result['accuracy']:.4f}")
            report.append(f"    • F1-Score: {result['f1_score']:.4f}")
            report.append(f"    • AUC: {result['auc']:.4f}")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


if __name__ == "__main__":
    # Test robustness tester
    print("Testing Robustness Tester...")
    
    # Create dummy test
    tester = RobustnessTester(noise_levels=[0.0, 0.1, 0.2])
    
    # Test noise addition
    dummy_images = torch.randn(4, 3, 224, 224)
    noisy_images = tester.add_gaussian_noise(dummy_images, noise_std=0.1)
    
    print(f"✓ Noise addition tested")
    print(f"  Original shape: {dummy_images.shape}")
    print(f"  Noisy shape: {noisy_images.shape}")
    print(f"  Mean noise magnitude: {(noisy_images - dummy_images).abs().mean():.4f}")
