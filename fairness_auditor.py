"""
Fairness Auditing Module
Implements Demographic Parity Difference (DPD) and other fairness metrics
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FairnessAuditor:
    """
    Auditor for evaluating fairness metrics in ML models.
    
    Key Metrics:
    - Demographic Parity Difference (DPD)
    - Equal Opportunity Difference
    - Equalized Odds
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize fairness auditor.
        
        Args:
            threshold: Acceptable threshold for DPD (default: 0.1 or 10%)
        """
        self.threshold = threshold
        self.audit_results = {}
    
    def calculate_demographic_parity_difference(
        self,
        predictions: np.ndarray,
        sensitive_attributes: np.ndarray
    ) -> float:
        """
        Calculate Demographic Parity Difference (DPD).
        
        DPD measures the difference in positive prediction rates between groups.
        DPD = |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
        
        Args:
            predictions: Binary predictions (0 or 1)
            sensitive_attributes: Binary sensitive attribute (0 or 1)
            
        Returns:
            DPD value (0 means perfect parity, higher means more disparity)
        """
        # Separate predictions by group
        group_0_preds = predictions[sensitive_attributes == 0]
        group_1_preds = predictions[sensitive_attributes == 1]
        
        # Calculate positive prediction rates
        if len(group_0_preds) > 0:
            rate_group_0 = np.mean(group_0_preds)
        else:
            rate_group_0 = 0.0
            
        if len(group_1_preds) > 0:
            rate_group_1 = np.mean(group_1_preds)
        else:
            rate_group_1 = 0.0
        
        # Calculate absolute difference
        dpd = abs(rate_group_0 - rate_group_1)
        
        return dpd
    
    def calculate_equal_opportunity_difference(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        sensitive_attributes: np.ndarray
    ) -> float:
        """
        Calculate Equal Opportunity Difference.
        
        Measures difference in True Positive Rates between groups.
        EOD = |TPR(A=0) - TPR(A=1)|
        
        Args:
            y_true: True labels
            predictions: Predicted labels
            sensitive_attributes: Binary sensitive attribute
            
        Returns:
            Equal Opportunity Difference
        """
        # Calculate TPR for each group
        tpr_0 = self._calculate_tpr(
            y_true[sensitive_attributes == 0],
            predictions[sensitive_attributes == 0]
        )
        
        tpr_1 = self._calculate_tpr(
            y_true[sensitive_attributes == 1],
            predictions[sensitive_attributes == 1]
        )
        
        return abs(tpr_0 - tpr_1)
    
    def calculate_equalized_odds(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        sensitive_attributes: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Equalized Odds (both TPR and FPR differences).
        
        Args:
            y_true: True labels
            predictions: Predicted labels
            sensitive_attributes: Binary sensitive attribute
            
        Returns:
            Tuple of (TPR difference, FPR difference)
        """
        # TPR difference
        tpr_diff = self.calculate_equal_opportunity_difference(
            y_true, predictions, sensitive_attributes
        )
        
        # FPR difference
        fpr_0 = self._calculate_fpr(
            y_true[sensitive_attributes == 0],
            predictions[sensitive_attributes == 0]
        )
        
        fpr_1 = self._calculate_fpr(
            y_true[sensitive_attributes == 1],
            predictions[sensitive_attributes == 1]
        )
        
        fpr_diff = abs(fpr_0 - fpr_1)
        
        return tpr_diff, fpr_diff
    
    def _calculate_tpr(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate True Positive Rate."""
        if len(y_true) == 0:
            return 0.0
        
        true_positives = np.sum((y_true == 1) & (predictions == 1))
        actual_positives = np.sum(y_true == 1)
        
        if actual_positives == 0:
            return 0.0
        
        return true_positives / actual_positives
    
    def _calculate_fpr(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate False Positive Rate."""
        if len(y_true) == 0:
            return 0.0
        
        false_positives = np.sum((y_true == 0) & (predictions == 1))
        actual_negatives = np.sum(y_true == 0)
        
        if actual_negatives == 0:
            return 0.0
        
        return false_positives / actual_negatives
    
    def audit_model(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict:
        """
        Perform comprehensive fairness audit on model.
        
        Args:
            model: Trained model
            data_loader: DataLoader containing images, labels, and sensitive attributes
            device: Device to run inference on
            
        Returns:
            Dictionary containing fairness metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_sensitive_attrs = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels, sensitive_attrs in data_loader:
                images = images.to(device)
                
                # Get predictions
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_sensitive_attrs.extend(sensitive_attrs.numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        sensitive_attrs = np.array(all_sensitive_attrs)
        probabilities = np.array(all_probabilities)
        
        # Calculate fairness metrics
        dpd = self.calculate_demographic_parity_difference(
            predictions, sensitive_attrs
        )
        
        eod = self.calculate_equal_opportunity_difference(
            labels, predictions, sensitive_attrs
        )
        
        tpr_diff, fpr_diff = self.calculate_equalized_odds(
            labels, predictions, sensitive_attrs
        )
        
        # Calculate group-specific metrics
        group_metrics = self._calculate_group_metrics(
            labels, predictions, probabilities, sensitive_attrs
        )
        
        # Store results
        self.audit_results = {
            'demographic_parity_difference': dpd,
            'equal_opportunity_difference': eod,
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'group_metrics': group_metrics,
            'dpd_acceptable': dpd <= self.threshold
        }
        
        return self.audit_results
    
    def _calculate_group_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        sensitive_attrs: np.ndarray
    ) -> Dict:
        """Calculate performance metrics for each group."""
        metrics = {}
        
        for group in [0, 1]:
            group_mask = sensitive_attrs == group
            group_labels = labels[group_mask]
            group_preds = predictions[group_mask]
            group_probs = probabilities[group_mask]
            
            if len(group_labels) == 0:
                continue
            
            # Calculate metrics
            accuracy = np.mean(group_labels == group_preds)
            
            # Confusion matrix
            if len(np.unique(group_labels)) > 1:
                tn, fp, fn, tp = confusion_matrix(
                    group_labels, group_preds
                ).ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0.0
            
            positive_rate = np.mean(group_preds == 1)
            avg_confidence = np.mean(group_probs)
            
            metrics[f'group_{group}'] = {
                'size': len(group_labels),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'positive_prediction_rate': positive_rate,
                'average_confidence': avg_confidence
            }
        
        return metrics
    
    def generate_fairness_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive fairness report.
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            Report as string
        """
        if not self.audit_results:
            return "No audit results available. Run audit_model() first."
        
        report = []
        report.append("=" * 70)
        report.append("FAIRNESS AUDIT REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall fairness metrics
        report.append("📊 FAIRNESS METRICS:")
        report.append(f"  • Demographic Parity Difference (DPD): {self.audit_results['demographic_parity_difference']:.4f}")
        report.append(f"    {'✓ ACCEPTABLE' if self.audit_results['dpd_acceptable'] else '✗ EXCEEDS THRESHOLD'} (threshold: {self.threshold})")
        report.append(f"  • Equal Opportunity Difference: {self.audit_results['equal_opportunity_difference']:.4f}")
        report.append(f"  • TPR Difference: {self.audit_results['tpr_difference']:.4f}")
        report.append(f"  • FPR Difference: {self.audit_results['fpr_difference']:.4f}")
        report.append("")
        
        # Group-specific metrics
        report.append("👥 GROUP-SPECIFIC METRICS:")
        for group_name, metrics in self.audit_results['group_metrics'].items():
            report.append(f"\n  {group_name.upper().replace('_', ' ')}:")
            report.append(f"    • Sample Size: {metrics['size']}")
            report.append(f"    • Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"    • Precision: {metrics['precision']:.4f}")
            report.append(f"    • Recall: {metrics['recall']:.4f}")
            report.append(f"    • F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"    • Positive Prediction Rate: {metrics['positive_prediction_rate']:.4f}")
            report.append(f"    • Average Confidence: {metrics['average_confidence']:.4f}")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def visualize_fairness_metrics(self, save_path: str = None):
        """
        Create visualizations of fairness metrics.
        
        Args:
            save_path: Path to save the figure
        """
        if not self.audit_results:
            print("No audit results available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fairness Audit Visualization', fontsize=16, fontweight='bold')
        
        group_metrics = self.audit_results['group_metrics']
        
        # 1. Performance comparison
        ax1 = axes[0, 0]
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        group_0_values = [group_metrics['group_0'][m] for m in metrics_names]
        group_1_values = [group_metrics['group_1'][m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax1.bar(x - width/2, group_0_values, width, label='Group 0', alpha=0.8)
        ax1.bar(x + width/2, group_1_values, width, label='Group 1', alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics by Group')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.capitalize() for m in metrics_names])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Fairness metrics bar chart
        ax2 = axes[0, 1]
        fairness_names = ['DPD', 'EOD', 'TPR Diff', 'FPR Diff']
        fairness_values = [
            self.audit_results['demographic_parity_difference'],
            self.audit_results['equal_opportunity_difference'],
            self.audit_results['tpr_difference'],
            self.audit_results['fpr_difference']
        ]
        
        colors = ['red' if fairness_values[0] > self.threshold else 'green'] + ['blue'] * 3
        ax2.bar(fairness_names, fairness_values, color=colors, alpha=0.7)
        ax2.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold})')
        ax2.set_ylabel('Difference')
        ax2.set_title('Fairness Metrics')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Group size comparison
        ax3 = axes[1, 0]
        sizes = [group_metrics['group_0']['size'], group_metrics['group_1']['size']]
        ax3.pie(sizes, labels=['Group 0', 'Group 1'], autopct='%1.1f%%', startangle=90)
        ax3.set_title('Sample Distribution')
        
        # 4. Positive prediction rates
        ax4 = axes[1, 1]
        ppr_0 = group_metrics['group_0']['positive_prediction_rate']
        ppr_1 = group_metrics['group_1']['positive_prediction_rate']
        ax4.bar(['Group 0', 'Group 1'], [ppr_0, ppr_1], color=['blue', 'orange'], alpha=0.7)
        ax4.set_ylabel('Rate')
        ax4.set_title('Positive Prediction Rate by Group')
        ax4.set_ylim([0, 1])
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fairness visualization saved to: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test fairness auditor
    print("Testing Fairness Auditor...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate biased predictions
    sensitive_attrs = np.random.binomial(1, 0.5, n_samples)
    predictions = np.random.binomial(1, 0.7, n_samples)
    
    # Introduce bias: Group 1 has higher positive prediction rate
    predictions[sensitive_attrs == 1] = np.random.binomial(1, 0.8, np.sum(sensitive_attrs == 1))
    
    # Initialize auditor
    auditor = FairnessAuditor(threshold=0.1)
    
    # Calculate DPD
    dpd = auditor.calculate_demographic_parity_difference(predictions, sensitive_attrs)
    print(f"\n✓ Demographic Parity Difference: {dpd:.4f}")
    print(f"  {'Acceptable' if dpd <= 0.1 else 'Exceeds threshold'}")
