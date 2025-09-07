"""
BIAM Visualizer
Main visualization class for BIAM model interpretation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os

class BIAMVisualizer:
    """
    Visualization class for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize BIAM visualizer
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create visualization directory
        self.viz_dir = "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def plot_training_curves(self, training_history, save_path=None):
        """
        Plot training curves
        
        Args:
            training_history: Training history dictionary
            save_path: Path to save plot
        """
        epochs = training_history['epochs']
        train_loss = training_history['train_loss']
        val_loss = training_history['val_loss']
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Training Loss', linewidth=2)
        plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot test metrics
        if 'test_metrics' in training_history and training_history['test_metrics']:
            plt.subplot(1, 2, 2)
            test_metrics = training_history['test_metrics']
            
            if self.config.task == 'regression':
                rmse_values = [m.get('rmse', 0) for m in test_metrics]
                plt.plot(epochs, rmse_values, label='Test RMSE', linewidth=2, color='green')
                plt.ylabel('RMSE')
                plt.title('Test RMSE')
            else:
                acc_values = [m.get('accuracy', 0) for m in test_metrics]
                f1_values = [m.get('f1_score', 0) for m in test_metrics]
                plt.plot(epochs, acc_values, label='Test Accuracy', linewidth=2, color='green')
                plt.plot(epochs, f1_values, label='Test F1 Score', linewidth=2, color='orange')
                plt.ylabel('Score')
                plt.title('Test Metrics')
            
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'training_curves.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_feature_importance(self, feature_importance, feature_names=None, top_k=20, save_path=None):
        """
        Plot feature importance
        
        Args:
            feature_importance: Feature importance scores
            feature_names: Names of features
            top_k: Number of top features to show
            save_path: Path to save plot
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        top_indices = sorted_indices[:top_k]
        
        top_importance = feature_importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_importance)), top_importance)
        plt.yticks(range(len(top_importance)), top_names)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'feature_importance.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_missing_value_analysis(self, missing_indicators, feature_names=None, save_path=None):
        """
        Plot missing value analysis
        
        Args:
            missing_indicators: Missing value indicators
            feature_names: Names of features
            save_path: Path to save plot
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(missing_indicators))]
        
        # Filter non-zero indicators
        non_zero_mask = np.abs(missing_indicators) > 1e-6
        non_zero_indicators = missing_indicators[non_zero_mask]
        non_zero_names = [feature_names[i] for i in np.where(non_zero_mask)[0]]
        
        if len(non_zero_indicators) == 0:
            print("No significant missing value indicators found.")
            return None
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(non_zero_indicators)), non_zero_indicators)
        plt.xticks(range(len(non_zero_indicators)), non_zero_names, rotation=45, ha='right')
        plt.ylabel('Missing Value Indicator')
        plt.title('Missing Value Indicators')
        plt.grid(True, alpha=0.3)
        
        # Color bars by sign
        colors = ['red' if x < 0 else 'blue' for x in non_zero_indicators]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'missing_value_analysis.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_sample_weights(self, sample_weights, save_path=None):
        """
        Plot sample weights distribution
        
        Args:
            sample_weights: Sample weights
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 5))
        
        # Histogram of weights
        plt.subplot(1, 2, 1)
        plt.hist(sample_weights, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Sample Weight')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sample Weights')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(sample_weights)
        plt.ylabel('Sample Weight')
        plt.title('Sample Weights Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'sample_weights.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_model_interpretation(self, interpretation, save_path=None):
        """
        Plot model interpretation results
        
        Args:
            interpretation: Model interpretation dictionary
            save_path: Path to save plot
        """
        n_plots = 0
        if 'feature_contributions' in interpretation:
            n_plots += 1
        if 'missing_indicators' in interpretation:
            n_plots += 1
        if 'interaction_weights' in interpretation:
            n_plots += 1
        
        if n_plots == 0:
            print("No interpretation data to plot.")
            return None
        
        plt.figure(figsize=(15, 5 * n_plots))
        plot_idx = 1
        
        # Feature contributions
        if 'feature_contributions' in interpretation:
            plt.subplot(n_plots, 1, plot_idx)
            contributions = interpretation['feature_contributions']
            sorted_indices = np.argsort(np.abs(contributions))[::-1]
            
            top_contributions = contributions[sorted_indices[:10]]
            top_names = [f"Feature_{i}" for i in sorted_indices[:10]]
            
            bars = plt.bar(range(len(top_contributions)), top_contributions)
            plt.xticks(range(len(top_contributions)), top_names, rotation=45, ha='right')
            plt.ylabel('Contribution')
            plt.title('Top Feature Contributions')
            plt.grid(True, alpha=0.3)
            
            # Color bars by sign
            colors = ['red' if x < 0 else 'blue' for x in top_contributions]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plot_idx += 1
        
        # Missing indicators
        if 'missing_indicators' in interpretation:
            plt.subplot(n_plots, 1, plot_idx)
            missing_indicators = interpretation['missing_indicators']
            non_zero_mask = np.abs(missing_indicators) > 1e-6
            
            if np.any(non_zero_mask):
                non_zero_indicators = missing_indicators[non_zero_mask]
                non_zero_names = [f"Feature_{i}" for i in np.where(non_zero_mask)[0]]
                
                bars = plt.bar(range(len(non_zero_indicators)), non_zero_indicators)
                plt.xticks(range(len(non_zero_indicators)), non_zero_names, rotation=45, ha='right')
                plt.ylabel('Missing Indicator')
                plt.title('Missing Value Indicators')
                plt.grid(True, alpha=0.3)
                
                # Color bars by sign
                colors = ['red' if x < 0 else 'blue' for x in non_zero_indicators]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            else:
                plt.text(0.5, 0.5, 'No significant missing indicators', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Missing Value Indicators')
            
            plot_idx += 1
        
        # Interaction weights
        if 'interaction_weights' in interpretation:
            plt.subplot(n_plots, 1, plot_idx)
            interaction_weights = interpretation['interaction_weights']
            
            # Plot heatmap of interaction weights
            im = plt.imshow(interaction_weights, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im)
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Index')
            plt.title('Feature Interaction Weights')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'model_interpretation.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_data_distribution(self, X, y, feature_names=None, save_path=None):
        """
        Plot data distribution
        
        Args:
            X: Input features
            y: Target labels
            feature_names: Names of features
            save_path: Path to save plot
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        n_features = min(6, X.shape[1])  # Plot up to 6 features
        
        plt.figure(figsize=(15, 10))
        
        for i in range(n_features):
            plt.subplot(2, 3, i + 1)
            
            if self.config.task == 'regression':
                plt.scatter(X[:, i], y, alpha=0.6, s=20)
                plt.xlabel(feature_names[i])
                plt.ylabel('Target')
                plt.title(f'{feature_names[i]} vs Target')
            else:
                # Classification: plot by class
                unique_classes = np.unique(y)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
                
                for j, class_label in enumerate(unique_classes):
                    mask = y == class_label
                    plt.scatter(X[mask, i], np.ones(np.sum(mask)) * j, 
                              alpha=0.6, s=20, color=colors[j], label=f'Class {class_label}')
                
                plt.xlabel(feature_names[i])
                plt.ylabel('Class')
                plt.title(f'{feature_names[i]} by Class')
                plt.legend()
            
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.viz_dir, 'data_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
