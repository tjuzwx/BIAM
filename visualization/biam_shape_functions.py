"""
BIAM Shape Functions
Visualization of shape functions and feature interactions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

class BIAMShapeFunctions:
    """
    Class for visualizing shape functions in BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize shape function visualizer
        
        Args:
            config: BIAM configuration
        """
        self.config = config
    
    def plot_shape_functions(self, model, data, feature_names=None, save_path=None):
        """
        Plot shape functions for all features
        
        Args:
            model: BIAM model
            data: Input data
            feature_names: Names of features
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
        
        n_features = min(6, data.shape[1])  # Plot up to 6 features
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            
            # Generate range of values for this feature
            feature_range = np.linspace(data[:, i].min(), data[:, i].max(), 100)
            
            # Create input data with varying feature i
            input_data = data.mean(axis=0).reshape(1, -1).repeat(100, axis=0)
            input_data[:, i] = feature_range
            
            # Get model predictions
            with torch.no_grad():
                predictions = model(torch.tensor(input_data, dtype=torch.float32))
                if self.config.task == 'classification':
                    predictions = torch.softmax(predictions, dim=1)
                    predictions = predictions[:, 1]  # Probability of positive class
            
            # Plot shape function
            ax.plot(feature_range, predictions.numpy(), linewidth=2, color='blue')
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel('Model Output')
            ax.set_title(f'Shape Function: {feature_names[i]}')
            ax.grid(True, alpha=0.3)
            
            # Add data points
            ax.scatter(data[:, i], np.zeros_like(data[:, i]), alpha=0.3, s=20, color='red')
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'shape_functions.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_feature_interactions(self, model, data, feature_pairs=None, save_path=None):
        """
        Plot feature interactions
        
        Args:
            model: BIAM model
            data: Input data
            feature_pairs: Pairs of features to plot interactions for
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if feature_pairs is None:
            # Select top 3 feature pairs
            feature_pairs = [(0, 1), (0, 2), (1, 2)]
        
        n_pairs = len(feature_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))
        
        if n_pairs == 1:
            axes = [axes]
        
        for idx, (i, j) in enumerate(feature_pairs):
            ax = axes[idx]
            
            # Create meshgrid for feature interaction
            x_range = np.linspace(data[:, i].min(), data[:, i].max(), 50)
            y_range = np.linspace(data[:, j].min(), data[:, j].max(), 50)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Create input data
            input_data = data.mean(axis=0).reshape(1, -1).repeat(2500, axis=0)
            input_data[:, i] = X.flatten()
            input_data[:, j] = Y.flatten()
            
            # Get model predictions
            with torch.no_grad():
                predictions = model(torch.tensor(input_data, dtype=torch.float32))
                if self.config.task == 'classification':
                    predictions = torch.softmax(predictions, dim=1)
                    predictions = predictions[:, 1]  # Probability of positive class
            
            # Reshape predictions for contour plot
            Z = predictions.numpy().reshape(50, 50)
            
            # Plot interaction
            contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            ax.set_xlabel(f'Feature_{i}')
            ax.set_ylabel(f'Feature_{j}')
            ax.set_title(f'Feature Interaction: {i} vs {j}')
            
            # Add colorbar
            plt.colorbar(contour, ax=ax)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'feature_interactions.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_missing_value_effects(self, model, data, missing_features=None, save_path=None):
        """
        Plot effects of missing values on predictions
        
        Args:
            model: BIAM model
            data: Input data
            missing_features: Features to analyze missing value effects for
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if missing_features is None:
            missing_features = [0, 1, 2]  # Analyze first 3 features
        
        n_features = len(missing_features)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, feature_idx in enumerate(missing_features):
            ax = axes[idx]
            
            # Create data with missing values
            data_missing = data.copy()
            data_missing[:, feature_idx] = np.nan
            
            # Get predictions with missing values
            with torch.no_grad():
                predictions_missing = model(torch.tensor(data_missing, dtype=torch.float32))
                if self.config.task == 'classification':
                    predictions_missing = torch.softmax(predictions_missing, dim=1)
                    predictions_missing = predictions_missing[:, 1]
            
            # Get predictions without missing values
            with torch.no_grad():
                predictions_complete = model(torch.tensor(data, dtype=torch.float32))
                if self.config.task == 'classification':
                    predictions_complete = torch.softmax(predictions_complete, dim=1)
                    predictions_complete = predictions_complete[:, 1]
            
            # Plot comparison
            ax.scatter(predictions_complete.numpy(), predictions_missing.numpy(), alpha=0.6)
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
            ax.set_xlabel('Complete Data Predictions')
            ax.set_ylabel('Missing Data Predictions')
            ax.set_title(f'Missing Value Effect: Feature_{feature_idx}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'missing_value_effects.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_model_complexity(self, model, data, save_path=None):
        """
        Plot model complexity analysis
        
        Args:
            model: BIAM model
            data: Input data
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Prediction variance across samples
        ax1 = axes[0, 0]
        with torch.no_grad():
            predictions = model(torch.tensor(data, dtype=torch.float32))
            if self.config.task == 'classification':
                predictions = torch.softmax(predictions, dim=1)
                predictions = predictions[:, 1]
        
        ax1.hist(predictions.numpy(), bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Model Predictions')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature importance
        ax2 = axes[0, 1]
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            top_features = np.argsort(importance)[-10:]
            ax2.barh(range(len(top_features)), importance[top_features])
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels([f'Feature_{i}' for i in top_features])
            ax2.set_xlabel('Importance')
            ax2.set_title('Top 10 Feature Importance')
        
        # 3. Model confidence
        ax3 = axes[1, 0]
        if self.config.task == 'classification':
            # Calculate prediction confidence
            with torch.no_grad():
                predictions = model(torch.tensor(data, dtype=torch.float32))
                predictions = torch.softmax(predictions, dim=1)
                confidence = torch.max(predictions, dim=1)[0]
            
            ax3.hist(confidence.numpy(), bins=30, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Prediction Confidence')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Model Confidence Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. Residual analysis (for regression)
        ax4 = axes[1, 1]
        if self.config.task == 'regression':
            with torch.no_grad():
                predictions = model(torch.tensor(data, dtype=torch.float32))
                # Assuming we have targets - this would need to be passed as parameter
                # residuals = targets - predictions
                # ax4.scatter(predictions, residuals, alpha=0.6)
                # ax4.axhline(y=0, color='r', linestyle='--')
                # ax4.set_xlabel('Predictions')
                # ax4.set_ylabel('Residuals')
                # ax4.set_title('Residual Analysis')
                ax4.text(0.5, 0.5, 'Residual Analysis\n(requires targets)', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Model Complexity\nAnalysis', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'model_complexity.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
