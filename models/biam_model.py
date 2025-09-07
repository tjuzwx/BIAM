"""
BIAM Model
Main model class that integrates additive model and weighting network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np

class BIAMModel(nn.Module):
    """
    Main BIAM model that combines additive model with weighting network
    """
    
    def __init__(self, config, device):
        """
        Initialize BIAM model
        
        Args:
            config: BIAM configuration
            device: Device to run on
        """
        super(BIAMModel, self).__init__()
        
        self.config = config
        self.device = device
        self.task = config.task
        
        # Initialize additive model
        self.additive_model = BIAMAdditiveModel(config, device)
        
        # Initialize weighting network
        self.weighting_network = BIAMWeightingNetwork(config, device)
        
        # Move to device
        self.to(device)
    
    def forward(self, x, return_weights=False):
        """
        Forward pass through BIAM model
        
        Args:
            x: Input features
            return_weights: Whether to return sample weights
            
        Returns:
            Model predictions and optionally weights
        """
        # Get predictions from additive model
        predictions = self.additive_model(x)
        
        if return_weights:
            # Calculate sample weights using weighting network
            with torch.no_grad():
                # Use prediction confidence as input to weighting network
                if self.task == 'regression':
                    # For regression, use prediction variance as uncertainty measure
                    prediction_uncertainty = torch.var(predictions, dim=1, keepdim=True)
                else:
                    # For classification, use prediction confidence
                    prediction_uncertainty = F.softmax(predictions, dim=1).max(dim=1, keepdim=True)[0]
                
                weights = self.weighting_network(prediction_uncertainty)
            
            return predictions, weights
        else:
            return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from additive model
        
        Returns:
            Feature importance scores
        """
        return self.additive_model.get_feature_importance()
    
    def get_missing_indicators(self):
        """
        Get missing value indicators
        
        Returns:
            Missing value indicators
        """
        return self.additive_model.get_missing_indicators()
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty estimation
        
        Args:
            x: Input features
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Mean predictions and uncertainty estimates
        """
        predictions = []
        
        for _ in range(n_samples):
            # Add small noise for uncertainty estimation
            x_noisy = x + torch.randn_like(x) * 0.01
            pred = self.forward(x_noisy)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
