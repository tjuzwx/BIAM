"""
BIAM Weighting Network
Neural network for learning sample weights in bilevel optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class BIAMWeightingNetwork(nn.Module):
    """
    Weighting network for BIAM model that learns sample weights
    """
    
    def __init__(self, config, device):
        """
        Initialize BIAM weighting network
        
        Args:
            config: BIAM configuration
            device: Device to run on
        """
        super(BIAMWeightingNetwork, self).__init__()
        
        self.config = config
        self.device = device
        
        # Network architecture based on MWNet paper
        self.input_dim = 1  # Input is loss/error values
        self.hidden_dim = 10
        self.output_dim = 1  # Output is weight
        
        # Define network layers
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(device)
    
    def _initialize_weights(self):
        """
        Initialize network weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through weighting network
        
        Args:
            x: Input features (typically loss values)
            
        Returns:
            Sample weights
        """
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        
        # Apply sigmoid to ensure positive weights
        weights = torch.sigmoid(x)
        
        return weights
    
    def get_weight_statistics(self, x):
        """
        Get statistics about learned weights
        
        Args:
            x: Input features
            
        Returns:
            Dictionary with weight statistics
        """
        with torch.no_grad():
            weights = self.forward(x)
            
            stats = {
                'mean_weight': weights.mean().item(),
                'std_weight': weights.std().item(),
                'min_weight': weights.min().item(),
                'max_weight': weights.max().item(),
                'weight_entropy': -(weights * torch.log(weights + 1e-8)).sum().item()
            }
            
            return stats
    
    def regularize_weights(self, weights, regularization_type='entropy'):
        """
        Apply regularization to weights
        
        Args:
            weights: Sample weights
            regularization_type: Type of regularization
            
        Returns:
            Regularized weights
        """
        if regularization_type == 'entropy':
            # Entropy regularization to encourage diversity
            entropy_loss = -(weights * torch.log(weights + 1e-8)).sum()
            return weights, entropy_loss
        elif regularization_type == 'sparsity':
            # Sparsity regularization
            sparsity_loss = torch.sum(weights)
            return weights, sparsity_loss
        else:
            return weights, torch.tensor(0.0, device=self.device)
    
    def update_weights_dynamically(self, losses, epoch, total_epochs):
        """
        Dynamically update weights based on training progress
        
        Args:
            losses: Current loss values
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Updated weights
        """
        # Base weights from network
        base_weights = self.forward(losses)
        
        # Dynamic adjustment based on training progress
        progress = epoch / total_epochs
        
        # Early in training: focus on high-loss samples
        # Later in training: focus on low-loss samples
        if progress < 0.5:
            # Focus on high-loss samples
            dynamic_factor = 1.0 + progress
        else:
            # Focus on low-loss samples
            dynamic_factor = 2.0 - progress
        
        # Apply dynamic adjustment
        adjusted_weights = base_weights * dynamic_factor
        
        # Normalize weights
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum() * adjusted_weights.size(0)
        
        return adjusted_weights
