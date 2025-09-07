"""
BIAM Additive Model
Additive model component with missing value handling and feature interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple

class BIAMAdditiveModel(nn.Module):
    """
    Additive model for BIAM with missing value indicators and feature interactions
    """
    
    def __init__(self, config, device):
        """
        Initialize BIAM additive model
        
        Args:
            config: BIAM configuration
            device: Device to run on
        """
        super(BIAMAdditiveModel, self).__init__()
        
        self.config = config
        self.device = device
        self.task = config.task
        
        # Model parameters
        self.input_dim = getattr(config, 'input_dim', 100)
        self.output_dim = 1 if self.task == 'regression' else getattr(config, 'num_classes', 2)
        self.spline_dim = 3 if self.task == 'regression' else 5
        
        # Calculate total dimension after spline expansion
        self.total_dim = self.input_dim * self.spline_dim
        
        # Define model components
        self._build_model()
        
        # Move to device
        self.to(device)
    
    def _build_model(self):
        """
        Build the additive model architecture
        """
        # Main prediction layer
        self.predict = nn.Linear(self.total_dim, self.output_dim)
        
        # Missing value indicators
        self.missing_indicators = nn.Parameter(
            torch.zeros(self.input_dim, device=self.device)
        )
        
        # Feature interaction weights
        self.interaction_weights = nn.Parameter(
            torch.zeros(self.input_dim, self.input_dim, device=self.device)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights
        """
        nn.init.xavier_uniform_(self.predict.weight)
        if self.predict.bias is not None:
            nn.init.constant_(self.predict.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through additive model
        
        Args:
            x: Input features
            
        Returns:
            Model predictions
        """
        # Apply spline transformation
        x_spline = self._apply_spline_transformation(x)
        
        # Add missing value indicators
        x_with_missing = self._add_missing_indicators(x_spline, x)
        
        # Apply feature interactions
        x_with_interactions = self._apply_feature_interactions(x_with_missing, x)
        
        # Final prediction
        output = self.predict(x_with_interactions)
        
        return output
    
    def _apply_spline_transformation(self, x):
        """
        Apply B-spline transformation to input features
        
        Args:
            x: Input features
            
        Returns:
            Spline-transformed features
        """
        batch_size, n_features = x.shape
        x_spline = torch.zeros(batch_size, self.total_dim, device=self.device)
        
        for i in range(n_features):
            start_idx = i * self.spline_dim
            end_idx = start_idx + self.spline_dim
            
            # Simple spline basis functions
            x_feature = x[:, i:i+1]
            
            # Linear spline basis
            x_spline[:, start_idx] = x_feature.squeeze()
            x_spline[:, start_idx + 1] = torch.relu(x_feature - 0.5).squeeze()
            x_spline[:, start_idx + 2] = torch.relu(x_feature - 0.8).squeeze()
            
            if self.spline_dim > 3:
                x_spline[:, start_idx + 3] = torch.sin(x_feature * np.pi).squeeze()
                x_spline[:, start_idx + 4] = torch.cos(x_feature * np.pi).squeeze()
        
        return x_spline
    
    def _add_missing_indicators(self, x_spline, x_original):
        """
        Add missing value indicators to features
        
        Args:
            x_spline: Spline-transformed features
            x_original: Original input features
            
        Returns:
            Features with missing indicators
        """
        # Create missing indicators
        missing_mask = torch.isnan(x_original)
        
        # Add missing indicators as additional features
        missing_indicators = missing_mask.float()
        
        # Concatenate with spline features
        x_with_missing = torch.cat([x_spline, missing_indicators], dim=1)
        
        return x_with_missing
    
    def _apply_feature_interactions(self, x, x_original):
        """
        Apply feature interactions
        
        Args:
            x: Current features
            x_original: Original input features
            
        Returns:
            Features with interactions
        """
        # Simple pairwise interactions
        interactions = []
        
        for i in range(min(5, x_original.shape[1])):  # Limit interactions for efficiency
            for j in range(i + 1, min(5, x_original.shape[1])):
                interaction = x_original[:, i] * x_original[:, j]
                interactions.append(interaction.unsqueeze(1))
        
        if interactions:
            interaction_features = torch.cat(interactions, dim=1)
            x_with_interactions = torch.cat([x, interaction_features], dim=1)
        else:
            x_with_interactions = x
        
        return x_with_interactions
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            Feature importance scores
        """
        with torch.no_grad():
            # Calculate importance based on weight magnitudes
            weights = self.predict.weight
            importance = torch.norm(weights, dim=0)
            
            # Reshape to match original features
            feature_importance = torch.zeros(self.input_dim, device=self.device)
            
            for i in range(self.input_dim):
                start_idx = i * self.spline_dim
                end_idx = start_idx + self.spline_dim
                feature_importance[i] = importance[start_idx:end_idx].sum()
            
            return feature_importance.cpu().numpy()
    
    def get_missing_indicators(self):
        """
        Get missing value indicators
        
        Returns:
            Missing value indicators
        """
        return self.missing_indicators.detach().cpu().numpy()
    
    def get_interaction_weights(self):
        """
        Get feature interaction weights
        
        Returns:
            Interaction weights
        """
        return self.interaction_weights.detach().cpu().numpy()
    
    def compute_regularization_loss(self, regularization_type='group_lasso'):
        """
        Compute regularization loss
        
        Args:
            regularization_type: Type of regularization
            
        Returns:
            Regularization loss
        """
        if regularization_type == 'group_lasso':
            # Group Lasso regularization on feature groups
            weights = self.predict.weight
            total_dim = weights.shape[1]
            spline_dim = self.spline_dim
            
            reg_loss = 0.0
            for i in range(0, total_dim, spline_dim):
                group_weights = weights[:, i:i+spline_dim]
                reg_loss += torch.norm(group_weights, p=2)
            
            return reg_loss
        
        elif regularization_type == 'l1':
            # L1 regularization
            return torch.norm(self.predict.weight, p=1)
        
        elif regularization_type == 'l2':
            # L2 regularization
            return torch.norm(self.predict.weight, p=2)
        
        else:
            return torch.tensor(0.0, device=self.device)
    
    def get_model_interpretation(self, x_sample):
        """
        Get model interpretation for a sample
        
        Args:
            x_sample: Input sample
            
        Returns:
            Dictionary with interpretation results
        """
        with torch.no_grad():
            # Get feature contributions
            x_spline = self._apply_spline_transformation(x_sample)
            contributions = x_spline * self.predict.weight[0]
            
            # Reshape contributions by feature
            feature_contributions = torch.zeros(self.input_dim, device=self.device)
            for i in range(self.input_dim):
                start_idx = i * self.spline_dim
                end_idx = start_idx + self.spline_dim
                feature_contributions[i] = contributions[0, start_idx:end_idx].sum()
            
            interpretation = {
                'feature_contributions': feature_contributions.cpu().numpy(),
                'missing_indicators': self.get_missing_indicators(),
                'interaction_weights': self.get_interaction_weights(),
                'prediction': self.forward(x_sample).cpu().numpy()
            }
            
            return interpretation
