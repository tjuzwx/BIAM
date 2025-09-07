"""
BIAM Bilevel Optimizer
Advanced bilevel optimization algorithms for BIAM model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
import numpy as np

class BIAMBilevelOptimizer:
    """
    Advanced bilevel optimizer for BIAM model
    """
    
    def __init__(self, config, upper_model, lower_model):
        """
        Initialize bilevel optimizer
        
        Args:
            config: BIAM configuration
            upper_model: Upper level model (weighting network)
            lower_model: Lower level model (additive model)
        """
        self.config = config
        self.upper_model = upper_model
        self.lower_model = lower_model
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimization parameters
        self.upper_lr = config.upper_lr
        self.lower_lr = config.lower_lr
        self.momentum = getattr(config, 'momentum', 0.9)
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Optimization history
        self.optimization_history = {
            'upper_loss': [],
            'lower_loss': [],
            'gradient_norms': []
        }
    
    def _initialize_optimizers(self):
        """
        Initialize optimizers for upper and lower level problems
        """
        # Upper level optimizer with advanced features
        self.upper_optimizer = optim.AdamW(
            self.upper_model.parameters(),
            lr=self.upper_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Lower level optimizer with advanced features
        self.lower_optimizer = optim.AdamW(
            self.lower_model.parameters(),
            lr=self.lower_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.upper_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.upper_optimizer, T_max=1000, eta_min=1e-6
        )
        
        self.lower_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.lower_optimizer, T_max=1000, eta_min=1e-6
        )
    
    def bilevel_step(self, train_data, train_target, val_data, val_target, epoch):
        """
        Single step of bilevel optimization
        
        Args:
            train_data: Training data
            train_target: Training targets
            val_data: Validation data
            val_target: Validation targets
            epoch: Current epoch
            
        Returns:
            Dictionary with optimization metrics
        """
        # Step 1: Lower level optimization (inner loop)
        lower_loss = self._lower_level_step(train_data, train_target)
        
        # Step 2: Upper level optimization (outer loop)
        upper_loss = self._upper_level_step(val_data, val_target)
        
        # Update learning rates
        self.upper_scheduler.step()
        self.lower_scheduler.step()
        
        # Store history
        self.optimization_history['upper_loss'].append(upper_loss)
        self.optimization_history['lower_loss'].append(lower_loss)
        
        return {
            'upper_loss': upper_loss,
            'lower_loss': lower_loss,
            'upper_lr': self.upper_scheduler.get_last_lr()[0],
            'lower_lr': self.lower_scheduler.get_last_lr()[0]
        }
    
    def _lower_level_step(self, train_data, train_target):
        """
        Lower level optimization step
        
        Args:
            train_data: Training data
            train_target: Training targets
            
        Returns:
            Lower level loss
        """
        self.lower_model.train()
        
        # Forward pass
        predictions = self.lower_model(train_data)
        
        # Calculate loss
        if self.config.task == 'regression':
            loss = nn.MSELoss()(predictions, train_target)
        else:
            loss = nn.CrossEntropyLoss()(predictions, train_target.long())
        
        # Add regularization
        reg_loss = self._compute_regularization_loss()
        total_loss = loss + self.config.penalty_coef * reg_loss
        
        # Backward pass
        self.lower_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.lower_model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.lower_optimizer.step()
        
        return total_loss.item()
    
    def _upper_level_step(self, val_data, val_target):
        """
        Upper level optimization step
        
        Args:
            val_data: Validation data
            val_target: Validation targets
            
        Returns:
            Upper level loss
        """
        self.upper_model.train()
        self.lower_model.eval()
        
        # Get sample weights from upper model
        with torch.no_grad():
            # Use validation loss as input to weighting network
            val_predictions = self.lower_model(val_data)
            if self.config.task == 'regression':
                val_losses = nn.MSELoss(reduction='none')(val_predictions, val_target)
            else:
                val_losses = nn.CrossEntropyLoss(reduction='none')(val_predictions, val_target.long())
            
            val_losses = val_losses.unsqueeze(1)
        
        # Get weights from upper model
        weights = self.upper_model(val_losses)
        
        # Calculate weighted validation loss
        weighted_loss = torch.mean(val_losses * weights)
        
        # Add entropy regularization for weight diversity
        entropy_reg = -torch.mean(weights * torch.log(weights + 1e-8))
        total_loss = weighted_loss + 0.01 * entropy_reg
        
        # Backward pass
        self.upper_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.upper_model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.upper_optimizer.step()
        
        return total_loss.item()
    
    def _compute_regularization_loss(self):
        """
        Compute regularization loss for lower level model
        
        Returns:
            Regularization loss
        """
        reg_loss = 0.0
        
        # L2 regularization
        for param in self.lower_model.parameters():
            reg_loss += torch.norm(param, p=2)
        
        # Group Lasso regularization on feature groups
        if hasattr(self.lower_model, 'predict'):
            weights = self.lower_model.predict.weight
            total_dim = weights.shape[1]
            spline_dim = self.config.get_spline_dim()
            
            for i in range(0, total_dim, spline_dim):
                group_weights = weights[:, i:i+spline_dim]
                reg_loss += torch.norm(group_weights, p=2)
        
        return reg_loss
    
    def get_optimization_history(self):
        """
        Get optimization history
        
        Returns:
            Optimization history dictionary
        """
        return self.optimization_history
    
    def save_optimizer_state(self, filepath):
        """
        Save optimizer state
        
        Args:
            filepath: Path to save state
        """
        state = {
            'upper_optimizer_state': self.upper_optimizer.state_dict(),
            'lower_optimizer_state': self.lower_optimizer.state_dict(),
            'upper_scheduler_state': self.upper_scheduler.state_dict(),
            'lower_scheduler_state': self.lower_scheduler.state_dict(),
            'optimization_history': self.optimization_history
        }
        
        torch.save(state, filepath)
    
    def load_optimizer_state(self, filepath):
        """
        Load optimizer state
        
        Args:
            filepath: Path to state file
        """
        state = torch.load(filepath, map_location=self.device)
        
        self.upper_optimizer.load_state_dict(state['upper_optimizer_state'])
        self.lower_optimizer.load_state_dict(state['lower_optimizer_state'])
        self.upper_scheduler.load_state_dict(state['upper_scheduler_state'])
        self.lower_scheduler.load_state_dict(state['lower_scheduler_state'])
        self.optimization_history = state['optimization_history']
