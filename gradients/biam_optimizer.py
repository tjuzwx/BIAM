"""
BIAM Optimizer
Main optimizer class for BIAM bilevel optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

class BIAMOptimizer:
    """
    Main optimizer for BIAM model implementing bilevel optimization
    """
    
    def __init__(self, config, biam_model, weighting_network):
        """
        Initialize BIAM optimizer
        
        Args:
            config: BIAM configuration
            biam_model: BIAM model instance
            weighting_network: Weighting network instance
        """
        self.config = config
        self.biam_model = biam_model
        self.weighting_network = weighting_network
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimization parameters
        self.upper_lr = config.upper_lr
        self.lower_lr = config.lower_lr
        self.penalty_coef = config.penalty_coef
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': []
        }
    
    def _initialize_optimizers(self):
        """
        Initialize optimizers for upper and lower level problems
        """
        # Upper level optimizer (for weighting network)
        self.upper_optimizer = optim.SGD(
            self.weighting_network.parameters(),
            lr=self.upper_lr,
            momentum=0.9
        )
        
        # Lower level optimizer (for additive model)
        self.lower_optimizer = optim.SGD(
            self.biam_model.additive_model.parameters(),
            lr=self.lower_lr,
            momentum=0.9
        )
    
    def train_epoch(self, train_loader, val_loader, epoch):
        """
        Train for one epoch using bilevel optimization
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.biam_model.train()
        self.weighting_network.train()
        
        total_train_loss = 0.0
        total_val_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Bilevel optimization steps
            train_loss = self._bilevel_optimization_step(data, target, val_loader, epoch)
            total_train_loss += train_loss
            num_batches += 1
        
        # Calculate average losses
        avg_train_loss = total_train_loss / num_batches
        
        # Evaluate on validation set
        val_loss = self._evaluate_validation(val_loader)
        
        # Store metrics
        self.training_history['train_loss'].append(avg_train_loss)
        self.training_history['val_loss'].append(val_loss)
        
        return {
            'loss': avg_train_loss,
            'val_loss': val_loss
        }
    
    def _bilevel_optimization_step(self, train_data, train_target, val_loader, epoch):
        """
        Single step of bilevel optimization
        
        Args:
            train_data: Training data batch
            train_target: Training targets
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Training loss
        """
        # Step 1: Update lower level parameters (additive model)
        meta_model = self._create_meta_model()
        meta_model.load_state_dict(self.biam_model.additive_model.state_dict())
        
        # Forward pass through meta model
        meta_predictions = meta_model(train_data)
        
        # Calculate weighted loss
        if self.config.task == 'regression':
            meta_losses = F.mse_loss(meta_predictions, train_target, reduction='none')
        else:
            meta_losses = F.cross_entropy(meta_predictions, train_target.long(), reduction='none')
        
        meta_losses = meta_losses.unsqueeze(1)
        
        # Get weights from weighting network
        weights = self.weighting_network(meta_losses.detach())
        
        # Weighted loss
        weighted_loss = torch.mean(meta_losses * weights)
        
        # Compute gradients for meta model
        meta_grads = torch.autograd.grad(
            weighted_loss, meta_model.parameters(), create_graph=True
        )
        
        # Update meta model parameters
        self._update_meta_model(meta_model, meta_grads)
        
        # Step 2: Update upper level parameters (weighting network)
        val_data, val_target = next(iter(val_loader))
        val_data, val_target = val_data.to(self.device), val_target.to(self.device)
        
        # Forward pass through updated meta model
        val_predictions = meta_model(val_data)
        
        # Validation loss
        if self.config.task == 'regression':
            val_loss = F.mse_loss(val_predictions, val_target)
        else:
            val_loss = F.cross_entropy(val_predictions, val_target.long())
        
        # Update weighting network
        self.upper_optimizer.zero_grad()
        val_loss.backward()
        self.upper_optimizer.step()
        
        # Step 3: Update main model parameters
        main_predictions = self.biam_model.additive_model(train_data)
        
        if self.config.task == 'regression':
            main_losses = F.mse_loss(main_predictions, train_target, reduction='none')
        else:
            main_losses = F.cross_entropy(main_predictions, train_target.long(), reduction='none')
        
        main_losses = main_losses.unsqueeze(1)
        
        # Get updated weights
        with torch.no_grad():
            updated_weights = self.weighting_network(main_losses)
        
        # Normalize weights
        if updated_weights.sum() > 0:
            updated_weights = updated_weights / updated_weights.sum() * updated_weights.size(0)
        
        # Weighted loss for main model
        main_weighted_loss = torch.mean(main_losses * updated_weights)
        
        # Add regularization
        reg_loss = self.biam_model.additive_model.compute_regularization_loss('group_lasso')
        total_loss = main_weighted_loss + self.penalty_coef * reg_loss
        
        # Update main model
        self.lower_optimizer.zero_grad()
        total_loss.backward()
        self.lower_optimizer.step()
        
        return total_loss.item()
    
    def _create_meta_model(self):
        """
        Create a copy of the additive model for meta-learning
        """
        from models.biam_additive_model import BIAMAdditiveModel
        meta_model = BIAMAdditiveModel(self.config, self.device)
        return meta_model
    
    def _update_meta_model(self, meta_model, grads):
        """
        Update meta model parameters using gradients
        
        Args:
            meta_model: Meta model to update
            grads: Gradients for parameters
        """
        for param, grad in zip(meta_model.parameters(), grads):
            param.data = param.data - self.lower_lr * grad
    
    def _evaluate_validation(self, val_loader):
        """
        Evaluate model on validation set
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.biam_model.eval()
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                predictions = self.biam_model.additive_model(data)
                
                if self.config.task == 'regression':
                    loss = F.mse_loss(predictions, target)
                else:
                    loss = F.cross_entropy(predictions, target.long())
                
                total_val_loss += loss.item()
                num_batches += 1
        
        return total_val_loss / num_batches
    
    def evaluate(self, test_data):
        """
        Evaluate model on test set
        
        Args:
            test_data: Test data tuple (X, y)
            
        Returns:
            Dictionary with test metrics
        """
        self.biam_model.eval()
        
        X_test, y_test = test_data
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.biam_model.additive_model(X_test)
            
            if self.config.task == 'regression':
                mse = F.mse_loss(predictions, y_test).item()
                mae = F.l1_loss(predictions, y_test).item()
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
            else:
                # Classification metrics
                pred_classes = torch.argmax(predictions, dim=1)
                accuracy = accuracy_score(y_test.cpu().numpy(), pred_classes.cpu().numpy())
                
                # F1 score
                f1 = f1_score(y_test.cpu().numpy(), pred_classes.cpu().numpy(), average='weighted')
                
                metrics = {
                    'accuracy': accuracy,
                    'f1_score': f1
                }
        
        return metrics
    
    def get_training_history(self):
        """
        Get training history
        
        Returns:
            Training history dictionary
        """
        return self.training_history
    
    def save_checkpoint(self, filepath):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'biam_model_state_dict': self.biam_model.state_dict(),
            'weighting_network_state_dict': self.weighting_network.state_dict(),
            'upper_optimizer_state_dict': self.upper_optimizer.state_dict(),
            'lower_optimizer_state_dict': self.lower_optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.biam_model.load_state_dict(checkpoint['biam_model_state_dict'])
        self.weighting_network.load_state_dict(checkpoint['weighting_network_state_dict'])
        self.upper_optimizer.load_state_dict(checkpoint['upper_optimizer_state_dict'])
        self.lower_optimizer.load_state_dict(checkpoint['lower_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
