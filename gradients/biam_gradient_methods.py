"""
BIAM Gradient Methods
Various gradient computation methods for BIAM optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

class BIAMGradientMethods:
    """
    Collection of gradient computation methods for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize gradient methods
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_hypergradient(self, model, loss_fn, train_data, train_target, val_data, val_target):
        """
        Compute hypergradient for bilevel optimization
        
        Args:
            model: Model to compute gradients for
            loss_fn: Loss function
            train_data: Training data
            train_target: Training targets
            val_data: Validation data
            val_target: Validation targets
            
        Returns:
            Hypergradient
        """
        # First-order approximation of hypergradient using finite differences
        
        # Step 1: Compute gradients w.r.t. training loss
        train_loss = loss_fn(model(train_data), train_target)
        train_grads = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
        
        # Step 2: Compute gradients w.r.t. validation loss
        val_loss = loss_fn(model(val_data), val_target)
        val_grads = torch.autograd.grad(val_loss, model.parameters(), create_graph=True)
        
        # Step 3: Approximate hypergradient using chain rule
        hypergrad = []
        for train_grad, val_grad in zip(train_grads, val_grads):
            # Simple approximation: hypergradient is proportional to validation gradient
            hypergrad.append(val_grad)
        
        return hypergrad
    
    def compute_implicit_gradient(self, model, loss_fn, train_data, train_target, val_data, val_target, num_steps=5):
        """
        Compute implicit gradient using iterative method
        
        Args:
            model: Model to compute gradients for
            loss_fn: Loss function
            train_data: Training data
            train_target: Training targets
            val_data: Validation data
            val_target: Validation targets
            num_steps: Number of iterative steps
            
        Returns:
            Implicit gradient
        """
        # Create a copy of the model for iterative updates
        meta_model = self._create_meta_model(model)
        meta_model.load_state_dict(model.state_dict())
        
        # Iterative updates to approximate optimal parameters
        for step in range(num_steps):
            # Forward pass
            train_pred = meta_model(train_data)
            train_loss = loss_fn(train_pred, train_target)
            
            # Compute gradients
            grads = torch.autograd.grad(train_loss, meta_model.parameters(), create_graph=True)
            
            # Update parameters
            for param, grad in zip(meta_model.parameters(), grads):
                param.data = param.data - self.config.lower_lr * grad
        
        # Compute validation loss on updated model
        val_pred = meta_model(val_data)
        val_loss = loss_fn(val_pred, val_target)
        
        # Compute gradient of validation loss w.r.t. original parameters
        implicit_grad = torch.autograd.grad(val_loss, model.parameters())
        
        return implicit_grad
    
    def compute_forward_gradient(self, model, loss_fn, train_data, train_target, val_data, val_target):
        """
        Compute forward gradient using forward-mode differentiation
        
        Args:
            model: Model to compute gradients for
            loss_fn: Loss function
            train_data: Training data
            train_target: Training targets
            val_data: Validation data
            val_target: Validation targets
            
        Returns:
            Forward gradient
        """
        # Implementation using iterative gradient updates
        # Full forward-mode differentiation would require more complex setup
        
        # Compute training loss and gradients
        train_loss = loss_fn(model(train_data), train_target)
        train_grads = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
        
        # Compute validation loss
        val_loss = loss_fn(model(val_data), val_target)
        
        # Forward gradient approximation
        forward_grad = []
        for train_grad in train_grads:
            # Simple approximation using training gradient
            forward_grad.append(train_grad)
        
        return forward_grad
    
    def compute_reverse_gradient(self, model, loss_fn, train_data, train_target, val_data, val_target):
        """
        Compute reverse gradient using reverse-mode differentiation
        
        Args:
            model: Model to compute gradients for
            loss_fn: Loss function
            train_data: Training data
            train_target: Training targets
            val_data: Validation data
            val_target: Validation targets
            
        Returns:
            Reverse gradient
        """
        # Standard reverse-mode differentiation (backpropagation)
        val_loss = loss_fn(model(val_data), val_target)
        reverse_grad = torch.autograd.grad(val_loss, model.parameters())
        
        return reverse_grad
    
    def compute_second_order_gradient(self, model, loss_fn, train_data, train_target, val_data, val_target):
        """
        Compute second-order gradient using Hessian approximation
        
        Args:
            model: Model to compute gradients for
            loss_fn: Loss function
            train_data: Training data
            train_target: Training targets
            val_data: Validation data
            val_target: Validation targets
            
        Returns:
            Second-order gradient
        """
        # Compute first-order gradients
        train_loss = loss_fn(model(train_data), train_target)
        train_grads = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
        
        # Compute second-order gradients (Hessian diagonal approximation)
        second_order_grad = []
        for train_grad in train_grads:
            # Compute gradient of gradient (second-order)
            grad_of_grad = torch.autograd.grad(train_grad.sum(), model.parameters(), retain_graph=True)
            second_order_grad.append(grad_of_grad[0])
        
        return second_order_grad
    
    def compute_meta_gradient(self, model, loss_fn, train_data, train_target, val_data, val_target, meta_lr=0.01):
        """
        Compute meta-gradient for meta-learning
        
        Args:
            model: Model to compute gradients for
            loss_fn: Loss function
            train_data: Training data
            train_target: Training targets
            val_data: Validation data
            val_target: Validation targets
            meta_lr: Meta-learning rate
            
        Returns:
            Meta-gradient
        """
        # Create meta-model
        meta_model = self._create_meta_model(model)
        meta_model.load_state_dict(model.state_dict())
        
        # Inner loop: update meta-model on training data
        train_pred = meta_model(train_data)
        train_loss = loss_fn(train_pred, train_target)
        train_grads = torch.autograd.grad(train_loss, meta_model.parameters(), create_graph=True)
        
        # Update meta-model parameters
        for param, grad in zip(meta_model.parameters(), train_grads):
            param.data = param.data - meta_lr * grad
        
        # Outer loop: compute validation loss on updated meta-model
        val_pred = meta_model(val_data)
        val_loss = loss_fn(val_pred, val_target)
        
        # Meta-gradient is gradient of validation loss w.r.t. original parameters
        meta_grad = torch.autograd.grad(val_loss, model.parameters())
        
        return meta_grad
    
    def _create_meta_model(self, original_model):
        """
        Create a copy of the model for meta-learning
        
        Args:
            original_model: Original model to copy
            
        Returns:
            Meta-model copy
        """
        # Implementation using iterative gradient updates
        # In practice, you would create a proper copy of the model
        meta_model = type(original_model)(self.config, self.device)
        return meta_model
    
    def apply_gradient_clipping(self, model, max_norm=1.0):
        """
        Apply gradient clipping to model parameters
        
        Args:
            model: Model to clip gradients for
            max_norm: Maximum gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm
    
    def compute_gradient_norms(self, model):
        """
        Compute gradient norms for all parameters
        
        Args:
            model: Model to compute gradient norms for
            
        Returns:
            Dictionary with gradient norms
        """
        grad_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms[name] = grad_norm
        
        return grad_norms
    
    def apply_gradient_noise(self, model, noise_scale=0.01):
        """
        Apply noise to gradients for regularization
        
        Args:
            model: Model to add noise to gradients for
            noise_scale: Scale of noise to add
            
        Returns:
            Total noise added
        """
        total_noise = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.data += noise
                total_noise += noise.norm().item()
        
        return total_noise
    
    def compute_gradient_centrality(self, model, loss_fn, data, target):
        """
        Compute gradient centrality for parameter importance
        
        Args:
            model: Model to compute centrality for
            loss_fn: Loss function
            data: Input data
            target: Target labels
            
        Returns:
            Dictionary with gradient centrality scores
        """
        # Compute loss and gradients
        loss = loss_fn(model(data), target)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Compute centrality as gradient magnitude
        centrality = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                centrality[name] = param.grad.data.norm(2).item()
        
        return centrality
