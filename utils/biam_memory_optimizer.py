"""
BIAM Memory Optimizer
GPU memory optimization utilities for BIAM model
"""

import torch
import torch.nn as nn
import gc
from typing import Dict, Any, Optional
import psutil
import os

class BIAMMemoryOptimizer:
    """
    Memory optimization utilities for BIAM model
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize memory optimizer
        
        Args:
            device: Device to optimize memory for
        """
        self.device = device
        self.memory_stats = {}
    
    def clear_cache(self):
        """
        Clear GPU cache and run garbage collection
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics
        
        Returns:
            Dictionary with memory usage statistics
        """
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            stats['gpu_cached'] = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
        
        # CPU memory
        process = psutil.Process(os.getpid())
        stats['cpu_memory'] = process.memory_info().rss / 1024**3  # GB
        stats['cpu_percent'] = process.memory_percent()
        
        return stats
    
    def log_memory_usage(self, stage: str = ""):
        """
        Log current memory usage
        
        Args:
            stage: Stage description for logging
        """
        stats = self.get_memory_usage()
        print(f"Memory Usage {stage}:")
        for key, value in stats.items():
            if 'percent' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.3f} GB")
    
    def optimize_model_memory(self, model: nn.Module):
        """
        Optimize model memory usage
        
        Args:
            model: Model to optimize
        """
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Move model to device
        model.to(self.device)
        
        # Clear cache after moving
        self.clear_cache()
    
    def enable_mixed_precision(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Enable mixed precision training for memory efficiency
        
        Args:
            model: Model to enable mixed precision for
            optimizer: Optimizer to use with mixed precision
            
        Returns:
            GradScaler for mixed precision
        """
        scaler = torch.cuda.amp.GradScaler()
        return scaler
    
    def profile_memory_usage(self, model: nn.Module, input_shape: tuple, num_iterations: int = 10):
        """
        Profile memory usage during forward pass
        
        Args:
            model: Model to profile
            input_shape: Input tensor shape
            num_iterations: Number of iterations to profile
            
        Returns:
            Memory usage statistics
        """
        model.eval()
        memory_stats = []
        
        with torch.no_grad():
            for i in range(num_iterations):
                # Clear cache before each iteration
                self.clear_cache()
                
                # Create test input
                test_input = torch.randn(input_shape).to(self.device)
                
                # Record memory before forward pass
                memory_before = self.get_memory_usage()
                
                # Forward pass
                _ = model(test_input)
                
                # Record memory after forward pass
                memory_after = self.get_memory_usage()
                
                # Store difference
                memory_diff = {}
                for key in memory_before:
                    if key in memory_after:
                        memory_diff[key] = memory_after[key] - memory_before[key]
                
                memory_stats.append(memory_diff)
                
                # Clean up
                del test_input
                self.clear_cache()
        
        return memory_stats
    
    def optimize_batch_size(self, model: nn.Module, input_shape: tuple, max_memory_gb: float = 8.0):
        """
        Find optimal batch size for given memory constraints
        
        Args:
            model: Model to optimize batch size for
            input_shape: Input tensor shape (without batch dimension)
            max_memory_gb: Maximum memory usage in GB
            
        Returns:
            Optimal batch size
        """
        model.eval()
        optimal_batch_size = 1
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                # Clear cache
                self.clear_cache()
                
                # Create batch input
                batch_input = torch.randn((batch_size,) + input_shape).to(self.device)
                
                # Record memory before
                memory_before = self.get_memory_usage()
                
                # Forward pass
                with torch.no_grad():
                    _ = model(batch_input)
                
                # Record memory after
                memory_after = self.get_memory_usage()
                
                # Check memory usage
                memory_used = memory_after.get('gpu_allocated', 0) - memory_before.get('gpu_allocated', 0)
                
                if memory_used <= max_memory_gb:
                    optimal_batch_size = batch_size
                else:
                    break
                
                # Clean up
                del batch_input
                self.clear_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        return optimal_batch_size
    
    def monitor_training_memory(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                              train_loader, num_batches: int = 10):
        """
        Monitor memory usage during training
        
        Args:
            model: Model being trained
            optimizer: Optimizer
            train_loader: Training data loader
            num_batches: Number of batches to monitor
            
        Returns:
            Memory usage statistics during training
        """
        model.train()
        memory_stats = []
        
        for i, (data, target) in enumerate(train_loader):
            if i >= num_batches:
                break
            
            # Clear cache
            self.clear_cache()
            
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Record memory before forward pass
            memory_before = self.get_memory_usage()
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            
            # Record memory after forward pass
            memory_after_forward = self.get_memory_usage()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record memory after backward pass
            memory_after_backward = self.get_memory_usage()
            
            # Store statistics
            batch_stats = {
                'batch': i,
                'memory_before': memory_before,
                'memory_after_forward': memory_after_forward,
                'memory_after_backward': memory_after_backward,
                'forward_memory_increase': memory_after_forward.get('gpu_allocated', 0) - memory_before.get('gpu_allocated', 0),
                'backward_memory_increase': memory_after_backward.get('gpu_allocated', 0) - memory_after_forward.get('gpu_allocated', 0)
            }
            
            memory_stats.append(batch_stats)
            
            # Clean up
            del data, target, output, loss
            self.clear_cache()
        
        return memory_stats
