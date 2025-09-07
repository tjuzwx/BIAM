"""
BIAM Batch Optimizer
Batch processing efficiency optimization utilities
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

class BIAMBatchOptimizer:
    """
    Batch processing optimization utilities for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize batch optimizer
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_optimized_dataloader(self, dataset, batch_size: int = None, 
                                  num_workers: int = None, pin_memory: bool = True,
                                  prefetch_factor: int = 2):
        """
        Create optimized data loader with best practices
        
        Args:
            dataset: Dataset to create loader for
            batch_size: Batch size (uses config default if None)
            num_workers: Number of worker processes (uses config default if None)
            pin_memory: Whether to pin memory
            prefetch_factor: Prefetch factor for data loading
            
        Returns:
            Optimized DataLoader
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        if num_workers is None:
            num_workers = getattr(self.config, 'num_workers', 4)
        
        # Optimize num_workers based on system
        if num_workers == 0:
            num_workers = min(4, torch.get_num_threads())
        
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
            drop_last=True  # For consistent batch sizes
        )
        
        return dataloader
    
    def optimize_batch_processing(self, model: nn.Module, dataloader: Data.DataLoader,
                                optimizer: torch.optim.Optimizer, num_epochs: int = 1):
        """
        Optimize batch processing with various techniques
        
        Args:
            model: Model to train
            dataloader: Data loader
            optimizer: Optimizer
            num_epochs: Number of epochs to run
            
        Returns:
            Training statistics
        """
        model.train()
        training_stats = {
            'epoch_times': [],
            'batch_times': [],
            'throughput': [],
            'memory_usage': []
        }
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_batch_times = []
            epoch_throughput = []
            
            for batch_idx, (data, target) in enumerate(dataloader):
                batch_start_time = time.time()
                
                # Move data to device efficiently
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = nn.MSELoss()(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate batch time and throughput
                batch_time = time.time() - batch_start_time
                throughput = data.size(0) / batch_time  # samples per second
                
                epoch_batch_times.append(batch_time)
                epoch_throughput.append(throughput)
                
                # Log progress
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.6f}, '
                          f'Time: {batch_time:.4f}s, '
                          f'Throughput: {throughput:.2f} samples/s')
            
            epoch_time = time.time() - epoch_start_time
            training_stats['epoch_times'].append(epoch_time)
            training_stats['batch_times'].extend(epoch_batch_times)
            training_stats['throughput'].extend(epoch_throughput)
            
            print(f'Epoch {epoch} completed in {epoch_time:.2f}s, '
                  f'Avg throughput: {np.mean(epoch_throughput):.2f} samples/s')
        
        return training_stats
    
    def implement_gradient_accumulation(self, model: nn.Module, dataloader: Data.DataLoader,
                                      optimizer: torch.optim.Optimizer, accumulation_steps: int = 4):
        """
        Implement gradient accumulation for larger effective batch sizes
        
        Args:
            model: Model to train
            dataloader: Data loader
            optimizer: Optimizer
            accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Training statistics
        """
        model.train()
        training_stats = {
            'losses': [],
            'gradient_norms': [],
            'effective_batch_size': []
        }
        
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Forward pass
            output = model(data)
            loss = nn.MSELoss()(output, target)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Calculate gradient norm before clipping
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                
                # Store statistics
                training_stats['losses'].append(loss.item() * accumulation_steps)
                training_stats['gradient_norms'].append(total_norm)
                training_stats['effective_batch_size'].append(data.size(0) * accumulation_steps)
                
                print(f'Accumulated batch {batch_idx // accumulation_steps}, '
                      f'Loss: {loss.item() * accumulation_steps:.6f}, '
                      f'Grad norm: {total_norm:.6f}, '
                      f'Effective batch size: {data.size(0) * accumulation_steps}')
        
        return training_stats
    
    def implement_mixed_precision_training(self, model: nn.Module, dataloader: Data.DataLoader,
                                         optimizer: torch.optim.Optimizer):
        """
        Implement mixed precision training for efficiency
        
        Args:
            model: Model to train
            dataloader: Data loader
            optimizer: Optimizer
            
        Returns:
            Training statistics
        """
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        
        training_stats = {
            'losses': [],
            'scaled_losses': [],
            'inf_count': 0
        }
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = nn.MSELoss()(output, target)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Store statistics
            training_stats['losses'].append(loss.item())
            training_stats['scaled_losses'].append(scaler.get_scale())
            
            # Check for inf/nan
            if torch.isinf(loss) or torch.isnan(loss):
                training_stats['inf_count'] += 1
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.6f}, '
                      f'Scale: {scaler.get_scale():.2f}')
        
        return training_stats
    
    def profile_data_loading(self, dataloader: Data.DataLoader, num_batches: int = 100):
        """
        Profile data loading performance
        
        Args:
            dataloader: Data loader to profile
            num_batches: Number of batches to profile
            
        Returns:
            Data loading statistics
        """
        loading_times = []
        batch_sizes = []
        
        start_time = time.time()
        
        for i, (data, target) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Simulate processing time
            _ = data.size()
            _ = target.size()
            
            batch_time = time.time() - batch_start
            loading_times.append(batch_time)
            batch_sizes.append(data.size(0))
        
        total_time = time.time() - start_time
        
        stats = {
            'total_time': total_time,
            'avg_loading_time': np.mean(loading_times),
            'std_loading_time': np.std(loading_times),
            'min_loading_time': np.min(loading_times),
            'max_loading_time': np.max(loading_times),
            'avg_batch_size': np.mean(batch_sizes),
            'throughput': sum(batch_sizes) / total_time
        }
        
        print(f"Data Loading Profile:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg loading time: {stats['avg_loading_time']:.4f}s")
        print(f"  Throughput: {stats['throughput']:.2f} samples/s")
        
        return stats
    
    def optimize_data_preprocessing(self, dataset, preprocessing_fn=None):
        """
        Optimize data preprocessing pipeline
        
        Args:
            dataset: Dataset to optimize
            preprocessing_fn: Preprocessing function to apply
            
        Returns:
            Optimized dataset
        """
        if preprocessing_fn is None:
            # Default preprocessing: normalization
            def default_preprocessing(data):
                return (data - data.mean()) / (data.std() + 1e-8)
            preprocessing_fn = default_preprocessing
        
        # Apply preprocessing
        if hasattr(dataset, 'data'):
            dataset.data = preprocessing_fn(dataset.data)
        
        return dataset
    
    def implement_async_data_loading(self, dataset, batch_size: int, num_workers: int = 4):
        """
        Implement asynchronous data loading
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Asynchronous data loader
        """
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        )
        
        return dataloader
