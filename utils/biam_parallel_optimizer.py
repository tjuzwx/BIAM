"""
BIAM Parallel Optimizer
Parallel computing support for BIAM model
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import os
import time

class BIAMParallelOptimizer:
    """
    Parallel computing optimization utilities for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize parallel optimizer
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.world_size = 1
        self.rank = 0
        self.is_distributed = False
    
    def setup_distributed_training(self, rank: int = 0, world_size: int = 1, 
                                 backend: str = 'nccl'):
        """
        Setup distributed training environment
        
        Args:
            rank: Process rank
            world_size: Number of processes
            backend: Communication backend
        """
        self.rank = rank
        self.world_size = world_size
        
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
            self.is_distributed = True
            
            # Set device for this process
            torch.cuda.set_device(rank)
            self.device = torch.device(f'cuda:{rank}')
    
    def cleanup_distributed(self):
        """
        Cleanup distributed training environment
        """
        if self.is_distributed:
            dist.destroy_process_group()
    
    def wrap_model_for_parallel(self, model: nn.Module, use_ddp: bool = True):
        """
        Wrap model for parallel training
        
        Args:
            model: Model to wrap
            use_ddp: Whether to use DistributedDataParallel
            
        Returns:
            Wrapped model
        """
        if use_ddp and self.is_distributed:
            # Use DistributedDataParallel for multi-GPU training
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.rank])
        elif torch.cuda.device_count() > 1:
            # Use DataParallel for single-node multi-GPU training
            model = model.to(self.device)
            model = DP(model)
        else:
            # Single GPU training
            model = model.to(self.device)
        
        return model
    
    def create_distributed_dataloader(self, dataset, batch_size: int, 
                                    num_workers: int = 4, shuffle: bool = True):
        """
        Create distributed data loader
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size per process
            num_workers: Number of worker processes
            shuffle: Whether to shuffle data
            
        Returns:
            Distributed data loader
        """
        if self.is_distributed:
            # Create distributed sampler
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
        
        return dataloader
    
    def distributed_training_step(self, model: nn.Module, dataloader, 
                                optimizer: torch.optim.Optimizer, 
                                loss_fn: Callable, epoch: int):
        """
        Perform distributed training step
        
        Args:
            model: Model to train
            dataloader: Data loader
            optimizer: Optimizer
            loss_fn: Loss function
            epoch: Current epoch
            
        Returns:
            Training statistics
        """
        model.train()
        
        if self.is_distributed:
            dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient synchronization for distributed training
            if self.is_distributed:
                self._sync_gradients(model)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0 and self.rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Average loss across all processes
        if self.is_distributed:
            total_loss = self._reduce_tensor(torch.tensor(total_loss).to(self.device))
            num_batches = self._reduce_tensor(torch.tensor(num_batches).to(self.device))
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def _sync_gradients(self, model: nn.Module):
        """
        Synchronize gradients across processes
        
        Args:
            model: Model with gradients to synchronize
        """
        if self.is_distributed:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
    
    def _reduce_tensor(self, tensor: torch.Tensor):
        """
        Reduce tensor across all processes
        
        Args:
            tensor: Tensor to reduce
            
        Returns:
            Reduced tensor
        """
        if self.is_distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        return tensor.item()
    
    def implement_model_parallelism(self, model: nn.Module, device_ids: List[int]):
        """
        Implement model parallelism across multiple devices
        
        Args:
            model: Model to parallelize
            device_ids: List of device IDs to use
            
        Returns:
            Parallelized model
        """
        if len(device_ids) > 1:
            # Split model across devices
            model = model.to(device_ids[0])
            
            # Example: Split layers across devices
            if hasattr(model, 'additive_model') and hasattr(model, 'weighting_network'):
                model.additive_model = model.additive_model.to(device_ids[0])
                model.weighting_network = model.weighting_network.to(device_ids[1])
        
        return model
    
    def implement_pipeline_parallelism(self, model: nn.Module, num_stages: int = 2):
        """
        Implement pipeline parallelism
        
        Args:
            model: Model to parallelize
            num_stages: Number of pipeline stages
            
        Returns:
            Pipeline parallel model
        """
        # Pipeline parallelism implementation
        # In practice, you would use torch.distributed.pipeline or similar
        
        if hasattr(model, 'additive_model') and hasattr(model, 'weighting_network'):
            # Split model into stages
            stage1 = model.additive_model
            stage2 = model.weighting_network
            
            # Create pipeline
            pipeline_model = nn.Sequential(stage1, stage2)
            
            return pipeline_model
        
        return model
    
    def profile_parallel_performance(self, model: nn.Module, dataloader, 
                                   optimizer: torch.optim.Optimizer, 
                                   num_batches: int = 100):
        """
        Profile parallel training performance
        
        Args:
            model: Model to profile
            dataloader: Data loader
            optimizer: Optimizer
            num_batches: Number of batches to profile
            
        Returns:
            Performance statistics
        """
        model.train()
        
        batch_times = []
        throughputs = []
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            batch_start = time.time()
            
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            if self.is_distributed:
                self._sync_gradients(model)
            
            optimizer.step()
            
            batch_time = time.time() - batch_start
            throughput = data.size(0) / batch_time
            
            batch_times.append(batch_time)
            throughputs.append(throughput)
        
        stats = {
            'avg_batch_time': np.mean(batch_times),
            'std_batch_time': np.std(batch_times),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'total_samples': sum(data.size(0) for _, (data, _) in enumerate(dataloader) if _ < num_batches)
        }
        
        if self.rank == 0:
            print(f"Parallel Performance Profile:")
            print(f"  Avg batch time: {stats['avg_batch_time']:.4f}s")
            print(f"  Avg throughput: {stats['avg_throughput']:.2f} samples/s")
            print(f"  Total samples processed: {stats['total_samples']}")
        
        return stats
    
    def implement_gradient_compression(self, model: nn.Module, compression_ratio: float = 0.1):
        """
        Implement gradient compression for communication efficiency
        
        Args:
            model: Model to compress gradients for
            compression_ratio: Compression ratio (0.1 = 10% of gradients)
            
        Returns:
            Model with gradient compression
        """
        # Pipeline parallelism implementation
        # In practice, you would use more sophisticated compression techniques
        
        original_backward = model.backward
        
        def compressed_backward(*args, **kwargs):
            # Perform backward pass
            original_backward(*args, **kwargs)
            
            # Compress gradients
            for param in model.parameters():
                if param.grad is not None:
                    # Simple top-k compression
                    grad_flat = param.grad.data.flatten()
                    k = int(len(grad_flat) * compression_ratio)
                    
                    if k > 0:
                        _, top_indices = torch.topk(torch.abs(grad_flat), k)
                        compressed_grad = torch.zeros_like(grad_flat)
                        compressed_grad[top_indices] = grad_flat[top_indices]
                        param.grad.data = compressed_grad.reshape(param.grad.data.shape)
        
        model.backward = compressed_backward
        return model
    
    def implement_async_optimization(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Implement asynchronous optimization
        
        Args:
            model: Model to optimize
            optimizer: Optimizer
            
        Returns:
            Asynchronous optimizer
        """
        # Pipeline parallelism implementation
        # In practice, you would use more sophisticated async techniques
        
        class AsyncOptimizer:
            def __init__(self, model, optimizer):
                self.model = model
                self.optimizer = optimizer
                self.gradient_queue = []
            
            def step_async(self, gradients):
                """Asynchronous optimization step"""
                self.gradient_queue.append(gradients)
                
                if len(self.gradient_queue) >= 2:  # Process when queue has 2 gradients
                    # Average gradients
                    avg_gradients = {}
                    for key in gradients.keys():
                        avg_gradients[key] = torch.stack([g[key] for g in self.gradient_queue]).mean(0)
                    
                    # Apply gradients
                    for name, param in self.model.named_parameters():
                        if name in avg_gradients:
                            param.grad = avg_gradients[name]
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.gradient_queue.clear()
        
        return AsyncOptimizer(model, optimizer)
