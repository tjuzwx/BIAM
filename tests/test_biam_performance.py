"""
Performance benchmark tests for BIAM model
"""

import unittest
import torch
import numpy as np
import time
import sys
import os
import pytest
import psutil
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from models.biam_additive_model import BIAMAdditiveModel
from data.biam_data_generator import BIAMDataGenerator
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_config import BIAMConfig
from utils.biam_memory_optimizer import BIAMMemoryOptimizer
from utils.biam_batch_optimizer import BIAMBatchOptimizer

class TestBIAMPerformance(unittest.TestCase):
    """
    Performance benchmark tests for BIAM model
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.config = BIAMConfig()
        self.config.task = 'classification'
        self.config.input_dim = 100
        self.config.num_classes = 2
        self.config.batch_size = 256
        self.config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        self.n_samples = 10000
        self.n_features = 100
        self.X = torch.randn(self.n_samples, self.n_features).to(self.config.device)
        self.y = torch.randint(0, 2, (self.n_samples,)).to(self.config.device)
        
        # Create data loader
        self.data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X, self.y),
            batch_size=self.config.batch_size,
            shuffle=True
        )
    
    def test_model_initialization_performance(self):
        """
        Test model initialization performance
        """
        start_time = time.time()
        
        # Initialize model
        biam_model = BIAMModel(self.config, self.config.device)
        
        init_time = time.time() - start_time
        
        # Should initialize quickly (less than 1 second)
        self.assertLess(init_time, 1.0)
        print(f"Model initialization time: {init_time:.4f}s")
    
    def test_forward_pass_performance(self):
        """
        Test forward pass performance
        """
        biam_model = BIAMModel(self.config, self.config.device)
        biam_model.eval()
        
        # Warm up
        with torch.no_grad():
            _ = biam_model(self.X[:100])
        
        # Benchmark forward pass
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = biam_model(self.X[:100])
        
        forward_time = (time.time() - start_time) / 10
        
        # Should be fast (less than 0.1s per batch)
        self.assertLess(forward_time, 0.1)
        print(f"Forward pass time per batch: {forward_time:.4f}s")
    
    def test_training_performance(self):
        """
        Test training performance
        """
        biam_model = BIAMModel(self.config, self.config.device)
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        optimizer = BIAMOptimizer(self.config, biam_model, weighting_network)
        
        # Benchmark training step
        start_time = time.time()
        
        for i, (data, target) in enumerate(self.data_loader):
            if i >= 5:  # Test 5 batches
                break
            
            # Create validation loader
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(data[:32], target[:32]),
                batch_size=32
            )
            
            train_metrics = optimizer.train_epoch(self.data_loader, val_loader, i)
        
        training_time = time.time() - start_time
        
        # Should train reasonably fast (less than 10s for 5 batches)
        self.assertLess(training_time, 10.0)
        print(f"Training time for 5 batches: {training_time:.4f}s")
    
    def test_memory_usage(self):
        """
        Test memory usage
        """
        memory_optimizer = BIAMMemoryOptimizer(self.config.device)
        
        # Get initial memory
        initial_memory = memory_optimizer.get_memory_usage()
        
        # Create model
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Get memory after model creation
        model_memory = memory_optimizer.get_memory_usage()
        
        # Calculate memory increase
        memory_increase = model_memory.get('gpu_allocated', 0) - initial_memory.get('gpu_allocated', 0)
        
        # Should use reasonable amount of memory (less than 1GB)
        self.assertLess(memory_increase, 1.0)
        print(f"Memory increase: {memory_increase:.3f} GB")
    
    def test_batch_processing_performance(self):
        """
        Test batch processing performance
        """
        batch_optimizer = BIAMBatchOptimizer(self.config)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        
        # Benchmark data loading
        start_time = time.time()
        
        dataloader = batch_optimizer.create_optimized_dataloader(
            dataset, batch_size=self.config.batch_size, num_workers=4
        )
        
        # Test data loading speed
        for i, (data, target) in enumerate(dataloader):
            if i >= 10:  # Test 10 batches
                break
        
        loading_time = time.time() - start_time
        
        # Should load data quickly (less than 5s for 10 batches)
        self.assertLess(loading_time, 5.0)
        print(f"Data loading time for 10 batches: {loading_time:.4f}s")
    
    def test_gradient_computation_performance(self):
        """
        Test gradient computation performance
        """
        biam_model = BIAMModel(self.config, self.config.device)
        biam_model.train()
        
        # Benchmark gradient computation
        start_time = time.time()
        
        for i, (data, target) in enumerate(self.data_loader):
            if i >= 5:  # Test 5 batches
                break
            
            # Forward pass
            output = biam_model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            
            # Backward pass
            loss.backward()
            
            # Clear gradients
            biam_model.zero_grad()
        
        gradient_time = time.time() - start_time
        
        # Should compute gradients reasonably fast (less than 5s for 5 batches)
        self.assertLess(gradient_time, 5.0)
        print(f"Gradient computation time for 5 batches: {gradient_time:.4f}s")
    
    def test_model_compression_performance(self):
        """
        Test model compression performance
        """
        from utils.biam_model_compression import BIAMModelCompression
        
        compression = BIAMModelCompression(self.config)
        
        # Create model
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Benchmark compression
        start_time = time.time()
        
        compressed_model = compression.prune_model(biam_model, pruning_ratio=0.2)
        
        compression_time = time.time() - start_time
        
        # Should compress quickly (less than 2s)
        self.assertLess(compression_time, 2.0)
        print(f"Model compression time: {compression_time:.4f}s")
        
        # Test compressed model performance
        compressed_model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            _ = compressed_model(self.X[:100])
        
        compressed_forward_time = time.time() - start_time
        
        # Compressed model should be faster
        print(f"Compressed model forward pass time: {compressed_forward_time:.4f}s")
    
    def test_parallel_processing_performance(self):
        """
        Test parallel processing performance
        """
        from utils.biam_parallel_optimizer import BIAMParallelOptimizer
        
        parallel_optimizer = BIAMParallelOptimizer(self.config)
        
        # Create model
        biam_model = BIAMModel(self.config, self.config.device)
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        
        # Wrap model for parallel processing
        start_time = time.time()
        
        parallel_model = parallel_optimizer.wrap_model_for_parallel(biam_model)
        
        parallel_time = time.time() - start_time
        
        # Should wrap quickly (less than 1s)
        self.assertLess(parallel_time, 1.0)
        print(f"Parallel model wrapping time: {parallel_time:.4f}s")
    
    def test_data_generation_performance(self):
        """
        Test data generation performance
        """
        generator = BIAMDataGenerator(self.config)
        
        # Benchmark data generation
        start_time = time.time()
        
        train_loader, val_loader, test_data = generator.generate_data()
        
        generation_time = time.time() - start_time
        
        # Should generate data quickly (less than 5s)
        self.assertLess(generation_time, 5.0)
        print(f"Data generation time: {generation_time:.4f}s")
    
    def test_visualization_performance(self):
        """
        Test visualization performance
        """
        from visualization.biam_visualizer import BIAMVisualizer
        
        visualizer = BIAMVisualizer(self.config)
        
        # Create training history
        training_history = {
            'epochs': list(range(100)),
            'train_loss': np.random.rand(100),
            'val_loss': np.random.rand(100),
            'test_metrics': [{'accuracy': np.random.rand()} for _ in range(100)]
        }
        
        # Benchmark visualization
        start_time = time.time()
        
        save_path = visualizer.plot_training_curves(training_history)
        
        viz_time = time.time() - start_time
        
        # Should create visualization quickly (less than 2s)
        self.assertLess(viz_time, 2.0)
        print(f"Visualization creation time: {viz_time:.4f}s")
        
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)
    
    def test_end_to_end_performance(self):
        """
        Test end-to-end performance
        """
        # Create data generator
        generator = BIAMDataGenerator(self.config)
        train_loader, val_loader, test_data = generator.generate_data()
        
        # Create model
        biam_model = BIAMModel(self.config, self.config.device)
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        optimizer = BIAMOptimizer(self.config, biam_model, weighting_network)
        
        # Benchmark end-to-end training
        start_time = time.time()
        
        for epoch in range(5):  # Test 5 epochs
            train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
            
            if epoch % 2 == 0:
                test_metrics = optimizer.evaluate(test_data)
        
        end_to_end_time = time.time() - start_time
        
        # Should complete end-to-end training reasonably fast (less than 30s)
        self.assertLess(end_to_end_time, 30.0)
        print(f"End-to-end training time for 5 epochs: {end_to_end_time:.4f}s")
    
    def test_scalability(self):
        """
        Test model scalability with different data sizes
        """
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            # Create data of different sizes
            X = torch.randn(size, self.n_features).to(self.config.device)
            y = torch.randint(0, 2, (size,)).to(self.config.device)
            
            # Create model
            biam_model = BIAMModel(self.config, self.config.device)
            biam_model.eval()
            
            # Benchmark forward pass
            start_time = time.time()
            
            with torch.no_grad():
                _ = biam_model(X)
            
            forward_time = time.time() - start_time
            times.append(forward_time)
            
            print(f"Size {size}: {forward_time:.4f}s")
        
        # Check that time scales reasonably with data size
        for i in range(1, len(times)):
            # Time should not increase more than linearly
            time_ratio = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            
            # Time ratio should be less than or equal to size ratio
            self.assertLessEqual(time_ratio, size_ratio * 1.5)  # Allow 50% overhead
    
    def test_memory_efficiency(self):
        """
        Test memory efficiency
        """
        memory_optimizer = BIAMMemoryOptimizer(self.config.device)
        
        # Test memory usage with different batch sizes
        batch_sizes = [32, 64, 128, 256]
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Clear cache
            memory_optimizer.clear_cache()
            
            # Get initial memory
            initial_memory = memory_optimizer.get_memory_usage()
            
            # Create data
            X = torch.randn(batch_size, self.n_features).to(self.config.device)
            y = torch.randint(0, 2, (batch_size,)).to(self.config.device)
            
            # Create model
            biam_model = BIAMModel(self.config, self.config.device)
            
            # Forward pass
            with torch.no_grad():
                _ = biam_model(X)
            
            # Get memory after forward pass
            final_memory = memory_optimizer.get_memory_usage()
            
            # Calculate memory usage
            memory_used = final_memory.get('gpu_allocated', 0) - initial_memory.get('gpu_allocated', 0)
            memory_usage.append(memory_used)
            
            print(f"Batch size {batch_size}: {memory_used:.3f} GB")
            
            # Clean up
            del X, y, biam_model
            memory_optimizer.clear_cache()
        
        # Check that memory usage scales reasonably with batch size
        for i in range(1, len(memory_usage)):
            memory_ratio = memory_usage[i] / memory_usage[i-1]
            batch_ratio = batch_sizes[i] / batch_sizes[i-1]
            
            # Memory ratio should be less than or equal to batch ratio
            self.assertLessEqual(memory_ratio, batch_ratio * 1.2)  # Allow 20% overhead

if __name__ == '__main__':
    unittest.main()
