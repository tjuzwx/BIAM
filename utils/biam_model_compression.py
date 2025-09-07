"""
BIAM Model Compression
Model compression techniques for BIAM model
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import copy

class BIAMModelCompression:
    """
    Model compression utilities for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize model compression
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prune_model(self, model: nn.Module, pruning_ratio: float = 0.2, 
                   pruning_type: str = 'magnitude'):
        """
        Prune model parameters
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of parameters to prune
            pruning_type: Type of pruning ('magnitude', 'random', 'gradient')
            
        Returns:
            Pruned model
        """
        model_copy = copy.deepcopy(model)
        
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        if pruning_type == 'magnitude':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
        elif pruning_type == 'random':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=pruning_ratio,
            )
        elif pruning_type == 'gradient':
            # This would require gradient information
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model_copy
    
    def quantize_model(self, model: nn.Module, quantization_type: str = 'dynamic'):
        """
        Quantize model for reduced precision
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
            
        Returns:
            Quantized model
        """
        model_copy = copy.deepcopy(model)
        
        if quantization_type == 'dynamic':
            # Dynamic quantization
            model_copy = torch.quantization.quantize_dynamic(
                model_copy, {nn.Linear}, dtype=torch.qint8
            )
        elif quantization_type == 'static':
            # Static quantization (requires calibration)
            model_copy.eval()
            model_copy = torch.quantization.quantize(
                model_copy, run_fn=self._calibrate_model, run_args=[model_copy]
            )
        elif quantization_type == 'qat':
            # Quantization aware training
            model_copy.train()
            model_copy.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            model_copy = torch.quantization.prepare_qat(model_copy)
        
        return model_copy
    
    def _calibrate_model(self, model, test_input):
        """
        Calibration function for static quantization
        
        Args:
            model: Model to calibrate
            test_input: Test input for calibration
        """
        model.eval()
        with torch.no_grad():
            _ = model(test_input)
    
    def compress_model_weights(self, model: nn.Module, compression_ratio: float = 0.5):
        """
        Compress model weights using SVD decomposition
        
        Args:
            model: Model to compress
            compression_ratio: Compression ratio
            
        Returns:
            Compressed model
        """
        model_copy = copy.deepcopy(model)
        
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear):
                # Apply SVD compression to linear layers
                weight = module.weight.data
                U, S, V = torch.svd(weight)
                
                # Keep only top k singular values
                k = int(weight.size(0) * compression_ratio)
                U_compressed = U[:, :k]
                S_compressed = S[:k]
                V_compressed = V[:, :k]
                
                # Reconstruct compressed weight
                compressed_weight = U_compressed @ torch.diag(S_compressed) @ V_compressed.T
                module.weight.data = compressed_weight
        
        return model_copy
    
    def knowledge_distillation(self, teacher_model: nn.Module, student_model: nn.Module,
                             train_loader, num_epochs: int = 10, temperature: float = 3.0,
                             alpha: float = 0.7):
        """
        Implement knowledge distillation
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            train_loader: Training data loader
            num_epochs: Number of distillation epochs
            temperature: Temperature for softmax
            alpha: Weight for distillation loss
            
        Returns:
            Distilled student model
        """
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                    teacher_soft = torch.softmax(teacher_output / temperature, dim=1)
                
                # Get student predictions
                student_output = student_model(data)
                student_soft = torch.log_softmax(student_output / temperature, dim=1)
                
                # Calculate distillation loss
                distillation_loss = criterion(student_soft, teacher_soft)
                
                # Calculate student loss
                student_loss = nn.CrossEntropyLoss()(student_output, target.long())
                
                # Combined loss
                loss = alpha * distillation_loss + (1 - alpha) * student_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Distillation Epoch {epoch}, Loss: {total_loss / len(train_loader):.6f}')
        
        return student_model
    
    def create_lightweight_model(self, original_model: nn.Module, compression_factor: float = 0.5):
        """
        Create lightweight version of the model
        
        Args:
            original_model: Original model
            compression_factor: Compression factor
            
        Returns:
            Lightweight model
        """
        # Lightweight model implementation
        # In practice, you would create a more sophisticated lightweight architecture
        
        class LightweightBIAM(nn.Module):
            def __init__(self, original_model, compression_factor):
                super().__init__()
                self.compression_factor = compression_factor
                
                # Extract dimensions from original model
                if hasattr(original_model, 'additive_model'):
                    original_dim = original_model.additive_model.predict.in_features
                    compressed_dim = int(original_dim * compression_factor)
                    
                    # Create compressed layers
                    self.compressed_linear = nn.Linear(original_dim, compressed_dim)
                    self.output_layer = nn.Linear(compressed_dim, original_model.additive_model.predict.out_features)
                    
                    # Copy weighting network
                    self.weighting_network = original_model.weighting_network
                
            def forward(self, x):
                # Compress input
                x_compressed = self.compressed_linear(x)
                # Output
                output = self.output_layer(x_compressed)
                return output
        
        lightweight_model = LightweightBIAM(original_model, compression_factor)
        return lightweight_model
    
    def analyze_model_complexity(self, model: nn.Module):
        """
        Analyze model complexity
        
        Args:
            model: Model to analyze
            
        Returns:
            Complexity analysis dictionary
        """
        total_params = 0
        trainable_params = 0
        layer_info = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                total_params += params
                trainable_params += trainable
                
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': params,
                    'trainable': trainable
                })
        
        # Calculate model size in MB
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        analysis = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'layer_info': layer_info,
            'compression_ratio': 1.0  # Will be updated after compression
        }
        
        return analysis
    
    def compare_models(self, original_model: nn.Module, compressed_model: nn.Module,
                      test_data, test_targets):
        """
        Compare original and compressed models
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            test_data: Test data
            test_targets: Test targets
            
        Returns:
            Comparison results
        """
        original_model.eval()
        compressed_model.eval()
        
        # Analyze complexity
        original_analysis = self.analyze_model_complexity(original_model)
        compressed_analysis = self.analyze_model_complexity(compressed_model)
        
        # Test performance
        with torch.no_grad():
            test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(self.device)
            
            # Original model predictions
            original_pred = original_model(test_data_tensor)
            
            # Compressed model predictions
            compressed_pred = compressed_model(test_data_tensor)
            
            # Calculate accuracy (for classification)
            if self.config.task == 'classification':
                original_acc = (torch.argmax(original_pred, dim=1) == torch.tensor(test_targets).to(self.device)).float().mean()
                compressed_acc = (torch.argmax(compressed_pred, dim=1) == torch.tensor(test_targets).to(self.device)).float().mean()
            else:
                # For regression, calculate MSE
                original_mse = nn.MSELoss()(original_pred, torch.tensor(test_targets, dtype=torch.float32).to(self.device))
                compressed_mse = nn.MSELoss()(compressed_pred, torch.tensor(test_targets, dtype=torch.float32).to(self.device))
        
        # Calculate compression ratio
        compression_ratio = compressed_analysis['total_parameters'] / original_analysis['total_parameters']
        
        comparison = {
            'original_parameters': original_analysis['total_parameters'],
            'compressed_parameters': compressed_analysis['total_parameters'],
            'compression_ratio': compression_ratio,
            'size_reduction': 1 - compression_ratio,
            'original_size_mb': original_analysis['model_size_mb'],
            'compressed_size_mb': compressed_analysis['model_size_mb']
        }
        
        if self.config.task == 'classification':
            comparison['original_accuracy'] = original_acc.item()
            comparison['compressed_accuracy'] = compressed_acc.item()
            comparison['accuracy_drop'] = original_acc.item() - compressed_acc.item()
        else:
            comparison['original_mse'] = original_mse.item()
            comparison['compressed_mse'] = compressed_mse.item()
            comparison['mse_increase'] = compressed_mse.item() - original_mse.item()
        
        return comparison
    
    def export_compressed_model(self, model: nn.Module, export_path: str, 
                              export_format: str = 'torchscript'):
        """
        Export compressed model
        
        Args:
            model: Model to export
            export_path: Path to save exported model
            export_format: Export format ('torchscript', 'onnx')
            
        Returns:
            Path to exported model
        """
        model.eval()
        
        if export_format == 'torchscript':
            # Export as TorchScript
            test_input = torch.randn(1, 100).to(self.device)  # Adjust input size
            traced_model = torch.jit.trace(model, test_input)
            traced_model.save(export_path)
        
        elif export_format == 'onnx':
            # Export as ONNX
            test_input = torch.randn(1, 100).to(self.device)  # Adjust input size
            torch.onnx.export(
                model, test_input, export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        
        return export_path
