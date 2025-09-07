"""
BIAM Logger
Logging utilities for BIAM model training and evaluation
"""

import logging
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

class BIAMLogger:
    """
    Logger class for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize BIAM logger
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        
        # Create logs directory
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Training history
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'test_metrics': []
        }
    
    def _setup_logging(self):
        """
        Setup logging configuration
        """
        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(self.logs_dir, f'biam_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_filename = log_filename
    
    def info(self, message):
        """
        Log info message
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def warning(self, message):
        """
        Log warning message
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def error(self, message):
        """
        Log error message
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
    
    def log_metrics(self, epoch, train_metrics, test_metrics):
        """
        Log training and test metrics
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics dictionary
            test_metrics: Test metrics dictionary
        """
        # Store in history
        self.training_history['epochs'].append(epoch)
        self.training_history['train_loss'].append(train_metrics.get('loss', 0))
        self.training_history['val_loss'].append(train_metrics.get('val_loss', 0))
        self.training_history['test_metrics'].append(test_metrics)
        
        # Log to console and file
        log_message = f"Epoch {epoch:4d} | "
        log_message += f"Train Loss: {train_metrics.get('loss', 0):.6f} | "
        log_message += f"Val Loss: {train_metrics.get('val_loss', 0):.6f} | "
        
        if self.config.task == 'regression':
            log_message += f"Test RMSE: {test_metrics.get('rmse', 0):.6f} | "
            log_message += f"Test MAE: {test_metrics.get('mae', 0):.6f}"
        else:
            log_message += f"Test Acc: {test_metrics.get('accuracy', 0):.4f} | "
            log_message += f"Test F1: {test_metrics.get('f1_score', 0):.4f}"
        
        self.info(log_message)
    
    def log_config(self):
        """
        Log configuration parameters
        """
        self.info("BIAM Configuration:")
        config_dict = self.config.to_dict()
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")
    
    def log_model_info(self, model):
        """
        Log model information
        
        Args:
            model: BIAM model instance
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"Model Information:")
        self.info(f"  Total parameters: {total_params:,}")
        self.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Log model architecture
        self.info("Model Architecture:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                self.info(f"  {name}: {module}")
    
    def save_training_curves(self, save_path: Optional[str] = None):
        """
        Save training curves plot
        
        Args:
            save_path: Path to save plot (optional)
        """
        if not self.training_history['epochs']:
            return
        
        epochs = self.training_history['epochs']
        train_loss = self.training_history['train_loss']
        val_loss = self.training_history['val_loss']
        
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Training Loss', alpha=0.8)
        plt.plot(epochs, val_loss, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot test metrics
        if self.training_history['test_metrics']:
            plt.subplot(1, 2, 2)
            test_metrics = self.training_history['test_metrics']
            
            if self.config.task == 'regression':
                rmse_values = [m.get('rmse', 0) for m in test_metrics]
                plt.plot(epochs, rmse_values, label='Test RMSE', color='green')
                plt.ylabel('RMSE')
                plt.title('Test RMSE')
            else:
                acc_values = [m.get('accuracy', 0) for m in test_metrics]
                f1_values = [m.get('f1_score', 0) for m in test_metrics]
                plt.plot(epochs, acc_values, label='Test Accuracy', color='green')
                plt.plot(epochs, f1_values, label='Test F1 Score', color='orange')
                plt.ylabel('Score')
                plt.title('Test Metrics')
            
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.logs_dir, 'training_curves.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.info(f"Training curves saved to: {save_path}")
    
    def save_training_history(self, save_path: Optional[str] = None):
        """
        Save training history to JSON file
        
        Args:
            save_path: Path to save history (optional)
        """
        if save_path is None:
            save_path = os.path.join(self.logs_dir, 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_to_save = {}
        for key, value in self.training_history.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                # Handle test_metrics list
                history_to_save[key] = value
            else:
                # Convert numpy arrays to lists
                history_to_save[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        
        with open(save_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        
        self.info(f"Training history saved to: {save_path}")
    
    def log_feature_importance(self, feature_importance, feature_names=None):
        """
        Log feature importance
        
        Args:
            feature_importance: Feature importance scores
            feature_names: Names of features (optional)
        """
        self.info("Feature Importance:")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        for i, idx in enumerate(sorted_indices[:10]):  # Top 10 features
            self.info(f"  {i+1:2d}. {feature_names[idx]:20s}: {feature_importance[idx]:.6f}")
    
    def log_missing_value_analysis(self, missing_indicators, feature_names=None):
        """
        Log missing value analysis
        
        Args:
            missing_indicators: Missing value indicators
            feature_names: Names of features (optional)
        """
        self.info("Missing Value Analysis:")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(missing_indicators))]
        
        for i, (name, indicator) in enumerate(zip(feature_names, missing_indicators)):
            if abs(indicator) > 1e-6:  # Only log non-zero indicators
                self.info(f"  {name:20s}: {indicator:.6f}")
    
    def log_model_interpretation(self, interpretation):
        """
        Log model interpretation results
        
        Args:
            interpretation: Model interpretation dictionary
        """
        self.info("Model Interpretation:")
        
        if 'feature_contributions' in interpretation:
            contributions = interpretation['feature_contributions']
            sorted_indices = np.argsort(np.abs(contributions))[::-1]
            
            self.info("  Top Feature Contributions:")
            for i, idx in enumerate(sorted_indices[:5]):
                self.info(f"    {i+1}. Feature_{idx}: {contributions[idx]:.6f}")
        
        if 'prediction' in interpretation:
            pred = interpretation['prediction']
            self.info(f"  Prediction: {pred}")
    
    def close(self):
        """
        Close logger and save final results
        """
        # Save training curves and history
        self.save_training_curves()
        self.save_training_history()
        
        self.info("Logger closed. Results saved.")
