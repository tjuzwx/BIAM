"""
BIAM Demo Script
Demonstration of BIAM model capabilities
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.biam_data_generator import BIAMDataGenerator
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_config import BIAMConfig
from utils.biam_logger import BIAMLogger
from visualization.biam_visualizer import BIAMVisualizer

def demo_biam_classification():
    """
    Demonstrate BIAM for classification task
    """
    print("=" * 60)
    print("BIAM Classification Demo")
    print("=" * 60)
    
    # Configuration
    config = BIAMConfig()
    config.task = 'classification'
    config.dataset = 'synthetic'
    config.missing_ratio = 0.3
    config.noise_ratio = 0.2
    config.imbalance_ratio = 0.15
    config.upper_lr = 1e-2
    config.lower_lr = 1e-2
    config.penalty_coef = 1e-5
    config.epochs = 1000
    config.batch_size = 100
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = BIAMLogger(config)
    logger.log_config()
    
    # Generate data
    print("\nGenerating synthetic classification data...")
    generator = BIAMDataGenerator(config)
    train_loader, val_loader, test_data = generator.generate_data()
    
    # Initialize model
    print("Initializing BIAM model...")
    biam_model = BIAMModel(config, device)
    weighting_network = BIAMWeightingNetwork(config, device)
    
    # Log model info
    logger.log_model_info(biam_model)
    
    # Initialize optimizer
    optimizer = BIAMOptimizer(config, biam_model, weighting_network)
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
        
        if epoch % 100 == 0:
            test_metrics = optimizer.evaluate(test_data)
            logger.log_metrics(epoch, train_metrics, test_metrics)
            
            print(f"Epoch {epoch:4d} | "
                  f"Train Loss: {train_metrics['loss']:.6f} | "
                  f"Val Loss: {train_metrics['val_loss']:.6f} | "
                  f"Test Acc: {test_metrics['accuracy']:.4f} | "
                  f"Test F1: {test_metrics['f1_score']:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = optimizer.evaluate(test_data)
    print(f"Final Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final Test F1 Score: {final_metrics['f1_score']:.4f}")
    
    # Feature importance analysis
    print("\nFeature importance analysis...")
    feature_importance = biam_model.get_feature_importance()
    logger.log_feature_importance(feature_importance)
    
    # Missing value analysis
    print("\nMissing value analysis...")
    missing_indicators = biam_model.get_missing_indicators()
    logger.log_missing_value_analysis(missing_indicators)
    
    # Visualization
    print("\nGenerating visualizations...")
    visualizer = BIAMVisualizer(config)
    
    # Training curves
    training_history = optimizer.get_training_history()
    visualizer.plot_training_curves(training_history)
    
    # Feature importance plot
    visualizer.plot_feature_importance(feature_importance)
    
    # Missing value analysis plot
    visualizer.plot_missing_value_analysis(missing_indicators)
    
    # Model interpretation
    X_test, y_test = test_data
    sample_idx = 0
    sample_data = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    interpretation = biam_model.predict_with_uncertainty(sample_data)
    logger.log_model_interpretation(interpretation)
    
    # Close logger
    logger.close()
    
    print("\nDemo completed successfully!")
    print("Check the 'logs' and 'visualizations' directories for outputs.")

def demo_biam_regression():
    """
    Demonstrate BIAM for regression task
    """
    print("=" * 60)
    print("BIAM Regression Demo")
    print("=" * 60)
    
    # Configuration
    config = BIAMConfig()
    config.task = 'regression'
    config.dataset = 'synthetic'
    config.missing_ratio = 0.2
    config.noise_ratio = 0.1
    config.upper_lr = 1e-2
    config.lower_lr = 1e-2
    config.penalty_coef = 1e-5
    config.epochs = 1000
    config.batch_size = 100
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = BIAMLogger(config)
    logger.log_config()
    
    # Generate data
    print("\nGenerating synthetic regression data...")
    generator = BIAMDataGenerator(config)
    train_loader, val_loader, test_data = generator.generate_data()
    
    # Initialize model
    print("Initializing BIAM model...")
    biam_model = BIAMModel(config, device)
    weighting_network = BIAMWeightingNetwork(config, device)
    
    # Initialize optimizer
    optimizer = BIAMOptimizer(config, biam_model, weighting_network)
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
        
        if epoch % 100 == 0:
            test_metrics = optimizer.evaluate(test_data)
            logger.log_metrics(epoch, train_metrics, test_metrics)
            
            print(f"Epoch {epoch:4d} | "
                  f"Train Loss: {train_metrics['loss']:.6f} | "
                  f"Val Loss: {train_metrics['val_loss']:.6f} | "
                  f"Test RMSE: {test_metrics['rmse']:.6f} | "
                  f"Test MAE: {test_metrics['mae']:.6f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = optimizer.evaluate(test_data)
    print(f"Final Test RMSE: {final_metrics['rmse']:.6f}")
    print(f"Final Test MAE: {final_metrics['mae']:.6f}")
    
    # Feature importance analysis
    print("\nFeature importance analysis...")
    feature_importance = biam_model.get_feature_importance()
    logger.log_feature_importance(feature_importance)
    
    # Visualization
    print("\nGenerating visualizations...")
    visualizer = BIAMVisualizer(config)
    
    # Training curves
    training_history = optimizer.get_training_history()
    visualizer.plot_training_curves(training_history)
    
    # Feature importance plot
    visualizer.plot_feature_importance(feature_importance)
    
    # Close logger
    logger.close()
    
    print("\nRegression demo completed successfully!")
    print("Check the 'logs' and 'visualizations' directories for outputs.")

def main():
    """
    Main demo function
    """
    print("BIAM: Bilevel Interactive Additive Model Demo")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run classification demo
        demo_biam_classification()
        
        print("\n" + "=" * 60)
        
        # Run regression demo
        demo_biam_regression()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
