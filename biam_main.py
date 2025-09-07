"""
BIAM: Bilevel Interactive Additive Model
Main entry point for training and evaluation
"""

import os
import sys
import argparse
import torch
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.biam_data_generator import BIAMDataGenerator
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_logger import BIAMLogger
from utils.biam_config import BIAMConfig

def main():
    """
    Main function for BIAM training and evaluation
    """
    parser = argparse.ArgumentParser(description='BIAM: Bilevel Interactive Additive Model')
    parser.add_argument('--task', type=str, default='classification', 
                       choices=['regression', 'classification'],
                       help='Task type: regression or classification')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'adult', 'credit', 'mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--missing_ratio', type=float, default=0.3,
                       help='Ratio of missing values')
    parser.add_argument('--noise_ratio', type=float, default=0.2,
                       help='Ratio of noisy labels')
    parser.add_argument('--imbalance_ratio', type=float, default=0.15,
                       help='Ratio for class imbalance')
    parser.add_argument('--upper_lr', type=float, default=1e-2,
                       help='Upper level learning rate')
    parser.add_argument('--lower_lr', type=float, default=1e-2,
                       help='Lower level learning rate')
    parser.add_argument('--penalty_coef', type=float, default=1e-5,
                       help='Penalty coefficient for regularization')
    parser.add_argument('--epochs', type=int, default=10000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch size')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    parser.add_argument('--project_name', type=str, default='biam-experiments',
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = BIAMConfig(args)
    
    # Initialize logger
    logger = BIAMLogger(config)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args),
            name=f"biam_{args.task}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate data
    data_generator = BIAMDataGenerator(config)
    train_loader, val_loader, test_data = data_generator.generate_data()
    
    # Initialize BIAM model
    biam_model = BIAMModel(config, device)
    weighting_network = BIAMWeightingNetwork(config, device)
    
    # Initialize optimizer
    optimizer = BIAMOptimizer(config, biam_model, weighting_network)
    
    # Training loop
    logger.info("Starting BIAM training...")
    for epoch in range(args.epochs):
        # Train one epoch
        train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
        
        # Evaluate on test set
        if epoch % 100 == 0:
            test_metrics = optimizer.evaluate(test_data)
            
            # Log metrics
            logger.log_metrics(epoch, train_metrics, test_metrics)
            
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': train_metrics['val_loss'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1_score']
                })
    
    logger.info("Training completed!")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
