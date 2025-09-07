# BIAM: Bilevel Interactive Additive Model

A PyTorch implementation of the Bilevel Interactive Additive Model (BIAM) for handling datasets with missing values, noisy labels, and imbalanced categories.

## Overview

BIAM is a novel machine learning framework that addresses three key challenges in real-world datasets:

1. **Missing Values**: Explicitly models missing value indicators and their interactions
2. **Noisy Labels**: Uses a bilevel optimization approach to learn robust sample weights
3. **Class Imbalance**: Dynamically adjusts sample weights to handle imbalanced data

## Key Features

- **Bilevel Optimization**: Upper-level optimization for sample weighting, lower-level optimization for model parameters
- **Additive Model Architecture**: Interpretable feature interactions with missing value handling
- **Advanced Gradient Methods**: Multiple gradient computation strategies for robust optimization
- **Comprehensive Visualization**: Shape function plots, feature importance, and model interpretation
- **Extensive Logging**: Integration with wandb, tensorboard, and custom logging systems

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from biam_main import main
import argparse

# Set up arguments
args = argparse.Namespace(
    task='classification',
    dataset='synthetic',
    missing_ratio=0.3,
    noise_ratio=0.2,
    imbalance_ratio=0.15,
    upper_lr=1e-2,
    lower_lr=1e-2,
    penalty_coef=1e-5,
    epochs=10000,
    batch_size=200,
    use_wandb=True,
    project_name='biam-experiments'
)

# Run training
main()
```

### Command Line Interface

```bash
# Regression task
python biam_main.py --task regression --dataset synthetic --epochs 5000

# Classification task with wandb logging
python biam_main.py --task classification --dataset adult --use_wandb --project_name my-experiment

# Custom configuration
python biam_main.py --task classification --missing_ratio 0.4 --noise_ratio 0.3 --imbalance_ratio 0.1
```

## Project Structure

```
BIAM/
├── biam_main.py                 # Main entry point
├── requirements.txt             # Dependencies
├── README.md                   # This file
├── data/                       # Data processing modules
│   ├── __init__.py
│   ├── biam_data_generator.py  # Synthetic and real data generation
│   ├── biam_binarizer.py       # Missing value handling and binarization
│   └── biam_data_utils.py      # Data utility functions
├── models/                     # Model components
│   ├── __init__.py
│   ├── biam_model.py           # Main BIAM model
│   ├── biam_weighting_network.py # Sample weighting network
│   └── biam_additive_model.py  # Additive model with interactions
├── gradients/                  # Optimization algorithms
│   ├── __init__.py
│   ├── biam_optimizer.py       # Main optimizer
│   ├── biam_bilevel_optimizer.py # Advanced bilevel optimization
│   └── biam_gradient_methods.py # Various gradient methods
├── utils/                      # Utility classes
│   ├── __init__.py
│   ├── biam_config.py          # Configuration management
│   └── biam_logger.py          # Logging utilities
├── visualization/              # Visualization tools
│   ├── __init__.py
│   ├── biam_visualizer.py      # Main visualizer
│   └── biam_shape_functions.py # Shape function plots
└── logs/                       # Training logs and outputs
```

## Model Architecture

### BIAM Framework

BIAM consists of two main components:

1. **Weighting Network**: A neural network that learns sample weights based on prediction errors
2. **Additive Model**: An interpretable model that handles missing values and feature interactions

### Bilevel Optimization

The optimization process involves two levels:

- **Upper Level**: Optimizes the weighting network parameters to minimize validation loss
- **Lower Level**: Optimizes the additive model parameters using weighted training loss

## Configuration

### Key Parameters

- `task`: Task type ('regression' or 'classification')
- `dataset`: Dataset to use ('synthetic', 'adult', 'credit', 'mnist', 'cifar10')
- `missing_ratio`: Ratio of missing values (0.0-1.0)
- `noise_ratio`: Ratio of noisy labels (0.0-1.0)
- `imbalance_ratio`: Ratio for class imbalance (0.0-1.0)
- `upper_lr`: Upper level learning rate
- `lower_lr`: Lower level learning rate
- `penalty_coef`: Regularization coefficient
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training

### Advanced Configuration

```python
from utils.biam_config import BIAMConfig

config = BIAMConfig()
config.update(
    task='classification',
    missing_ratio=0.3,
    noise_ratio=0.2,
    imbalance_ratio=0.15,
    upper_lr=1e-2,
    lower_lr=1e-2,
    penalty_coef=1e-5,
    epochs=10000,
    batch_size=200
)
```

## Data Generation

### Synthetic Data

BIAM includes comprehensive synthetic data generation for testing:

```python
from data.biam_data_generator import BIAMDataGenerator

generator = BIAMDataGenerator(config)
train_loader, val_loader, test_data = generator.generate_data()
```

### Supported Datasets

- **Synthetic**: Custom generated data with various noise patterns
- **Adult**: UCI Adult dataset with missing values
- **Credit**: Credit scoring dataset with imbalance
- **MNIST**: Image classification with label noise
- **CIFAR-10**: Natural image classification

## Visualization

### Training Curves

```python
from visualization.biam_visualizer import BIAMVisualizer

visualizer = BIAMVisualizer(config)
visualizer.plot_training_curves(training_history)
```

### Feature Importance

```python
feature_importance = model.get_feature_importance()
visualizer.plot_feature_importance(feature_importance, feature_names)
```

### Shape Functions

```python
from visualization.biam_shape_functions import BIAMShapeFunctions

shape_plotter = BIAMShapeFunctions(config)
shape_plotter.plot_shape_functions(model, data, feature_names)
```

## Logging and Monitoring

### Wandb Integration

```python
# Enable wandb logging
python biam_main.py --use_wandb --project_name my-biam-experiment
```

### Custom Logging

```python
from utils.biam_logger import BIAMLogger

logger = BIAMLogger(config)
logger.log_metrics(epoch, train_metrics, test_metrics)
logger.log_feature_importance(feature_importance)
```

## Advanced Features

### Multiple Gradient Methods

BIAM supports various gradient computation methods:

- Hypergradient computation
- Implicit differentiation
- Forward-mode differentiation
- Second-order gradients
- Meta-gradient computation

### Regularization Strategies

- Group Lasso regularization
- L1/L2 regularization
- Entropy regularization for weight diversity
- Gradient clipping and noise

### Model Interpretation

- Feature importance analysis
- Missing value indicator analysis
- Feature interaction visualization
- Model complexity analysis

## Performance Optimization

### GPU Acceleration

```python
# Automatic GPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Memory Optimization

- Gradient checkpointing
- Mixed precision training
- Efficient data loading

## Examples

### Regression Example

```python
# Generate synthetic regression data
config = BIAMConfig()
config.task = 'regression'
config.missing_ratio = 0.2
config.noise_ratio = 0.1

generator = BIAMDataGenerator(config)
train_loader, val_loader, test_data = generator.generate_data()

# Train model
model = BIAMModel(config, device)
optimizer = BIAMOptimizer(config, model, weighting_network)

for epoch in range(config.epochs):
    train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
    if epoch % 100 == 0:
        test_metrics = optimizer.evaluate(test_data)
        print(f"Epoch {epoch}: Test RMSE = {test_metrics['rmse']:.4f}")
```

### Classification Example

```python
# Generate synthetic classification data with imbalance
config = BIAMConfig()
config.task = 'classification'
config.imbalance_ratio = 0.1
config.noise_ratio = 0.3

generator = BIAMDataGenerator(config)
train_loader, val_loader, test_data = generator.generate_data()

# Train model
model = BIAMModel(config, device)
optimizer = BIAMOptimizer(config, model, weighting_network)

for epoch in range(config.epochs):
    train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
    if epoch % 100 == 0:
        test_metrics = optimizer.evaluate(test_data)
        print(f"Epoch {epoch}: Test Accuracy = {test_metrics['accuracy']:.4f}")
```

## Citation

If you use BIAM in your research, please cite:

```bibtex
@article{zhou2025biam,
  title={BIAM: Bilevel Interactive Additive Model for Datasets with Missing Values, Noisy Labels and Imbalanced Categories},
  author={Zhou, Wenxing and Xu, Chao and Zhang, Xuelin},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This implementation is based on the BIAM paper
- Inspired by Meta-Weight-Net and other bilevel optimization methods
- Built with PyTorch and modern deep learning practices