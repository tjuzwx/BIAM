"""
BIAM Data Utilities
Utility functions for data preprocessing and augmentation
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

class BIAMDataUtils:
    """
    Utility functions for BIAM data processing
    """
    
    @staticmethod
    def create_data_loader(X, y, batch_size=200, shuffle=True):
        """
        Create PyTorch data loader
        
        Args:
            X: Input features
            y: Target labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            PyTorch DataLoader
        """
        dataset = Data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        
        return loader
    
    @staticmethod
    def standardize_data(X_train, X_val, X_test, y_train=None, y_val=None, y_test=None):
        """
        Standardize input features and optionally targets
        
        Args:
            X_train, X_val, X_test: Input features
            y_train, y_val, y_test: Target labels (optional)
            
        Returns:
            Standardized data
        """
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        if y_train is not None:
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            return (X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y)
        else:
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler_X
    
    @staticmethod
    def add_missing_values(X, missing_ratio=0.3, missing_pattern='MCAR'):
        """
        Add missing values to data
        
        Args:
            X: Input data
            missing_ratio: Ratio of missing values
            missing_pattern: Missing pattern ('MCAR', 'MAR', 'MNAR')
            
        Returns:
            Data with missing values
        """
        X_missing = X.copy()
        n_samples, n_features = X.shape
        
        if missing_pattern == 'MCAR':
            # Missing Completely At Random
            mask = np.random.random((n_samples, n_features)) < missing_ratio
            X_missing[mask] = np.nan
            
        elif missing_pattern == 'MAR':
            # Missing At Random - depends on observed values
            for i in range(n_features):
                if i > 0:  # Use previous feature to determine missingness
                    prob_missing = 1 / (1 + np.exp(-X[:, i-1]))
                    mask = np.random.random(n_samples) < prob_missing * missing_ratio
                    X_missing[mask, i] = np.nan
                    
        elif missing_pattern == 'MNAR':
            # Missing Not At Random - depends on missing values themselves
            for i in range(n_features):
                prob_missing = 1 / (1 + np.exp(-X[:, i]))
                mask = np.random.random(n_samples) < prob_missing * missing_ratio
                X_missing[mask, i] = np.nan
        
        return X_missing
    
    @staticmethod
    def add_label_noise(y, noise_ratio=0.2):
        """
        Add noise to labels
        
        Args:
            y: Labels
            noise_ratio: Ratio of noisy labels
            
        Returns:
            Labels with noise
        """
        y_noisy = y.copy()
        n_samples = len(y)
        noise_indices = np.random.choice(n_samples, int(n_samples * noise_ratio), replace=False)
        
        # Flip labels for binary classification
        if len(np.unique(y)) == 2:
            y_noisy[noise_indices] = 1 - y_noisy[noise_indices]
        else:
            # For multi-class, randomly assign different labels
            unique_labels = np.unique(y)
            for idx in noise_indices:
                other_labels = unique_labels[unique_labels != y[idx]]
                y_noisy[idx] = np.random.choice(other_labels)
        
        return y_noisy
    
    @staticmethod
    def create_class_imbalance(y, imbalance_ratio=0.15):
        """
        Create class imbalance in data
        
        Args:
            y: Labels
            imbalance_ratio: Ratio of minority class
            
        Returns:
            Indices for imbalanced data
        """
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            return np.arange(len(y))  # Return all indices for multi-class
        
        # Find minority and majority classes
        label_counts = np.bincount(y.astype(int))
        minority_class = np.argmin(label_counts)
        majority_class = np.argmax(label_counts)
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Calculate target counts
        n_minority = len(minority_indices)
        n_majority = int(n_minority / imbalance_ratio)
        
        # Sample majority class
        if len(majority_indices) > n_majority:
            selected_majority = np.random.choice(majority_indices, n_majority, replace=False)
        else:
            selected_majority = majority_indices
        
        # Combine indices
        selected_indices = np.concatenate([minority_indices, selected_majority])
        np.random.shuffle(selected_indices)
        
        return selected_indices
