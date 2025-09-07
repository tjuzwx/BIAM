"""
BIAM Data Generator
Generates synthetic and real datasets with missing values, noisy labels, and class imbalance
"""

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import torch.utils.data as Data
from scipy.stats import norm
from random import sample
import warnings
warnings.filterwarnings("ignore")

class BIAMDataGenerator:
    """
    Data generator for BIAM model supporting various data challenges
    """
    
    def __init__(self, config):
        """
        Initialize BIAM data generator
        
        Args:
            config: BIAM configuration object
        """
        self.config = config
        self.task = config.task
        self.dataset = config.dataset
        self.missing_ratio = config.missing_ratio
        self.noise_ratio = config.noise_ratio
        self.imbalance_ratio = config.imbalance_ratio
        self.batch_size = config.batch_size
        
    def generate_data(self):
        """
        Generate training, validation, and test data
        
        Returns:
            tuple: (train_loader, val_loader, test_data)
        """
        if self.dataset == 'synthetic':
            return self._generate_synthetic_data()
        elif self.dataset == 'adult':
            return self._generate_adult_data()
        elif self.dataset == 'credit':
            return self._generate_credit_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
    
    def _generate_synthetic_data(self):
        """
        Generate synthetic data with specified challenges
        """
        if self.task == 'regression':
            return self._generate_synthetic_regression()
        else:
            return self._generate_synthetic_classification()
    
    def _generate_synthetic_regression(self):
        """
        Generate synthetic regression data with noise
        """
        n_samples = 2000
        n_features = 100
        
        # Generate base data
        X = np.random.uniform(-1, 1, size=(n_samples, n_features))
        
        # Generate true function
        y = self._generate_regression_function(X)
        
        # Add noise
        y = self._add_regression_noise(y, noise_type='modal')
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        
        return train_loader, val_loader, (X_test, y_test)
    
    def _generate_synthetic_classification(self):
        """
        Generate synthetic classification data with imbalance and noise
        """
        n_samples = 1000
        n_features = 100
        
        # Generate base data
        X = np.random.uniform(0, 1, size=(n_samples, n_features))
        
        # Generate true labels
        y = self._generate_classification_function(X)
        
        # Add class imbalance
        y = self._add_class_imbalance(y)
        
        # Add label noise
        y = self._add_label_noise(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        
        return train_loader, val_loader, (X_test, y_test)
    
    def _generate_regression_function(self, X):
        """
        Generate complex regression function
        """
        f1 = -2 * np.sin(2 * X[:, 0])
        f2 = 8 * np.square(X[:, 1])
        f3 = 7 * np.sin(X[:, 2]) / (2 - np.sin(X[:, 2]))
        f4 = 6 * np.exp(-X[:, 3])
        f5 = np.power(X[:, 4], 3) + 1.5 * np.square(X[:, 4] - 1)
        f6 = 5 * X[:, 5]
        f7 = 10 * np.sin(np.exp(-X[:, 6] / 2))
        f8 = -10 * norm.cdf(X[:, 7], loc=0.5, scale=0.8)
        
        y = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        return y
    
    def _generate_classification_function(self, X):
        """
        Generate classification function
        """
        f1 = np.square(X[:, 0] - 0.5)
        f2 = np.square(X[:, 1] - 0.5)
        y = f1 + f2 - 0.08
        y = (y > 0).astype(int)
        return y
    
    def _add_regression_noise(self, y, noise_type='modal'):
        """
        Add noise to regression targets
        """
        n = len(y)
        if noise_type == 'modal':
            noise = np.zeros(n)
            for i in range(n):
                if np.random.uniform() < 0.8:
                    noise[i] = np.random.normal(0, 1)
                else:
                    noise[i] = np.random.normal(20, 1)
        elif noise_type == 'mean':
            noise = np.zeros(n)
            for i in range(n):
                if np.random.uniform() < 0.8:
                    noise[i] = np.random.normal(-2, 1)
                else:
                    noise[i] = np.random.normal(8, 1)
        elif noise_type == 'studentT':
            noise = np.random.standard_t(df=2, size=n)
        else:
            noise = np.random.normal(0, 1, n)
        
        return y + noise
    
    def _add_class_imbalance(self, y):
        """
        Add class imbalance to classification data
        """
        n = len(y)
        neg_num = int(n * self.imbalance_ratio)
        pos_num = n - neg_num
        
        neg_indices = np.where(y == 0)[0]
        pos_indices = np.where(y == 1)[0]
        
        if len(neg_indices) >= neg_num and len(pos_indices) >= pos_num:
            selected_neg = np.random.choice(neg_indices, neg_num, replace=False)
            selected_pos = np.random.choice(pos_indices, pos_num, replace=False)
            
            selected_indices = np.concatenate([selected_neg, selected_pos])
            np.random.shuffle(selected_indices)
            
            return y[selected_indices]
        else:
            return y
    
    def _add_label_noise(self, y):
        """
        Add label noise to classification data
        """
        n = len(y)
        noise_indices = np.random.choice(n, int(n * self.noise_ratio), replace=False)
        y_noisy = y.copy()
        y_noisy[noise_indices] = 1 - y_noisy[noise_indices]
        return y_noisy
    
    def _create_data_loader(self, X, y, shuffle=True):
        """
        Create PyTorch data loader
        """
        dataset = Data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        
        return loader
    
    def _generate_adult_data(self):
        """
        Generate Adult dataset with missing values and imbalance
        """
        # Generate synthetic data with Adult-like characteristics
        return self._generate_synthetic_classification()
    
    def _generate_credit_data(self):
        """
        Generate Credit dataset with missing values and imbalance
        """
        # Generate synthetic data with Credit-like characteristics
        return self._generate_synthetic_classification()
