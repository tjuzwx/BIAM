"""
BIAM Dataset Loader
Extended dataset support for BIAM model
"""

import torch
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine, load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple, Optional
import os
import urllib.request
import zipfile

class BIAMDatasetLoader:
    """
    Extended dataset loader for BIAM model
    """
    
    def __init__(self, config):
        """
        Initialize dataset loader
        
        Args:
            config: BIAM configuration
        """
        self.config = config
        self.data_dir = "datasets"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_adult_dataset(self, missing_ratio: float = 0.3, noise_ratio: float = 0.2):
        """
        Load Adult dataset with missing values and noise
        
        Args:
            missing_ratio: Ratio of missing values
            noise_ratio: Ratio of noisy labels
            
        Returns:
            Tuple of (train_loader, val_loader, test_data)
        """
        try:
            # Load Adult dataset from OpenML
            adult = fetch_openml('adult', version=2, as_frame=True)
            X, y = adult.data, adult.target
            
            # Preprocess categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target variable
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            
            # Convert to numpy
            X = X.values.astype(np.float32)
            y = y.astype(np.float32)
            
            # Add missing values
            if missing_ratio > 0:
                missing_mask = np.random.random(X.shape) < missing_ratio
                X[missing_mask] = np.nan
            
            # Add label noise
            if noise_ratio > 0:
                noise_indices = np.random.choice(len(y), int(len(y) * noise_ratio), replace=False)
                y[noise_indices] = 1 - y[noise_indices]
            
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
            train_loader = self._create_dataloader(X_train, y_train)
            val_loader = self._create_dataloader(X_val, y_val)
            
            return train_loader, val_loader, (X_test, y_test)
            
        except Exception as e:
            print(f"Error loading Adult dataset: {e}")
            return self._load_fallback_dataset()
    
    def load_credit_dataset(self, missing_ratio: float = 0.3, noise_ratio: float = 0.2):
        """
        Load Credit dataset with missing values and noise
        
        Args:
            missing_ratio: Ratio of missing values
            noise_ratio: Ratio of noisy labels
            
        Returns:
            Tuple of (train_loader, val_loader, test_data)
        """
        try:
            # Load German Credit dataset from OpenML
            credit = fetch_openml('credit-g', version=1, as_frame=True)
            X, y = credit.data, credit.target
            
            # Preprocess categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target variable
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            
            # Convert to numpy
            X = X.values.astype(np.float32)
            y = y.astype(np.float32)
            
            # Add missing values
            if missing_ratio > 0:
                missing_mask = np.random.random(X.shape) < missing_ratio
                X[missing_mask] = np.nan
            
            # Add label noise
            if noise_ratio > 0:
                noise_indices = np.random.choice(len(y), int(len(y) * noise_ratio), replace=False)
                y[noise_indices] = 1 - y[noise_indices]
            
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
            train_loader = self._create_dataloader(X_train, y_train)
            val_loader = self._create_dataloader(X_val, y_val)
            
            return train_loader, val_loader, (X_test, y_test)
            
        except Exception as e:
            print(f"Error loading Credit dataset: {e}")
            return self._load_fallback_dataset()
    
    def load_breast_cancer_dataset(self, missing_ratio: float = 0.3, noise_ratio: float = 0.2):
        """
        Load Breast Cancer dataset with missing values and noise
        
        Args:
            missing_ratio: Ratio of missing values
            noise_ratio: Ratio of noisy labels
            
        Returns:
            Tuple of (train_loader, val_loader, test_data)
        """
        try:
            # Load Breast Cancer dataset
            cancer = load_breast_cancer()
            X, y = cancer.data, cancer.target
            
            # Convert to float32
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            
            # Add missing values
            if missing_ratio > 0:
                missing_mask = np.random.random(X.shape) < missing_ratio
                X[missing_mask] = np.nan
            
            # Add label noise
            if noise_ratio > 0:
                noise_indices = np.random.choice(len(y), int(len(y) * noise_ratio), replace=False)
                y[noise_indices] = 1 - y[noise_indices]
            
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
            train_loader = self._create_dataloader(X_train, y_train)
            val_loader = self._create_dataloader(X_val, y_val)
            
            return train_loader, val_loader, (X_test, y_test)
            
        except Exception as e:
            print(f"Error loading Breast Cancer dataset: {e}")
            return self._load_fallback_dataset()
    
    def load_wine_dataset(self, missing_ratio: float = 0.3, noise_ratio: float = 0.2):
        """
        Load Wine dataset with missing values and noise
        
        Args:
            missing_ratio: Ratio of missing values
            noise_ratio: Ratio of noisy labels
            
        Returns:
            Tuple of (train_loader, val_loader, test_data)
        """
        try:
            # Load Wine dataset
            wine = load_wine()
            X, y = wine.data, wine.target
            
            # Convert to float32
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            
            # Add missing values
            if missing_ratio > 0:
                missing_mask = np.random.random(X.shape) < missing_ratio
                X[missing_mask] = np.nan
            
            # Add label noise
            if noise_ratio > 0:
                noise_indices = np.random.choice(len(y), int(len(y) * noise_ratio), replace=False)
                # For multi-class, randomly assign different labels
                unique_labels = np.unique(y)
                for idx in noise_indices:
                    other_labels = unique_labels[unique_labels != y[idx]]
                    y[idx] = np.random.choice(other_labels)
            
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
            train_loader = self._create_dataloader(X_train, y_train)
            val_loader = self._create_dataloader(X_val, y_val)
            
            return train_loader, val_loader, (X_test, y_test)
            
        except Exception as e:
            print(f"Error loading Wine dataset: {e}")
            return self._load_fallback_dataset()
    
    def load_iris_dataset(self, missing_ratio: float = 0.3, noise_ratio: float = 0.2):
        """
        Load Iris dataset with missing values and noise
        
        Args:
            missing_ratio: Ratio of missing values
            noise_ratio: Ratio of noisy labels
            
        Returns:
            Tuple of (train_loader, val_loader, test_data)
        """
        try:
            # Load Iris dataset
            iris = load_iris()
            X, y = iris.data, iris.target
            
            # Convert to float32
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            
            # Add missing values
            if missing_ratio > 0:
                missing_mask = np.random.random(X.shape) < missing_ratio
                X[missing_mask] = np.nan
            
            # Add label noise
            if noise_ratio > 0:
                noise_indices = np.random.choice(len(y), int(len(y) * noise_ratio), replace=False)
                # For multi-class, randomly assign different labels
                unique_labels = np.unique(y)
                for idx in noise_indices:
                    other_labels = unique_labels[unique_labels != y[idx]]
                    y[idx] = np.random.choice(other_labels)
            
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
            train_loader = self._create_dataloader(X_train, y_train)
            val_loader = self._create_dataloader(X_val, y_val)
            
            return train_loader, val_loader, (X_test, y_test)
            
        except Exception as e:
            print(f"Error loading Iris dataset: {e}")
            return self._load_fallback_dataset()
    
    def load_custom_dataset(self, file_path: str, target_column: str, 
                          missing_ratio: float = 0.3, noise_ratio: float = 0.2):
        """
        Load custom dataset from file
        
        Args:
            file_path: Path to dataset file
            target_column: Name of target column
            missing_ratio: Ratio of missing values
            noise_ratio: Ratio of noisy labels
            
        Returns:
            Tuple of (train_loader, val_loader, test_data)
        """
        try:
            # Load dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Preprocess categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target variable
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            
            # Convert to numpy
            X = X.values.astype(np.float32)
            y = y.astype(np.float32)
            
            # Add missing values
            if missing_ratio > 0:
                missing_mask = np.random.random(X.shape) < missing_ratio
                X[missing_mask] = np.nan
            
            # Add label noise
            if noise_ratio > 0:
                noise_indices = np.random.choice(len(y), int(len(y) * noise_ratio), replace=False)
                if len(np.unique(y)) == 2:
                    y[noise_indices] = 1 - y[noise_indices]
                else:
                    unique_labels = np.unique(y)
                    for idx in noise_indices:
                        other_labels = unique_labels[unique_labels != y[idx]]
                        y[idx] = np.random.choice(other_labels)
            
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
            train_loader = self._create_dataloader(X_train, y_train)
            val_loader = self._create_dataloader(X_val, y_val)
            
            return train_loader, val_loader, (X_test, y_test)
            
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            return self._load_fallback_dataset()
    
    def _create_dataloader(self, X, y, batch_size: int = None):
        """
        Create PyTorch data loader
        
        Args:
            X: Input features
            y: Target labels
            batch_size: Batch size
            
        Returns:
            PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        dataset = Data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        return loader
    
    def _load_fallback_dataset(self):
        """
        Load fallback synthetic dataset
        
        Returns:
            Tuple of (train_loader, val_loader, test_data)
        """
        print("Loading fallback synthetic dataset...")
        
        # Generate synthetic data
        X = np.random.randn(1000, 10).astype(np.float32)
        y = (np.sum(X, axis=1) > 0).astype(np.float32)
        
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
        train_loader = self._create_dataloader(X_train, y_train)
        val_loader = self._create_dataloader(X_val, y_val)
        
        return train_loader, val_loader, (X_test, y_test)
    
    def get_available_datasets(self):
        """
        Get list of available datasets
        
        Returns:
            List of available dataset names
        """
        return [
            'synthetic',
            'adult',
            'credit',
            'breast_cancer',
            'wine',
            'iris',
            'custom'
        ]
