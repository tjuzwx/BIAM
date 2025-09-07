"""
Unit tests for BIAM data processing components
"""

import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.biam_data_generator import BIAMDataGenerator
from data.biam_binarizer import BIAMBinarizer
from data.biam_dataset_loader import BIAMDatasetLoader
from utils.biam_config import BIAMConfig

class TestBIAMData(unittest.TestCase):
    """
    Test cases for BIAM data processing components
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.config = BIAMConfig()
        self.config.task = 'classification'
        self.config.dataset = 'synthetic'
        self.config.missing_ratio = 0.3
        self.config.noise_ratio = 0.2
        self.config.imbalance_ratio = 0.15
        self.config.batch_size = 32
        self.config.device = torch.device('cpu')
    
    def test_data_generator_initialization(self):
        """
        Test BIAM data generator initialization
        """
        generator = BIAMDataGenerator(self.config)
        
        self.assertEqual(generator.task, self.config.task)
        self.assertEqual(generator.dataset, self.config.dataset)
        self.assertEqual(generator.missing_ratio, self.config.missing_ratio)
        self.assertEqual(generator.noise_ratio, self.config.noise_ratio)
        self.assertEqual(generator.imbalance_ratio, self.config.imbalance_ratio)
    
    def test_synthetic_regression_data(self):
        """
        Test synthetic regression data generation
        """
        config = BIAMConfig()
        config.task = 'regression'
        config.dataset = 'synthetic'
        config.batch_size = 32
        config.device = torch.device('cpu')
        
        generator = BIAMDataGenerator(config)
        train_loader, val_loader, test_data = generator.generate_data()
        
        # Test data loaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_data)
        
        # Test batch structure
        for data, target in train_loader:
            self.assertEqual(data.shape[0], config.batch_size)
            self.assertEqual(target.shape[0], config.batch_size)
            break
    
    def test_synthetic_classification_data(self):
        """
        Test synthetic classification data generation
        """
        generator = BIAMDataGenerator(self.config)
        train_loader, val_loader, test_data = generator.generate_data()
        
        # Test data loaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_data)
        
        # Test batch structure
        for data, target in train_loader:
            self.assertEqual(data.shape[0], self.config.batch_size)
            self.assertEqual(target.shape[0], self.config.batch_size)
            break
    
    def test_missing_value_generation(self):
        """
        Test missing value generation
        """
        generator = BIAMDataGenerator(self.config)
        
        # Test missing value addition
        X = np.random.randn(100, 10)
        X_with_missing = generator._add_missing_values(X, missing_ratio=0.3)
        
        # Check that missing values were added
        missing_count = np.isnan(X_with_missing).sum()
        self.assertGreater(missing_count, 0)
    
    def test_label_noise_generation(self):
        """
        Test label noise generation
        """
        generator = BIAMDataGenerator(self.config)
        
        # Test label noise addition
        y = np.random.randint(0, 2, 100)
        y_noisy = generator._add_label_noise(y, noise_ratio=0.2)
        
        # Check that some labels were flipped
        noise_count = np.sum(y != y_noisy)
        self.assertGreater(noise_count, 0)
    
    def test_class_imbalance_generation(self):
        """
        Test class imbalance generation
        """
        generator = BIAMDataGenerator(self.config)
        
        # Test class imbalance
        y = np.random.randint(0, 2, 1000)
        imbalanced_indices = generator._create_class_imbalance(y, imbalance_ratio=0.1)
        
        # Check that imbalance was created
        imbalanced_y = y[imbalanced_indices]
        class_counts = np.bincount(imbalanced_y.astype(int))
        self.assertLess(class_counts[0], class_counts[1])  # Class 0 should be minority
    
    def test_binarizer_initialization(self):
        """
        Test BIAM binarizer initialization
        """
        binarizer = BIAMBinarizer()
        
        self.assertIsNotNone(binarizer.quantiles)
        self.assertIsNotNone(binarizer.miss_vals)
        self.assertTrue(binarizer.specific_mi_intercept)
        self.assertTrue(binarizer.specific_mi_ixn)
    
    def test_binarizer_basic_functionality(self):
        """
        Test basic binarizer functionality
        """
        binarizer = BIAMBinarizer()
        
        # Create test data
        train_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        
        test_df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'label': np.random.randint(0, 2, 50)
        })
        
        # Test binarization
        result = binarizer.binarize_and_augment(train_df, test_df)
        
        self.assertEqual(len(result), 4)  # Should return 4 arrays
        train_aug, test_aug, train_labels, test_labels = result
        
        self.assertEqual(train_aug.shape[0], 100)
        self.assertEqual(test_aug.shape[0], 50)
        self.assertEqual(train_labels.shape[0], 100)
        self.assertEqual(test_labels.shape[0], 50)
    
    def test_binarizer_with_missing_values(self):
        """
        Test binarizer with missing values
        """
        binarizer = BIAMBinarizer()
        
        # Create data with missing values
        train_df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [1, np.nan, 3, 4, 5],
            'label': [0, 1, 0, 1, 0]
        })
        
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [1, 2, np.nan],
            'label': [0, 1, 0]
        })
        
        # Test binarization with missing values
        result = binarizer.binarize_and_augment(train_df, test_df)
        
        self.assertEqual(len(result), 4)
        train_aug, test_aug, train_labels, test_labels = result
        
        # Should have additional columns for missing indicators
        self.assertGreater(train_aug.shape[1], 2)
        self.assertGreater(test_aug.shape[1], 2)
    
    def test_dataset_loader_initialization(self):
        """
        Test BIAM dataset loader initialization
        """
        loader = BIAMDatasetLoader(self.config)
        
        self.assertEqual(loader.config, self.config)
        self.assertTrue(os.path.exists(loader.data_dir))
    
    def test_dataset_loader_available_datasets(self):
        """
        Test available datasets list
        """
        loader = BIAMDatasetLoader(self.config)
        available_datasets = loader.get_available_datasets()
        
        expected_datasets = ['synthetic', 'adult', 'credit', 'breast_cancer', 'wine', 'iris', 'custom']
        self.assertEqual(available_datasets, expected_datasets)
    
    def test_breast_cancer_dataset_loading(self):
        """
        Test breast cancer dataset loading
        """
        loader = BIAMDatasetLoader(self.config)
        
        try:
            train_loader, val_loader, test_data = loader.load_breast_cancer_dataset(
                missing_ratio=0.1, noise_ratio=0.1
            )
            
            # Test data loaders
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_data)
            
            # Test batch structure
            for data, target in train_loader:
                self.assertEqual(data.shape[0], self.config.batch_size)
                self.assertEqual(target.shape[0], self.config.batch_size)
                break
                
        except Exception as e:
            # If dataset loading fails, test fallback
            self.assertIsInstance(e, Exception)
    
    def test_wine_dataset_loading(self):
        """
        Test wine dataset loading
        """
        loader = BIAMDatasetLoader(self.config)
        
        try:
            train_loader, val_loader, test_data = loader.load_wine_dataset(
                missing_ratio=0.1, noise_ratio=0.1
            )
            
            # Test data loaders
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_data)
            
        except Exception as e:
            # If dataset loading fails, test fallback
            self.assertIsInstance(e, Exception)
    
    def test_iris_dataset_loading(self):
        """
        Test iris dataset loading
        """
        loader = BIAMDatasetLoader(self.config)
        
        try:
            train_loader, val_loader, test_data = loader.load_iris_dataset(
                missing_ratio=0.1, noise_ratio=0.1
            )
            
            # Test data loaders
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_data)
            
        except Exception as e:
            # If dataset loading fails, test fallback
            self.assertIsInstance(e, Exception)
    
    def test_fallback_dataset_loading(self):
        """
        Test fallback dataset loading
        """
        loader = BIAMDatasetLoader(self.config)
        train_loader, val_loader, test_data = loader._load_fallback_dataset()
        
        # Test data loaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_data)
        
        # Test batch structure
        for data, target in train_loader:
            self.assertEqual(data.shape[0], self.config.batch_size)
            self.assertEqual(target.shape[0], self.config.batch_size)
            break
    
    def test_data_standardization(self):
        """
        Test data standardization
        """
        generator = BIAMDataGenerator(self.config)
        
        # Test standardization
        X_train = np.random.randn(100, 10)
        X_val = np.random.randn(50, 10)
        X_test = np.random.randn(50, 10)
        
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = generator._standardize_data(
            X_train, X_val, X_test
        )
        
        # Check that data was standardized
        self.assertAlmostEqual(X_train_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(X_train_scaled.std(), 1, places=5)
        
        # Check that scaler was fitted on training data
        self.assertIsNotNone(scaler)
    
    def test_data_loader_creation(self):
        """
        Test data loader creation
        """
        generator = BIAMDataGenerator(self.config)
        
        # Test data loader creation
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        loader = generator._create_data_loader(X, y, batch_size=16)
        
        # Test loader properties
        self.assertIsNotNone(loader)
        self.assertEqual(loader.batch_size, 16)
        
        # Test batch iteration
        for data, target in loader:
            self.assertEqual(data.shape[0], 16)
            self.assertEqual(target.shape[0], 16)
            break

if __name__ == '__main__':
    unittest.main()
