"""
Unit tests for BIAM model components
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from models.biam_additive_model import BIAMAdditiveModel
from utils.biam_config import BIAMConfig

class TestBIAMModels(unittest.TestCase):
    """
    Test cases for BIAM model components
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.config = BIAMConfig()
        self.config.task = 'classification'
        self.config.input_dim = 10
        self.config.num_classes = 2
        self.config.device = torch.device('cpu')
        
        self.batch_size = 32
        self.input_dim = 10
        
        # Create test data
        self.test_input = torch.randn(self.batch_size, self.input_dim)
        self.test_target = torch.randint(0, 2, (self.batch_size,))
    
    def test_biam_weighting_network_initialization(self):
        """
        Test BIAM weighting network initialization
        """
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        
        # Test network structure
        self.assertEqual(weighting_network.input_dim, 1)
        self.assertEqual(weighting_network.hidden_dim, 10)
        self.assertEqual(weighting_network.output_dim, 1)
        
        # Test forward pass
        test_losses = torch.randn(self.batch_size, 1)
        weights = weighting_network(test_losses)
        
        self.assertEqual(weights.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(weights >= 0))  # Weights should be positive
        self.assertTrue(torch.all(weights <= 1))  # Weights should be <= 1 (sigmoid output)
    
    def test_biam_additive_model_initialization(self):
        """
        Test BIAM additive model initialization
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        # Test model structure
        self.assertEqual(additive_model.input_dim, self.config.input_dim)
        self.assertEqual(additive_model.output_dim, self.config.num_classes)
        
        # Test forward pass
        output = additive_model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.config.num_classes))
    
    def test_biam_model_initialization(self):
        """
        Test BIAM model initialization
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Test model components
        self.assertIsInstance(biam_model.additive_model, BIAMAdditiveModel)
        self.assertIsInstance(biam_model.weighting_network, BIAMWeightingNetwork)
        
        # Test forward pass
        output = biam_model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.config.num_classes))
    
    def test_biam_model_with_weights(self):
        """
        Test BIAM model with weight return
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Test forward pass with weights
        output, weights = biam_model(self.test_input, return_weights=True)
        
        self.assertEqual(output.shape, (self.batch_size, self.config.num_classes))
        self.assertEqual(weights.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(weights >= 0))
    
    def test_feature_importance(self):
        """
        Test feature importance calculation
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Test feature importance
        importance = biam_model.get_feature_importance()
        
        self.assertEqual(len(importance), self.config.input_dim)
        self.assertTrue(np.all(importance >= 0))  # Importance should be non-negative
    
    def test_missing_indicators(self):
        """
        Test missing value indicators
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Test missing indicators
        missing_indicators = biam_model.get_missing_indicators()
        
        self.assertEqual(len(missing_indicators), self.config.input_dim)
    
    def test_model_uncertainty(self):
        """
        Test model uncertainty estimation
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Test uncertainty estimation
        mean_pred, std_pred = biam_model.predict_with_uncertainty(self.test_input)
        
        self.assertEqual(mean_pred.shape, (self.batch_size, self.config.num_classes))
        self.assertEqual(std_pred.shape, (self.batch_size, self.config.num_classes))
        self.assertTrue(torch.all(std_pred >= 0))  # Standard deviation should be non-negative
    
    def test_regression_mode(self):
        """
        Test BIAM model in regression mode
        """
        config = BIAMConfig()
        config.task = 'regression'
        config.input_dim = 10
        config.device = torch.device('cpu')
        
        biam_model = BIAMModel(config, config.device)
        
        # Test forward pass
        output = biam_model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, 1))  # Single output for regression
    
    def test_model_interpretation(self):
        """
        Test model interpretation
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Test model interpretation
        sample_data = self.test_input[:1]  # Single sample
        interpretation = biam_model.additive_model.get_model_interpretation(sample_data)
        
        self.assertIn('feature_contributions', interpretation)
        self.assertIn('missing_indicators', interpretation)
        self.assertIn('interaction_weights', interpretation)
        self.assertIn('prediction', interpretation)
        
        self.assertEqual(len(interpretation['feature_contributions']), self.config.input_dim)
    
    def test_regularization_loss(self):
        """
        Test regularization loss calculation
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        # Test different regularization types
        reg_loss_l1 = additive_model.compute_regularization_loss('l1')
        reg_loss_l2 = additive_model.compute_regularization_loss('l2')
        reg_loss_group = additive_model.compute_regularization_loss('group_lasso')
        
        self.assertIsInstance(reg_loss_l1, torch.Tensor)
        self.assertIsInstance(reg_loss_l2, torch.Tensor)
        self.assertIsInstance(reg_loss_group, torch.Tensor)
        
        self.assertTrue(reg_loss_l1 >= 0)
        self.assertTrue(reg_loss_l2 >= 0)
        self.assertTrue(reg_loss_group >= 0)
    
    def test_weight_statistics(self):
        """
        Test weight statistics calculation
        """
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        
        # Test weight statistics
        test_losses = torch.randn(self.batch_size, 1)
        stats = weighting_network.get_weight_statistics(test_losses)
        
        self.assertIn('mean_weight', stats)
        self.assertIn('std_weight', stats)
        self.assertIn('min_weight', stats)
        self.assertIn('max_weight', stats)
        self.assertIn('weight_entropy', stats)
        
        self.assertTrue(0 <= stats['mean_weight'] <= 1)
        self.assertTrue(stats['min_weight'] >= 0)
        self.assertTrue(stats['max_weight'] <= 1)
    
    def test_dynamic_weight_update(self):
        """
        Test dynamic weight update
        """
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        
        # Test dynamic weight update
        test_losses = torch.randn(self.batch_size, 1)
        epoch = 100
        total_epochs = 1000
        
        updated_weights = weighting_network.update_weights_dynamically(
            test_losses, epoch, total_epochs
        )
        
        self.assertEqual(updated_weights.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(updated_weights >= 0))
    
    def test_spline_transformation(self):
        """
        Test spline transformation
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        # Test spline transformation
        x_spline = additive_model._apply_spline_transformation(self.test_input)
        
        expected_dim = self.config.input_dim * self.config.get_spline_dim()
        self.assertEqual(x_spline.shape, (self.batch_size, expected_dim))
    
    def test_missing_value_handling(self):
        """
        Test missing value handling
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        # Create input with missing values
        input_with_missing = self.test_input.clone()
        input_with_missing[0, 0] = float('nan')  # Add missing value
        
        # Test missing value handling
        x_spline = additive_model._apply_spline_transformation(input_with_missing)
        x_with_missing = additive_model._add_missing_indicators(x_spline, input_with_missing)
        
        # Should have additional missing indicators
        expected_dim = x_spline.shape[1] + self.config.input_dim
        self.assertEqual(x_with_missing.shape, (self.batch_size, expected_dim))
    
    def test_feature_interactions(self):
        """
        Test feature interactions
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        # Test feature interactions
        x_spline = additive_model._apply_spline_transformation(self.test_input)
        x_with_interactions = additive_model._apply_feature_interactions(x_spline, self.test_input)
        
        # Should have additional interaction features
        self.assertTrue(x_with_interactions.shape[1] >= x_spline.shape[1])

if __name__ == '__main__':
    unittest.main()
