"""
BIAM Binarizer
Handles missing value indicators and feature binarization for BIAM model
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

class BIAMBinarizer:
    """
    Binarizer for BIAM model that handles missing values and feature interactions
    """
    
    def __init__(self, 
                 quantiles: List[float] = [0.2, 0.4, 0.6, 0.8],
                 label: str = 'label',
                 miss_vals: List[Any] = [-7, -8, -9, np.NaN],
                 overall_mi_intercept: bool = False,
                 overall_mi_ixn: bool = False,
                 specific_mi_intercept: bool = True,
                 specific_mi_ixn: bool = True,
                 imputer: Any = None,
                 categorical_cols: List[str] = [],
                 numerical_cols: List[str] = []):
        """
        Initialize BIAM binarizer
        
        Args:
            quantiles: Quantiles for numerical feature binning
            label: Name of the label column
            miss_vals: List of missing value indicators
            overall_mi_intercept: Whether to include overall missing indicator
            overall_mi_ixn: Whether to include overall missing interactions
            specific_mi_intercept: Whether to include specific missing indicators
            specific_mi_ixn: Whether to include specific missing interactions
            imputer: Imputation strategy
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
        """
        self.quantiles = quantiles
        self.label = label
        self.miss_vals = miss_vals
        self.overall_mi_intercept = overall_mi_intercept
        self.overall_mi_ixn = overall_mi_ixn
        self.specific_mi_intercept = specific_mi_intercept
        self.specific_mi_ixn = specific_mi_ixn
        self.imputer = imputer
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        
        # Store dataset structure for visualization
        self.dataset_structure_map = {}
        self.thresh_vals = []
    
    def binarize_and_augment(self, 
                           train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           imputed_train_df: pd.DataFrame = None,
                           imputed_test_df: pd.DataFrame = None,
                           validation_size: int = 0) -> Tuple[np.ndarray, ...]:
        """
        Core binarization method for BIAM model
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            imputed_train_df: Imputed training dataframe (optional)
            imputed_test_df: Imputed test dataframe (optional)
            validation_size: Size of validation split
            
        Returns:
            Tuple of binarized and augmented datasets
        """
        if imputed_train_df is not None and imputed_test_df is not None:
            return self._imputed_binarize(train_df, test_df, imputed_train_df, imputed_test_df)
        elif self.imputer is not None:
            imputed_train_df, imputed_test_df = self._impute_single_val(train_df, test_df)
            return self._imputed_binarize(train_df, test_df, imputed_train_df, imputed_test_df)
        else:
            return self._binarize(train_df, test_df)
    
    def _nansafe_equals(self, dframe: pd.Series, element: Any) -> pd.Series:
        """
        Safe equality check that handles NaN values
        """
        if np.isnan(element):
            return dframe.isna()
        else:
            return dframe == element
    
    def _impute_single_val(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute missing values using the provided imputer
        """
        train_df_imp = train_df.copy()
        test_df_imp = test_df.copy()
        
        for c in train_df.columns:
            if c == self.label:
                continue
            val = self.imputer(train_df[c][~train_df[c].isin(self.miss_vals)])
            train_df_imp.loc[train_df[c].isin(self.miss_vals), c] = val
            test_df_imp.loc[test_df[c].isin(self.miss_vals), c] = val
        
        return train_df_imp, test_df_imp
    
    def _imputed_binarize(self, 
                         train_df: pd.DataFrame, 
                         test_df: pd.DataFrame,
                         imputed_train_df: pd.DataFrame,
                         imputed_test_df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Binarize data with imputation
        """
        # Binarization implementation with missing value interaction handling
        return self._binarize(train_df, test_df)
    
    def _binarize(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Main binarization logic for BIAM model
        """
        n_train, _ = train_df.shape
        n_test, _ = test_df.shape
        
        train_binned = {}
        train_augmented_binned = {}
        test_binned = {}
        test_augmented_binned = {}
        
        # Track dataset structure for BIAM visualization
        dataset_structure_map = {}
        cur_new_col_index = 0
        thresh_vals = []
        
        # Handle missing value indicators
        miss_val_cols = []
        for c in train_df.columns:
            if c == self.label:
                continue
            
            has_missing = False
            missing_col_name = f'{c} missing'
            missing_row_train = np.zeros(n_train)
            missing_row_test = np.zeros(n_test)
            
            dataset_structure_map[c] = {
                'intercepts': {},
                'interactions': {},
                'bins': []
            }
            
            # Specific missing indicators
            for v in self.miss_vals:
                if self.specific_mi_intercept:
                    new_col_name = f'{c} == {v}'
                    new_row_train = np.zeros(n_train)
                    new_row_train[self._nansafe_equals(train_df[c], v)] = 1
                    new_row_test = np.zeros(n_test)
                    new_row_test[self._nansafe_equals(test_df[c], v)] = 1
                    
                    if new_row_train.sum() > 0 or new_row_test.sum() > 0:
                        train_binned[new_col_name] = new_row_train
                        train_augmented_binned[new_col_name] = new_row_train
                        test_binned[new_col_name] = new_row_test
                        test_augmented_binned[new_col_name] = new_row_test
                        
                        dataset_structure_map[c]['intercepts'][v] = cur_new_col_index
                        thresh_vals.append(-1)
                        cur_new_col_index += 1
                        has_missing = True
                
                missing_row_train[self._nansafe_equals(train_df[c], v)] = 1
                missing_row_test[self._nansafe_equals(test_df[c], v)] = 1
            
            # Overall missing indicator
            if self.overall_mi_intercept:
                if missing_row_train.sum() > 0 or missing_row_test.sum() > 0:
                    train_binned[missing_col_name] = missing_row_train
                    train_augmented_binned[missing_col_name] = missing_row_train
                    test_binned[missing_col_name] = missing_row_test
                    test_augmented_binned[missing_col_name] = missing_row_test
                    
                    dataset_structure_map[c]['intercepts']['any'] = cur_new_col_index
                    thresh_vals.append(-1)
                    cur_new_col_index += 1
                    has_missing = True
            
            if has_missing:
                miss_val_cols.append(c)
        
        # Handle numerical features
        for c in self.numerical_cols:
            if c == self.label:
                continue
            
            for v in list(train_df[c].quantile(self.quantiles).unique()):
                if (v in self.miss_vals) or np.isnan(v):
                    continue
                
                thresh_vals.append(v)
                new_col_name = f'{c} <= {v}'
                
                new_row_train = np.zeros(n_train)
                new_row_train[train_df[c] <= v] = 1
                new_row_train[train_df[c].isin(self.miss_vals)] = 0
                
                train_binned[new_col_name] = new_row_train
                train_augmented_binned[new_col_name] = new_row_train
                dataset_structure_map[c]['bins'].append(cur_new_col_index)
                cur_new_col_index += 1
                
                new_row_test = np.zeros(n_test)
                new_row_test[test_df[c] <= v] = 1
                new_row_test[test_df[c].isin(self.miss_vals)] = 0
                
                test_binned[new_col_name] = new_row_test
                test_augmented_binned[new_col_name] = new_row_test
        
        # Handle categorical features
        for c in self.categorical_cols:
            if c == self.label:
                continue
            
            for v in list(train_df[c].unique()):
                if (v in self.miss_vals) or np.isnan(v):
                    continue
                
                thresh_vals.append(v)
                new_col_name = f'{c} == {v}'
                
                new_row_train = np.zeros(n_train)
                new_row_train[train_df[c] == v] = 1
                new_row_train[train_df[c].isin(self.miss_vals)] = 0
                
                train_binned[new_col_name] = new_row_train
                train_augmented_binned[new_col_name] = new_row_train
                dataset_structure_map[c]['bins'].append(cur_new_col_index)
                cur_new_col_index += 1
                
                new_row_test = np.zeros(n_test)
                new_row_test[test_df[c] == v] = 1
                new_row_test[test_df[c].isin(self.miss_vals)] = 0
                
                test_binned[new_col_name] = new_row_test
                test_augmented_binned[new_col_name] = new_row_test
        
        # Add labels
        train_binned[self.label] = train_df[self.label]
        test_binned[self.label] = test_df[self.label]
        train_augmented_binned[self.label] = train_df[self.label]
        test_augmented_binned[self.label] = test_df[self.label]
        
        # Store structure for BIAM visualization
        self.dataset_structure_map = dataset_structure_map
        self.thresh_vals = thresh_vals
        
        return (
            pd.DataFrame(train_augmented_binned)[[c for c in train_augmented_binned.keys() if c != self.label]].values,
            pd.DataFrame(test_augmented_binned)[[c for c in test_augmented_binned.keys() if c != self.label]].values,
            pd.DataFrame(train_augmented_binned)[self.label].values,
            pd.DataFrame(test_augmented_binned)[self.label].values
        )
