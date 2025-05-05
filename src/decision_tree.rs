use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use crate::decision_node::DecisionNode;
use rayon::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;

/// Criterion types for decision trees
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Gini impurity - measures the frequency of misclassification
    Gini,
    /// Entropy - measures the disorder or uncertainty
    Entropy,
    /// Mean squared error - for regression trees
    MSE,
}

/// Wrapper for f64 to allow it to be used as a HashMap key
#[derive(Debug, Clone, Copy)]
pub struct FloatWrapper(pub f64);

impl PartialEq for FloatWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatWrapper {}

impl Hash for FloatWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Decision tree base structure that can be used for both classification and regression
pub struct DecisionTree {
    /// Root node of the decision tree
    pub root: Option<DecisionNode>,
    
    /// Minimum number of samples required to split a node
    pub min_samples_split: usize,
    
    /// Minimum impurity required to split a node
    pub min_impurity: f64,
    
    /// Maximum depth of the tree
    pub max_depth: usize,
    
    /// Function to calculate impurity for splitting
    pub impurity_calculation: Option<Box<dyn Fn(&[f64], &[f64], &[f64]) -> f64 + Send + Sync>>,
    
    /// Function to calculate leaf value
    pub leaf_value_calculation: Option<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
}

impl Clone for DecisionTree {
    fn clone(&self) -> Self {
        DecisionTree {
            root: self.root.clone(),
            min_samples_split: self.min_samples_split,
            min_impurity: self.min_impurity,
            max_depth: self.max_depth,
            impurity_calculation: None, // Can't clone Fn, will be set by the specific tree type
            leaf_value_calculation: None, // Can't clone Fn, will be set by the specific tree type
        }
    }
}

impl DecisionTree {
    /// Creates a new decision tree with the given parameters
    ///
    /// # Arguments
    /// * `min_samples_split` - Minimum number of samples required to split a node
    /// * `min_impurity` - Minimum impurity required to split a node
    /// * `max_depth` - Maximum depth of the tree
    ///
    /// # Returns
    /// A new DecisionTree
    pub fn new(min_samples_split: usize, min_impurity: f64, max_depth: usize) -> Self {
        DecisionTree {
            root: None,
            min_samples_split,
            min_impurity,
            max_depth,
            impurity_calculation: None,
            leaf_value_calculation: None,
        }
    }

    /// Calculate entropy for a vector of target values
    ///
    /// # Arguments
    /// * `y` - Vector of target values
    ///
    /// # Returns
    /// Entropy value
    pub fn calculate_entropy(y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        
        let mut counts = HashMap::new();
        for &val in y.iter() {
            let entry = counts.entry(FloatWrapper(val)).or_insert(0);
            *entry += 1;
        }
        
        let n = y.len() as f64;
        let entropy = -counts.values()
            .map(|count| {
                let p = *count as f64 / n;
                if p > 0.0 {
                    p * p.log2()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        
        entropy
    }

    /// Calculate Gini impurity for a vector of target values
    ///
    /// # Arguments
    /// * `y` - Vector of target values
    ///
    /// # Returns
    /// Gini impurity value
    pub fn calculate_gini_impurity(y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        
        let mut counts = HashMap::new();
        for &val in y.iter() {
            let entry = counts.entry(FloatWrapper(val)).or_insert(0);
            *entry += 1;
        }
        
        let n = y.len() as f64;
        let gini = 1.0 - counts.values()
            .map(|count| (*count as f64 / n).powi(2))
            .sum::<f64>();
        
        gini
    }

    /// Divide samples based on a feature and threshold
    ///
    /// # Arguments
    /// * `X` - Feature matrix
    /// * `y` - Target values
    /// * `feature_i` - Index of the feature to split on
    /// * `threshold` - Threshold value for the split
    ///
    /// # Returns
    /// Tuple containing (X_left, y_left, X_right, y_right)
    pub fn divide_on_feature(
        X: &[Vec<f64>], 
        y: &[f64], 
        feature_i: usize, 
        threshold: f64
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let mut X_left = Vec::new();
        let mut y_left = Vec::new();
        let mut X_right = Vec::new();
        let mut y_right = Vec::new();
        
        for (sample, &target) in X.iter().zip(y.iter()) {
            if sample[feature_i] >= threshold {
                X_left.push(sample.clone());
                y_left.push(target);
            } else {
                X_right.push(sample.clone());
                y_right.push(target);
            }
        }
        
        (X_left, y_left, X_right, y_right)
    }

    /// Build a decision tree recursively with parallel processing
    ///
    /// # Arguments
    /// * `X` - Feature matrix
    /// * `y` - Target values
    /// * `current_depth` - Current depth in the tree
    ///
    /// # Returns
    /// A decision node (the root of the subtree)
    pub fn build_tree_parallel(&self, X: &[Vec<f64>], y: &[f64], current_depth: usize) -> DecisionNode {
        let mut largest_impurity = 0.0;
        let mut best_criteria: Option<(usize, f64)> = None;
        let mut best_sets: Option<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> = None;

        // Check if we should split further
        if X.len() >= self.min_samples_split && current_depth <= self.max_depth {
            // Parallel processing for feature selection
            let results: Vec<_> = (0..X[0].len())
                .into_par_iter()
                .map(|feature_i| {
                    let mut best_impurity = 0.0;
                    let mut best_threshold = None;
                    let mut best_split = None;

                    // Get unique values for the feature
                    let mut feature_values: Vec<f64> = X.iter().map(|x| x[feature_i]).collect();
                    feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    feature_values.dedup();

                    // Try each value as a threshold
                    for &threshold in feature_values.iter() {
                        let (X_left, y_left, X_right, y_right) = 
                            Self::divide_on_feature(X, y, feature_i, threshold);
                        
                        if X_left.is_empty() || X_right.is_empty() {
                            continue;
                        }
                        
                        if let Some(ref impurity_calc) = self.impurity_calculation {
                            let impurity = impurity_calc(y, &y_left, &y_right);
                            
                            if impurity > best_impurity {
                                best_impurity = impurity;
                                best_threshold = Some(threshold);
                                best_split = Some((X_left, y_left, X_right, y_right));
                            }
                        }
                    }

                    (feature_i, best_impurity, best_threshold, best_split)
                })
                .collect();

            // Find the best split across all features
            for (feature_i, impurity, threshold, split) in results {
                if impurity > largest_impurity {
                    largest_impurity = impurity;
                    if let Some(t) = threshold {
                        best_criteria = Some((feature_i, t));
                    }
                    best_sets = split;
                }
            }
        }

        // If we found a good split
        if largest_impurity > self.min_impurity && best_sets.is_some() {
            let (X_left, y_left, X_right, y_right) = best_sets.unwrap();
            let (feature_i, threshold) = best_criteria.unwrap();
            
            // Build subtrees recursively in parallel
            let (true_branch, false_branch) = rayon::join(
                || Box::new(self.build_tree_parallel(&X_left, &y_left, current_depth + 1)),
                || Box::new(self.build_tree_parallel(&X_right, &y_right, current_depth + 1))
            );
            
            // Return decision node
            DecisionNode::new(
                Some(feature_i),
                Some(threshold),
                None,
                Some(true_branch),
                Some(false_branch)
            )
        } else {
            // Return leaf node
            let leaf_value = match &self.leaf_value_calculation {
                Some(calc) => calc(y),
                None => panic!("Leaf value calculation function not set"),
            };
            
            DecisionNode::leaf(leaf_value)
        }
    }

    /// Fit the decision tree to the training data
    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<f64>) -> &mut Self {
        assert!(!X.is_empty() && !y.is_empty(), "Training data cannot be empty");
        assert_eq!(X.len(), y.len(), "X and y must have the same length");
        assert!(self.impurity_calculation.is_some(), "Impurity calculation function must be set");
        assert!(self.leaf_value_calculation.is_some(), "Leaf value calculation function must be set");
        
        self.root = Some(self.build_tree_parallel(&X, &y, 0));
        self
    }

    /// Predict multiple samples using the decision tree
    pub fn predict(&self, X: Vec<Vec<f64>>) -> Vec<f64> {
        assert!(self.root.is_some(), "Model must be trained before prediction");
        
        let root = self.root.as_ref().unwrap();
        X.par_iter()
            .map(|x| self.predict_value(x, root))
            .collect()
    }

    /// Predict a single sample using the decision tree
    ///
    /// # Arguments
    /// * `x` - Feature vector
    /// * `node` - Current node in the tree
    ///
    /// # Returns
    /// Predicted value
    pub fn predict_value(&self, x: &[f64], node: &DecisionNode) -> f64 {
        // If leaf node, return value
        if let Some(value) = node.value {
            return value;
        }
        
        // Otherwise, continue traversing the tree
        let feature_i = node.feature_i.expect("Internal node must have feature_i");
        let threshold = node.threshold.expect("Internal node must have threshold");
        
        // Must match the condition in divide_on_feature
        if x[feature_i] >= threshold {
            self.predict_value(x, node.true_branch.as_ref().expect("True branch must exist"))
        } else {
            self.predict_value(x, node.false_branch.as_ref().expect("False branch must exist"))
        }
    }
    
    /// Get the depth of the tree
    ///
    /// # Returns
    /// Depth of the tree
    pub fn depth(&self) -> usize {
        match &self.root {
            Some(root) => root.depth(),
            None => 0,
        }
    }
    
    /// Get the importance of each feature
    ///
    /// # Arguments
    /// * `n_features` - Number of features
    ///
    /// # Returns
    /// Vector of feature importance values
    pub fn feature_importances(&self, n_features: usize) -> Vec<f64> {
        let mut importances = vec![0.0; n_features];
        let total_samples = 1.0; // Fixed value since we're using proportions
        
        // Helper function to traverse the tree and calculate importances
        fn traverse_tree(
            node: &DecisionNode,
            importances: &mut [f64],
            total_samples: f64,
            current_samples: f64,
        ) {
            if node.is_leaf() {
                return;
            }
            
            let feature_i = node.feature_i.unwrap();
            importances[feature_i] += current_samples / total_samples;
            
            // Assume equal split for simplicity
            let child_samples = current_samples / 2.0;
            
            if let Some(ref branch) = node.true_branch {
                traverse_tree(branch, importances, total_samples, child_samples);
            }
            
            if let Some(ref branch) = node.false_branch {
                traverse_tree(branch, importances, total_samples, child_samples);
            }
        }
        
        if let Some(root) = &self.root {
            traverse_tree(root, &mut importances, total_samples, total_samples);
        }
        
        // Normalize
        let sum: f64 = importances.iter().sum();
        if sum > 0.0 {
            for imp in importances.iter_mut() {
                *imp /= sum;
            }
        } else {
            // If we have a perfect fit with just a root node (no splits), 
            // assign equal importance to all features
            for imp in importances.iter_mut() {
                *imp = 1.0 / n_features as f64;
            }
        }
        
        importances
    }

    /// Build a decision tree with support for categorical features and missing values
    pub fn build_tree_advanced(
        &self,
        X: &[Vec<f64>],
        y: &[f64],
        current_depth: usize,
        feature_info: &[FeatureInfo]
    ) -> DecisionNode {
        let mut largest_impurity = 0.0;
        let mut best_criteria: Option<(usize, f64)> = None;
        let mut best_sets: Option<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> = None;

        if X.len() >= self.min_samples_split && current_depth <= self.max_depth {
            for (feature_i, info) in feature_info.iter().enumerate() {
                match info.feature_type {
                    FeatureType::Numerical => {
                        // Handle numerical features with missing values
                        let (threshold, impurity, split) = self.find_best_numerical_split(
                            X, y, feature_i, info.missing_value
                        );
                        
                        if impurity > largest_impurity {
                            largest_impurity = impurity;
                            best_criteria = Some((feature_i, threshold));
                            best_sets = split;
                        }
                    },
                    FeatureType::Categorical => {
                        // Handle categorical features
                        if let Some(categories) = &info.categories {
                            let (category, impurity, split) = self.find_best_categorical_split(
                                X, y, feature_i, categories
                            );
                            
                            if impurity > largest_impurity {
                                largest_impurity = impurity;
                                best_criteria = Some((feature_i, category as f64));
                                best_sets = split;
                            }
                        }
                    }
                }
            }
        }

        if largest_impurity > self.min_impurity && best_sets.is_some() {
            let (X_left, y_left, X_right, y_right) = best_sets.unwrap();
            let (feature_i, threshold) = best_criteria.unwrap();
            
            let true_branch = Box::new(self.build_tree_advanced(
                &X_left, &y_left, current_depth + 1, feature_info
            ));
            let false_branch = Box::new(self.build_tree_advanced(
                &X_right, &y_right, current_depth + 1, feature_info
            ));
            
            DecisionNode::new(
                Some(feature_i),
                Some(threshold),
                None,
                Some(true_branch),
                Some(false_branch)
            )
        } else {
            let leaf_value = match &self.leaf_value_calculation {
                Some(calc) => calc(y),
                None => panic!("Leaf value calculation function not set"),
            };
            
            DecisionNode::leaf(leaf_value)
        }
    }

    /// Find the best split for a numerical feature
    fn find_best_numerical_split(
        &self,
        X: &[Vec<f64>],
        y: &[f64],
        feature_i: usize,
        missing_value: Option<f64>
    ) -> (f64, f64, Option<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)>) {
        let mut best_impurity = 0.0;
        let mut best_threshold = 0.0;
        let mut best_split = None;

        // Get unique values for the feature, excluding missing values
        let mut feature_values: Vec<f64> = X.iter()
            .filter_map(|x| {
                let val = x[feature_i];
                if missing_value.map_or(true, |mv| val != mv) {
                    Some(val)
                } else {
                    None
                }
            })
            .collect();
        
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        feature_values.dedup();

        for &threshold in feature_values.iter() {
            let (X_left, y_left, X_right, y_right) = Self::divide_on_feature(X, y, feature_i, threshold);
            
            if X_left.is_empty() || X_right.is_empty() {
                continue;
            }
            
            if let Some(ref impurity_calc) = self.impurity_calculation {
                let impurity = impurity_calc(y, &y_left, &y_right);
                
                if impurity > best_impurity {
                    best_impurity = impurity;
                    best_threshold = threshold;
                    best_split = Some((X_left, y_left, X_right, y_right));
                }
            }
        }

        (best_threshold, best_impurity, best_split)
    }

    /// Find the best split for a categorical feature
    fn find_best_categorical_split(
        &self,
        X: &[Vec<f64>],
        y: &[f64],
        feature_i: usize,
        categories: &[String]
    ) -> (usize, f64, Option<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)>) {
        let mut best_impurity = 0.0;
        let mut best_category = 0;
        let mut best_split = None;

        for (category_idx, _) in categories.iter().enumerate() {
            let (X_left, y_left, X_right, y_right) = Self::divide_on_categorical_feature(
                X, y, feature_i, category_idx
            );
            
            if X_left.is_empty() || X_right.is_empty() {
                continue;
            }
            
            if let Some(ref impurity_calc) = self.impurity_calculation {
                let impurity = impurity_calc(y, &y_left, &y_right);
                
                if impurity > best_impurity {
                    best_impurity = impurity;
                    best_category = category_idx;
                    best_split = Some((X_left, y_left, X_right, y_right));
                }
            }
        }

        (best_category, best_impurity, best_split)
    }

    /// Divide dataset based on categorical feature
    fn divide_on_categorical_feature(
        X: &[Vec<f64>],
        y: &[f64],
        feature_i: usize,
        category_idx: usize
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let mut X_left = Vec::new();
        let mut y_left = Vec::new();
        let mut X_right = Vec::new();
        let mut y_right = Vec::new();

        for (i, x) in X.iter().enumerate() {
            if x[feature_i] == category_idx as f64 {
                X_left.push(x.clone());
                y_left.push(y[i]);
            } else {
                X_right.push(x.clone());
                y_right.push(y[i]);
            }
        }

        (X_left, y_left, X_right, y_right)
    }
}

#[derive(Debug, Clone)]
pub struct FeatureInfo {
    pub name: String,
    pub feature_type: FeatureType,
    pub categories: Option<Vec<String>>,
    pub missing_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureType {
    Numerical,
    Categorical,
}
