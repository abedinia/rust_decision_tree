use std::fmt;
use crate::decision_tree::{DecisionTree, Criterion};
use std::sync::Arc;

/// Regression tree for predicting continuous values
pub struct RegressionTree {
    /// Inner decision tree
    pub tree: DecisionTree,
}

impl Default for RegressionTree {
    /// Creates a new regression tree with default parameters:
    /// - MSE criterion
    /// - min_samples_split = 2
    /// - min_impurity = 1e-9
    /// - max_depth = usize::MAX
    fn default() -> Self {
        Self::new(2, 1e-9, usize::MAX)
    }
}

impl RegressionTree {
    /// Creates a new regression tree with the given parameters
    ///
    /// # Arguments
    /// * `min_samples_split` - Minimum number of samples required to split a node
    /// * `min_impurity` - Minimum impurity required to split a node
    /// * `max_depth` - Maximum depth of the tree
    ///
    /// # Returns
    /// A new RegressionTree
    pub fn new(min_samples_split: usize, min_impurity: f64, max_depth: usize) -> Self {
        let impurity_calc = Arc::new(|y: &[f64], y_left: &[f64], y_right: &[f64]| {
            let mse_parent = Self::calculate_mse(y);
            let mse_left = Self::calculate_mse(y_left);
            let mse_right = Self::calculate_mse(y_right);
            let n_left = y_left.len() as f64;
            let n_right = y_right.len() as f64;
            let n_total = (n_left + n_right) as f64;
            mse_parent - (n_left / n_total) * mse_left - (n_right / n_total) * mse_right
        }) as Arc<dyn Fn(&[f64], &[f64], &[f64]) -> f64 + Send + Sync>;

        let leaf_calc = Arc::new(|y: &[f64]| {
            y.iter().sum::<f64>() / y.len() as f64
        }) as Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

        RegressionTree {
            tree: DecisionTree {
                root: None,
                min_samples_split,
                min_impurity,
                max_depth,
                impurity_calculation: Some(Box::new(move |y, y_left, y_right| {
                    impurity_calc(y, y_left, y_right)
                })),
                leaf_value_calculation: Some(Box::new(move |y| {
                    leaf_calc(y)
                })),
            },
        }
    }
    
    /// Set the minimum number of samples required to split a node
    ///
    /// # Arguments
    /// * `min_samples_split` - Minimum number of samples required to split a node
    ///
    /// # Returns
    /// Self for method chaining
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.tree.min_samples_split = min_samples_split;
        self
    }
    
    /// Set the minimum variance reduction required to split a node
    ///
    /// # Arguments
    /// * `min_impurity` - Minimum variance reduction required to split a node
    ///
    /// # Returns
    /// Self for method chaining
    pub fn min_impurity(mut self, min_impurity: f64) -> Self {
        self.tree.min_impurity = min_impurity;
        self
    }
    
    /// Set the maximum depth of the tree
    ///
    /// # Arguments
    /// * `max_depth` - Maximum depth of the tree
    ///
    /// # Returns
    /// Self for method chaining
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.tree.max_depth = max_depth;
        self
    }

    /// Calculate mean squared error for a vector of target values
    ///
    /// # Arguments
    /// * `y` - Vector of target values
    ///
    /// # Returns
    /// Mean squared error
    pub fn calculate_mse(y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        y.iter()
            .map(|&val| (val - mean).powi(2))
            .sum::<f64>() / y.len() as f64
    }

    /// Fit the regression tree to the training data
    ///
    /// # Arguments
    /// * `X` - Feature matrix
    /// * `y` - Target values
    ///
    /// # Returns
    /// Reference to self for method chaining
    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<f64>) -> &mut Self {
        self.tree.fit(X, y);
        self
    }

    /// Predict multiple samples using the regression tree
    ///
    /// # Arguments
    /// * `X` - Feature matrix
    ///
    /// # Returns
    /// Vector of predicted values
    pub fn predict(&self, X: Vec<Vec<f64>>) -> Vec<f64> {
        self.tree.predict(X)
    }
    
    /// Get the depth of the tree
    ///
    /// # Returns
    /// Depth of the tree
    pub fn depth(&self) -> usize {
        self.tree.depth()
    }
    
    /// Get the importance of each feature
    ///
    /// # Arguments
    /// * `n_features` - Number of features
    ///
    /// # Returns
    /// Vector of feature importance values
    pub fn feature_importances(&self, n_features: usize) -> Vec<f64> {
        self.tree.feature_importances(n_features)
    }
    
    /// Get the criterion used (always MSE for regression)
    ///
    /// # Returns
    /// MSE criterion
    pub fn get_criterion(&self) -> Criterion {
        Criterion::MSE
    }
}

impl fmt::Display for RegressionTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegressionTree(criterion=MSE, depth={})", self.depth())
    }
}

impl Clone for RegressionTree {
    fn clone(&self) -> Self {
        RegressionTree {
            tree: self.tree.clone(),
        }
    }
} 