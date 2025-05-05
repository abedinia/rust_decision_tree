use std::collections::HashMap;
use std::fmt;
use crate::decision_tree::{DecisionTree, FloatWrapper, Criterion};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Classification tree for predicting discrete class labels
pub struct ClassificationTree {
    /// Inner decision tree
    pub tree: DecisionTree,
    /// The criterion used for splitting (Gini or Entropy)
    criterion: Criterion,
}

impl Default for ClassificationTree {
    /// Creates a new classification tree with default parameters:
    /// - Gini criterion
    /// - min_samples_split = 2
    /// - min_impurity = 1e-7
    /// - max_depth = usize::MAX
    fn default() -> Self {
        Self::new(
            Criterion::Gini, 
            2, 
            1e-7, 
            usize::MAX
        )
    }
}

impl ClassificationTree {
    /// Creates a new classification tree with the given parameters
    ///
    /// # Arguments
    /// * `criterion` - The criterion to use for splitting (Gini or Entropy)
    /// * `min_samples_split` - Minimum number of samples required to split a node
    /// * `min_impurity` - Minimum impurity required to split a node
    /// * `max_depth` - Maximum depth of the tree
    ///
    /// # Returns
    /// A new ClassificationTree
    pub fn new(
        criterion: Criterion,
        min_samples_split: usize,
        min_impurity: f64,
        max_depth: usize
    ) -> Self {
        let impurity_calc = match criterion {
            Criterion::Gini => Arc::new(|y: &[f64], y_left: &[f64], y_right: &[f64]| {
                let gini_parent = DecisionTree::calculate_gini_impurity(y);
                let gini_left = DecisionTree::calculate_gini_impurity(y_left);
                let gini_right = DecisionTree::calculate_gini_impurity(y_right);
                let n_left = y_left.len() as f64;
                let n_right = y_right.len() as f64;
                let n_total = (n_left + n_right) as f64;
                gini_parent - (n_left / n_total) * gini_left - (n_right / n_total) * gini_right
            }) as Arc<dyn Fn(&[f64], &[f64], &[f64]) -> f64 + Send + Sync>,
            Criterion::Entropy => Arc::new(|y: &[f64], y_left: &[f64], y_right: &[f64]| {
                let entropy_parent = DecisionTree::calculate_entropy(y);
                let entropy_left = DecisionTree::calculate_entropy(y_left);
                let entropy_right = DecisionTree::calculate_entropy(y_right);
                let n_left = y_left.len() as f64;
                let n_right = y_right.len() as f64;
                let n_total = (n_left + n_right) as f64;
                entropy_parent - (n_left / n_total) * entropy_left - (n_right / n_total) * entropy_right
            }) as Arc<dyn Fn(&[f64], &[f64], &[f64]) -> f64 + Send + Sync>,
            _ => panic!("Invalid criterion for classification tree"),
        };

        let leaf_calc = Arc::new(|y: &[f64]| {
            let mut counts = HashMap::new();
            for &val in y.iter() {
                let entry = counts.entry(FloatWrapper(val)).or_insert(0);
                *entry += 1;
            }
            counts.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(val, _)| val.0)
                .unwrap_or(0.0)
        }) as Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

        ClassificationTree {
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
            criterion,
        }
    }
    
    /// Set the criterion for splitting
    ///
    /// # Arguments
    /// * `criterion` - The criterion to use (Gini or Entropy)
    ///
    /// # Returns
    /// Self for method chaining
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        
        match criterion {
            Criterion::Gini => {
                self.tree.impurity_calculation = Some(Box::new(Self::gini_calculate_information_gain));
            },
            Criterion::Entropy => {
                self.tree.impurity_calculation = Some(Box::new(Self::calculate_information_gain));
            },
            _ => panic!("Invalid criterion for classification tree: only Gini or Entropy are supported"),
        }
        
        // Set the leaf value calculation function
        self.tree.leaf_value_calculation = Some(Box::new(Self::majority_vote));
        self
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
    
    /// Set the minimum impurity required to split a node
    ///
    /// # Arguments
    /// * `min_impurity` - Minimum impurity required to split a node
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

    /// Calculate information gain using entropy
    ///
    /// # Arguments
    /// * `y` - Target values
    /// * `y_left` - Target values in the left branch
    /// * `y_right` - Target values in the right branch
    ///
    /// # Returns
    /// Information gain
    fn calculate_information_gain(y: &[f64], y_left: &[f64], y_right: &[f64]) -> f64 {
        let p = y_left.len() as f64 / y.len() as f64;
        let entropy = DecisionTree::calculate_entropy(y);
        entropy - p * DecisionTree::calculate_entropy(y_left) - (1.0 - p) * DecisionTree::calculate_entropy(y_right)
    }

    /// Calculate majority vote for leaf value
    ///
    /// # Arguments
    /// * `y` - Target values
    ///
    /// # Returns
    /// Most common class label
    fn majority_vote(y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        
        let mut counts = HashMap::new();
        for &val in y.iter() {
            let entry = counts.entry(FloatWrapper(val)).or_insert(0);
            *entry += 1;
        }
        counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| val.0)
            .unwrap()
    }

    /// Calculate information gain using Gini impurity
    ///
    /// # Arguments
    /// * `y` - Target values
    /// * `y_left` - Target values in the left branch
    /// * `y_right` - Target values in the right branch
    ///
    /// # Returns
    /// Information gain
    fn gini_calculate_information_gain(y: &[f64], y_left: &[f64], y_right: &[f64]) -> f64 {
        let original_impurity = DecisionTree::calculate_gini_impurity(y);
        
        // Skip computation if either side is empty
        if y_left.is_empty() || y_right.is_empty() {
            return 0.0;
        }
        
        let impurity_left = DecisionTree::calculate_gini_impurity(y_left);
        let impurity_right = DecisionTree::calculate_gini_impurity(y_right);
        let p_left = y_left.len() as f64 / y.len() as f64;
        let p_right = y_right.len() as f64 / y.len() as f64;
        
        original_impurity - (p_left * impurity_left + p_right * impurity_right)
    }

    /// Fit the classification tree to the training data
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

    /// Predict class labels for samples
    ///
    /// # Arguments
    /// * `X` - Feature matrix
    ///
    /// # Returns
    /// Vector of predicted class labels
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
    
    /// Get the criterion used for splitting
    ///
    /// # Returns
    /// The criterion used
    pub fn get_criterion(&self) -> Criterion {
        self.criterion
    }
}

impl fmt::Display for ClassificationTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClassificationTree(criterion={:?}, depth={})", self.criterion, self.depth())
    }
}

impl Clone for ClassificationTree {
    fn clone(&self) -> Self {
        let mut new_tree = self.tree.clone();
        new_tree.impurity_calculation = match self.criterion {
            Criterion::Gini => Some(Box::new(Self::gini_calculate_information_gain)),
            Criterion::Entropy => Some(Box::new(Self::calculate_information_gain)),
            _ => panic!("Invalid criterion for classification tree: only Gini or Entropy are supported"),
        };
        new_tree.leaf_value_calculation = Some(Box::new(Self::majority_vote));
        
        ClassificationTree { 
            tree: new_tree,
            criterion: self.criterion,
        }
    }
}
