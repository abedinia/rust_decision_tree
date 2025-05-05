use std::fmt;

/// A node in a decision tree.
///
/// Each node can be either:
/// - An internal node with a feature index and threshold for decision making
/// - A leaf node with a value that represents the prediction
#[derive(Clone, Debug)]
pub struct DecisionNode {
    /// Index of the feature to split on (None for leaf nodes)
    pub feature_i: Option<usize>,
    
    /// Threshold value for the split (None for leaf nodes)
    pub threshold: Option<f64>,
    
    /// Prediction value (None for internal nodes, Some for leaf nodes)
    pub value: Option<f64>,
    
    /// True branch (samples where feature_i >= threshold)
    pub true_branch: Option<Box<DecisionNode>>,
    
    /// False branch (samples where feature_i < threshold)
    pub false_branch: Option<Box<DecisionNode>>,
}

impl DecisionNode {
    /// Creates a new decision node
    ///
    /// # Arguments
    /// * `feature_i` - Index of the feature to split on (None for leaf nodes)
    /// * `threshold` - Threshold value for the split (None for leaf nodes)
    /// * `value` - Prediction value (None for internal nodes, Some for leaf nodes)
    /// * `true_branch` - True branch (samples where feature_i >= threshold)
    /// * `false_branch` - False branch (samples where feature_i < threshold)
    ///
    /// # Returns
    /// A new DecisionNode
    pub fn new(
        feature_i: Option<usize>,
        threshold: Option<f64>,
        value: Option<f64>,
        true_branch: Option<Box<DecisionNode>>,
        false_branch: Option<Box<DecisionNode>>,
    ) -> Self {
        DecisionNode {
            feature_i,
            threshold,
            value,
            true_branch,
            false_branch,
        }
    }
    
    /// Creates a new leaf node with a prediction value
    ///
    /// # Arguments
    /// * `value` - The prediction value for this leaf node
    ///
    /// # Returns
    /// A new leaf node
    pub fn leaf(value: f64) -> Self {
        DecisionNode {
            feature_i: None,
            threshold: None,
            value: Some(value),
            true_branch: None,
            false_branch: None,
        }
    }
    
    /// Check if this node is a leaf node
    ///
    /// # Returns
    /// true if this is a leaf node, false otherwise
    pub fn is_leaf(&self) -> bool {
        self.value.is_some()
    }
    
    /// Returns the depth of the tree rooted at this node
    ///
    /// # Returns
    /// The depth of the tree
    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            return 0;
        }
        
        let true_depth = self.true_branch.as_ref()
            .map_or(0, |branch| branch.depth());
        
        let false_depth = self.false_branch.as_ref()
            .map_or(0, |branch| branch.depth());
        
        1 + std::cmp::max(true_depth, false_depth)
    }
}

impl fmt::Display for DecisionNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(value) = self.value {
            write!(f, "Leaf(value={})", value)
        } else if let (Some(feature_i), Some(threshold)) = (self.feature_i, self.threshold) {
            write!(f, "Node(feature={}, threshold={})", feature_i, threshold)
        } else {
            write!(f, "Invalid node")
        }
    }
}
