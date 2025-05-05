use std::fs::File;
use std::io::{self, Write};

/// Splits data into training and testing sets
/// 
/// # Arguments
/// * `X` - Feature matrix
/// * `y` - Target values
/// * `test_size` - Proportion of the data to include in the test split (0.0 to 1.0)
/// 
/// # Returns
/// Tuple containing (X_train, X_test, y_train, y_test)
pub fn train_test_split(
    X: Vec<Vec<f64>>, 
    y: Vec<f64>, 
    test_size: f64
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    assert!(test_size > 0.0 && test_size < 1.0, "test_size must be between 0 and 1");
    assert_eq!(X.len(), y.len(), "X and y must have the same length");
    
    let total_samples = X.len();
    let test_samples = (total_samples as f64 * test_size).round() as usize;
    let train_samples = total_samples - test_samples;
    
    // Create shuffled indices
    let mut indices: Vec<usize> = (0..total_samples).collect();
    // Simple shuffling - for production use, consider a proper RNG
    for i in (1..indices.len()).rev() {
        let j = i % indices.len();
        indices.swap(i, j);
    }
    
    let mut X_train = Vec::with_capacity(train_samples);
    let mut X_test = Vec::with_capacity(test_samples);
    let mut y_train = Vec::with_capacity(train_samples);
    let mut y_test = Vec::with_capacity(test_samples);
    
    for (idx, &i) in indices.iter().enumerate() {
        if idx < train_samples {
            X_train.push(X[i].clone());
            y_train.push(y[i]);
        } else {
            X_test.push(X[i].clone());
            y_test.push(y[i]);
        }
    }
    
    (X_train, X_test, y_train, y_test)
}

/// Calculate classification accuracy
/// 
/// # Arguments
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
/// 
/// # Returns
/// Accuracy as a proportion of correctly classified samples (0.0 to 1.0)
pub fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "y_true and y_pred must have the same length");
    
    if y_true.is_empty() {
        return 0.0;
    }
    
    let correct = y_true.iter()
        .zip(y_pred.iter())
        .filter(|&(&true_val, &pred_val)| (true_val - pred_val).abs() < 1e-9)
        .count();
    
    correct as f64 / y_true.len() as f64
}

/// Calculate mean squared error for regression evaluation
/// 
/// # Arguments
/// * `y_true` - Ground truth values
/// * `y_pred` - Predicted values
/// 
/// # Returns
/// Mean squared error
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "y_true and y_pred must have the same length");
    
    if y_true.is_empty() {
        return 0.0;
    }
    
    let squared_errors: f64 = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum();
    
    squared_errors / y_true.len() as f64
}

/// Export a tree to DOT format for visualization with Graphviz
/// 
/// # Arguments
/// * `node` - Root node of the tree
/// * `feature_names` - Optional vector of feature names
/// * `class_names` - Optional vector of class names (for classification trees)
/// * `filename` - Output filename
/// 
/// # Returns
/// io::Result indicating success or failure
#[allow(unused_variables)]
pub fn export_graphviz<T: AsRef<str>>(
    node: &crate::decision_node::DecisionNode,
    feature_names: Option<&[T]>,
    class_names: Option<&[T]>,
    filename: &str
) -> io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Write DOT file header
    writeln!(file, "digraph Tree {{")?;
    writeln!(file, "  node [shape=box, style=\"filled\", color=\"black\", fontname=\"helvetica\"]")?;
    writeln!(file, "  edge [fontname=\"helvetica\"]")?;
    
    // Start node counter
    let mut count = 0;
    
    // Recursive function to traverse tree and write nodes
    fn write_tree<T: AsRef<str>>(
        file: &mut File,
        node: &crate::decision_node::DecisionNode,
        parent_id: Option<usize>,
        is_left: bool,
        node_id: usize,
        feature_names: Option<&[T]>,
        class_names: Option<&[T]>,
        count: &mut usize
    ) -> io::Result<()> {
        // Node content
        let label = if let Some(value) = node.value {
            if let Some(class_names) = class_names {
                if value < class_names.len() as f64 && value >= 0.0 {
                    format!("value = {}", class_names[value as usize].as_ref())
                } else {
                    format!("value = {:.4}", value)
                }
            } else {
                format!("value = {:.4}", value)
            }
        } else if let (Some(feature_i), Some(threshold)) = (node.feature_i, node.threshold) {
            let feature_name = if let Some(feature_names) = feature_names {
                if feature_i < feature_names.len() {
                    feature_names[feature_i].as_ref().to_owned()
                } else {
                    format!("X[{}]", feature_i)
                }
            } else {
                format!("X[{}]", feature_i)
            };
            
            format!("{} <= {:.4}", feature_name, threshold)
        } else {
            "Unknown".to_string()
        };
        
        // Write current node
        writeln!(file, "  {} [label=\"{}\", fillcolor=\"#e5813900\"]", node_id, label)?;
        
        // Connect to parent if not root
        if let Some(parent) = parent_id {
            writeln!(
                file,
                "  {} -> {} [labeldistance=2.5, labelangle={}, headlabel=\"{}\"]",
                parent,
                node_id,
                if is_left { 45 } else { -45 },
                if is_left { "True" } else { "False" }
            )?;
        }
        
        // Process children recursively
        *count += 1;
        
        if let Some(ref left) = node.true_branch {
            write_tree(
                file,
                left,
                Some(node_id),
                true,
                *count,
                feature_names,
                class_names,
                count
            )?;
        }
        
        if let Some(ref right) = node.false_branch {
            *count += 1;
            write_tree(
                file,
                right,
                Some(node_id),
                false,
                *count,
                feature_names,
                class_names,
                count
            )?;
        }
        
        Ok(())
    }
    
    // Start tree writing with root node
    write_tree(&mut file, node, None, false, 0, feature_names, class_names, &mut count)?;
    
    // Close DOT file
    writeln!(file, "}}")?;
    
    Ok(())
} 