# Rust Decision Tree

A high-performance decision tree implementation in Rust, supporting both classification and regression tasks with advanced features and optimizations.

## Features

- **Classification Trees**
  - Gini impurity and entropy-based splitting
  - Support for multi-class classification
  - Majority vote for leaf predictions
  - Feature importance calculation

- **Regression Trees**
  - Mean squared error (MSE) based splitting
  - Mean value prediction for leaves
  - Feature importance calculation

- **Advanced Features**
  - **Parallel Processing**
    - Multi-threaded tree building using Rayon
    - Parallel prediction for multiple samples
    - Optimized for multi-core systems

  - **Categorical Feature Support**
    - Native handling of categorical variables
    - Automatic category encoding
    - Optimal splitting for categorical features

  - **Missing Value Handling**
    - Support for missing values in features
    - Configurable missing value indicators
    - Intelligent splitting with missing data

  - **Performance Optimizations**
    - Efficient memory usage with minimal cloning
    - Thread-safe function pointers
    - Optimized data structures for fast lookups
    - Parallel feature selection during tree building

  - **Additional Features**
    - Graphviz export for tree visualization
    - Feature importance calculation
    - Tree depth control
    - Minimum samples per split
    - Minimum impurity reduction for splits

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
decision_tree = "0.1.0"
```

## Usage

### Classification Tree

```rust
use decision_tree::{ClassificationTree, Criterion};

// Create a classification tree with Gini impurity
let mut tree = ClassificationTree::new(
    Criterion::Gini,  // or Criterion::Entropy
    2,                // min_samples_split
    1e-7,             // min_impurity
    10                // max_depth
);

// Train the tree
tree.fit(X_train, y_train);

// Make predictions
let predictions = tree.predict(X_test);
```

### Regression Tree

```rust
use decision_tree::RegressionTree;

// Create a regression tree
let mut tree = RegressionTree::new(
    2,      // min_samples_split
    1e-9,   // min_impurity
    10      // max_depth
);

// Train the tree
tree.fit(X_train, y_train);

// Make predictions
let predictions = tree.predict(X_test);
```

### Advanced Usage

```rust
use decision_tree::{ClassificationTree, FeatureInfo, FeatureType};

// Create feature information for advanced tree building
let feature_info = vec![
    FeatureInfo {
        name: "age".to_string(),
        feature_type: FeatureType::Numerical,
        categories: None,
        missing_value: Some(-1.0),
    },
    FeatureInfo {
        name: "color".to_string(),
        feature_type: FeatureType::Categorical,
        categories: Some(vec!["red".to_string(), "blue".to_string(), "green".to_string()]),
        missing_value: None,
    },
];

// Create and train a tree with advanced features
let mut tree = ClassificationTree::new(
    Criterion::Gini,
    2,
    1e-7,
    10
);

// The tree will automatically handle categorical features and missing values
tree.fit(X_train, y_train);
```

## Performance

The implementation includes several optimizations:

- **Parallel Processing**: Utilizes Rayon for parallel tree building and prediction
- **Memory Efficiency**: Minimizes cloning and uses efficient data structures
- **Thread Safety**: All operations are thread-safe and can be used in concurrent environments
- **Optimized Splitting**: Efficient algorithms for finding optimal splits

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Modules
### DecisionNode

The DecisionNode struct represents a node in the decision tree.

### DecisionTree
The DecisionTree struct provides methods for building and using decision trees.

### ClassificationTree
The ClassificationTree struct builds a decision tree for classification purposes.

