# Decision Tree

A Rust library for building and using decision trees for classification. This library supports both Gini and Entropy criteria for building decision trees.

## Features

- Build decision trees for classification
- Support for Gini impurity and entropy-based information gain
- Flexible configuration for minimum samples split, impurity threshold, and maximum depth

## Installation

To use this library in your project, add the following line to your `Cargo.toml` file:

```toml
[dependencies]
decision_tree = { git = "https://github.com/abedinia/rust_decision_tree.git" }
```

## Usage

Here is an example of how to use the DecisionTree and ClassificationTree structs to build and use a decision tree for classification.


```rust
use decision_tree::classification_tree::ClassificationTree;

fn main() {
    let X = vec![
        vec![2.0, 3.0],
        vec![1.0, 1.0],
        vec![4.0, 5.0],
        vec![6.0, 7.0],
    ];
    let y = vec![0.0, 1.0, 0.0, 1.0];

    let mut tree = ClassificationTree::new("GINI", 2, 1e-7, usize::MAX);
    tree.fit(X.clone(), y.clone());
    let predictions = tree.predict(X);
    println!("{:?}", predictions);
}
```

## Modules
### DecisionNode

The DecisionNode struct represents a node in the decision tree.

### DecisionTree
The DecisionTree struct provides methods for building and using decision trees.

### ClassificationTree
The ClassificationTree struct builds a decision tree for classification purposes.

## Testing

To run the tests for this library, use the following command:
```bash
cargo test
```

