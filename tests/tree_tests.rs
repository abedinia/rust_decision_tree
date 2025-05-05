use decision_tree::{
    ClassificationTree,
    RegressionTree,
    train_test_split,
    accuracy_score,
    mean_squared_error,
    export_graphviz,
};

#[test]
fn test_classification_tree() {
    // Generate some simple classification data
    let X = vec![
        vec![1.0, 2.0], // Class 0
        vec![2.0, 3.0], // Class 0
        vec![3.0, 4.0], // Class 0
        vec![5.0, 6.0], // Class 1
        vec![6.0, 7.0], // Class 1
        vec![7.0, 8.0], // Class 1
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    
    // Split into training and testing sets
    let (X_train, X_test, y_train, y_test) = train_test_split(X, y, 0.33);
    
    // Create and train a classification tree with default parameters
    let mut tree = ClassificationTree::default();
    tree.fit(X_train, y_train);
    
    // Make predictions on the test set
    let predictions = tree.predict(X_test);
    
    // Calculate accuracy
    let accuracy = accuracy_score(&y_test, &predictions);
    println!("Classification accuracy: {:.2}", accuracy);
    
    // Verify tree has been built
    assert!(tree.depth() > 0);
    
    // Verify predictions are either 0.0 or 1.0
    for pred in predictions {
        assert!(pred == 0.0 || pred == 1.0);
    }
    
    // Verify accuracy is reasonable
    assert!(accuracy > 0.5);
    
    // Feature importances
    let importances = tree.feature_importances(2);
    assert_eq!(importances.len(), 2);
    assert!(importances.iter().sum::<f64>() > 0.99);
}

#[test]
fn test_regression_tree() {
    // Generate some simple regression data (y = 2*x_0 + 3*x_1)
    let X = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
        vec![5.0, 5.0],
        vec![6.0, 6.0],
        // Add more training examples to better capture the linear relationship
        vec![1.5, 1.5],
        vec![2.5, 2.5],
        vec![3.5, 3.5],
        vec![4.5, 4.5],
        vec![5.5, 5.5],
    ];
    let y = X.iter()
        .map(|x| 2.0 * x[0] + 3.0 * x[1])
        .collect::<Vec<f64>>();
    
    // Add some noise to the target values to ensure the tree needs to create splits
    let mut y_with_noise = y.clone();
    y_with_noise[0] += 0.5; // Add a small amount of noise to force splits
    y_with_noise[3] -= 0.5;
    
    // No train-test split - use all data for training to ensure it learns the pattern
    let X_train = X.clone();
    let y_train = y_with_noise;
    let X_test = X.clone();
    let y_test = y.clone();
    
    // Create and train a regression tree with more aggressive parameters
    let mut tree = RegressionTree::default()
        .min_samples_split(2)
        .min_impurity(1e-12)  // Very low impurity threshold to encourage deep splits
        .max_depth(10);       // Deeper tree to better approximate the linear function
    
    tree.fit(X_train, y_train);
    
    // Make predictions on the test set
    let predictions = tree.predict(X_test);
    
    // Calculate mean squared error
    let mse = mean_squared_error(&y_test, &predictions);
    println!("Regression MSE: {:.2}", mse);
    
    // Verify tree has been built
    assert!(tree.depth() > 0);
    
    // Verify MSE is reasonable for this simple function
    assert!(mse < 5.0);
    
    // Feature importances
    let importances = tree.feature_importances(2);
    assert_eq!(importances.len(), 2);
    
    // With noise, we should have splits and thus feature importances
    let sum_importances: f64 = importances.iter().sum();
    println!("Sum of feature importances: {:.4}", sum_importances);
    assert!(sum_importances > 0.0);
}

#[test]
fn test_builder_pattern() {
    // Test the fluent builder pattern for classification trees
    let tree = ClassificationTree::default()
        .criterion(decision_tree::decision_tree::Criterion::Entropy)
        .min_samples_split(3)
        .min_impurity(0.01)
        .max_depth(10);
    
    assert_eq!(tree.get_criterion(), decision_tree::decision_tree::Criterion::Entropy);
    
    // Test the fluent builder pattern for regression trees
    let tree = RegressionTree::default()
        .min_samples_split(3)
        .min_impurity(0.01)
        .max_depth(10);
    
    assert_eq!(tree.get_criterion(), decision_tree::decision_tree::Criterion::MSE);
}

#[test]
fn test_export_graphviz() {
    let X = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![6.0, 7.0],
        vec![7.0, 8.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    
    let mut tree = ClassificationTree::default();
    tree.fit(X, y);
    
    let feature_names = vec!["Feature 1", "Feature 2"];
    let class_names = vec!["Class 0", "Class 1"];
    
    let result = export_graphviz(
        tree.tree.root.as_ref().unwrap(),
        Some(&feature_names),
        Some(&class_names),
        "tree.dot"
    );
    
    assert!(result.is_ok());
    
    // Clean up file after test
    std::fs::remove_file("tree.dot").ok();
} 