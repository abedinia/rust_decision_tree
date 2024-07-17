use decision_tree::classification_tree::ClassificationTree;

#[test]
fn test_classification_tree() {
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
    assert_eq!(predictions, vec![0.0, 1.0, 0.0, 1.0]);
}
