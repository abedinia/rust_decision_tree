use std::collections::HashMap;
use std::rc::Rc;
use crate::decision_tree::{DecisionTree, FloatWrapper};

pub struct ClassificationTree {
    pub tree: DecisionTree,
}

impl ClassificationTree {
    pub fn new(strategy: &str, min_samples_split: usize, min_impurity: f64, max_depth: usize) -> Self {
        let mut tree = DecisionTree::new(min_samples_split, min_impurity, max_depth);
        if strategy == "GINI" {
            tree.impurity_calculation = Some(Rc::new(ClassificationTree::gini_calculate_information_gain));
        } else if strategy == "ENTROPY" {
            tree.impurity_calculation = Some(Rc::new(ClassificationTree::calculate_information_gain));
        }
        tree.leaf_value_calculation = Some(Rc::new(ClassificationTree::majority_vote));
        ClassificationTree { tree }
    }

    fn calculate_information_gain(y: &Vec<f64>, y_left: &Vec<f64>, y_right: &Vec<f64>) -> f64 {
        let p = y_left.len() as f64 / y.len() as f64;
        let entropy = DecisionTree::calculate_entropy(y);
        entropy - p * DecisionTree::calculate_entropy(y_left) - (1.0 - p) * DecisionTree::calculate_entropy(y_right)
    }

    fn majority_vote(y: &Vec<f64>) -> f64 {
        let mut counts = HashMap::new();
        for &val in y.iter() {
            let entry = counts.entry(FloatWrapper(val)).or_insert(0);
            *entry += 1;
        }
        counts.into_iter().max_by_key(|&(_, count)| count).map(|(val, _)| val.0).unwrap()
    }

    fn gini_calculate_information_gain(y: &Vec<f64>, y_left: &Vec<f64>, y_right: &Vec<f64>) -> f64 {
        let original_impurity = DecisionTree::calculate_gini_impurity(y);
        let impurity_left = DecisionTree::calculate_gini_impurity(y_left);
        let impurity_right = DecisionTree::calculate_gini_impurity(y_right);
        let p_left = y_left.len() as f64 / y.len() as f64;
        let p_right = y_right.len() as f64 / y.len() as f64;
        original_impurity - (p_left * impurity_left + p_right * impurity_right)
    }

    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<f64>) {
        self.tree.fit(X, y);
    }

    pub fn predict(&self, X: Vec<Vec<f64>>) -> Vec<f64> {
        self.tree.predict(X)
    }
}
