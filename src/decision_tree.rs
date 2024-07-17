use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use crate::decision_node::DecisionNode;

#[derive(Clone)]
pub struct DecisionTree {
    pub root: Option<DecisionNode>,
    pub min_samples_split: usize,
    pub min_impurity: f64,
    pub max_depth: usize,
    pub impurity_calculation: Option<Rc<dyn Fn(&Vec<f64>, &Vec<f64>, &Vec<f64>) -> f64>>,
    pub leaf_value_calculation: Option<Rc<dyn Fn(&Vec<f64>) -> f64>>,
}

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

impl DecisionTree {
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

    pub fn calculate_entropy(y: &Vec<f64>) -> f64 {
        let mut counts = HashMap::new();
        for &val in y.iter() {
            let entry = counts.entry(FloatWrapper(val)).or_insert(0);
            *entry += 1;
        }
        let total = y.len() as f64;
        counts.values()
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum()
    }

    pub fn calculate_gini_impurity(y: &Vec<f64>) -> f64 {
        let mut counts = HashMap::new();
        for &val in y.iter() {
            let entry = counts.entry(FloatWrapper(val)).or_insert(0);
            *entry += 1;
        }
        let total = y.len() as f64;
        1.0 - counts.values()
            .map(|&count| {
                let p = count as f64 / total;
                p * p
            })
            .sum::<f64>()
    }

    pub fn divide_on_feature(X: &Vec<Vec<f64>>, feature_i: usize, threshold: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut left = vec![];
        let mut right = vec![];
        for sample in X.iter() {
            if sample[feature_i] >= threshold {
                left.push(sample.clone());
            } else {
                right.push(sample.clone());
            }
        }
        (left, right)
    }

    pub fn build_tree(&mut self, X: &Vec<Vec<f64>>, y: &Vec<f64>, current_depth: usize) -> DecisionNode {
        let mut largest_impurity = 0.0;
        let mut best_criteria: Option<(usize, f64)> = None;
        let mut best_sets: Option<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> = None;

        if X.len() >= self.min_samples_split && current_depth <= self.max_depth {
            for feature_i in 0..X[0].len() {
                let mut unique_values: Vec<f64> = X.iter().map(|x| x[feature_i]).collect();
                unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                unique_values.dedup();

                for &threshold in unique_values.iter() {
                    let (X_left, X_right) = DecisionTree::divide_on_feature(X, feature_i, threshold);
                    if !X_left.is_empty() && !X_right.is_empty() {
                        let y_left: Vec<f64> = X.iter()
                            .zip(y.iter())
                            .filter(|(x, _)| x[feature_i] >= threshold)
                            .map(|(_, &val)| val)
                            .collect();
                        let y_right: Vec<f64> = X.iter()
                            .zip(y.iter())
                            .filter(|(x, _)| x[feature_i] < threshold)
                            .map(|(_, &val)| val)
                            .collect();
                        if let Some(ref impurity_calc) = self.impurity_calculation {
                            let impurity = impurity_calc(y, &y_left, &y_right);
                            if impurity > largest_impurity {
                                largest_impurity = impurity;
                                best_criteria = Some((feature_i, threshold));
                                best_sets = Some((X_left.clone(), y_left, X_right.clone(), y_right));
                            }
                        }
                    }
                }
            }
        }

        if largest_impurity > self.min_impurity {
            let (leftX, lefty, rightX, righty) = best_sets.unwrap();
            let true_branch = self.build_tree(&leftX, &lefty, current_depth + 1);
            let false_branch = self.build_tree(&rightX, &righty, current_depth + 1);
            DecisionNode::new(best_criteria.map(|(i, _t)| i), best_criteria.map(|(_, t)| t), None, Some(Rc::new(true_branch)), Some(Rc::new(false_branch)))
        } else {
            let leaf_value = self.leaf_value_calculation.as_ref().unwrap()(y);
            DecisionNode::new(None, None, Some(leaf_value), None, None)
        }
    }

    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<f64>) {
        self.root = Some(self.build_tree(&X, &y, 0));
    }

    pub fn predict_value(&self, x: &Vec<f64>, tree: &DecisionNode) -> f64 {
        if let Some(value) = tree.value {
            return value;
        }
        let feature_value = x[tree.feature_i.unwrap()];
        if feature_value < tree.threshold.unwrap() {
            self.predict_value(x, tree.false_branch.as_ref().unwrap())
        } else {
            self.predict_value(x, tree.true_branch.as_ref().unwrap())
        }
    }

    pub fn predict(&self, X: Vec<Vec<f64>>) -> Vec<f64> {
        X.iter().map(|x| self.predict_value(x, self.root.as_ref().unwrap())).collect()
    }
}
