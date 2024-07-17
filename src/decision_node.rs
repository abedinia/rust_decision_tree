use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct DecisionNode {
    pub feature_i: Option<usize>,
    pub threshold: Option<f64>,
    pub value: Option<f64>,
    pub true_branch: Option<Rc<DecisionNode>>,
    pub false_branch: Option<Rc<DecisionNode>>,
}

impl DecisionNode {
    pub fn new(
        feature_i: Option<usize>,
        threshold: Option<f64>,
        value: Option<f64>,
        true_branch: Option<Rc<DecisionNode>>,
        false_branch: Option<Rc<DecisionNode>>,
    ) -> Self {
        DecisionNode {
            feature_i,
            threshold,
            value,
            true_branch,
            false_branch,
        }
    }
}
