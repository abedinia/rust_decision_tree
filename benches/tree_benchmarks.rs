use criterion::{black_box, criterion_group, criterion_main, Criterion};
use decision_tree::{ClassificationTree, RegressionTree};

fn create_large_dataset(size: usize, features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut X = Vec::with_capacity(size);
    let mut y = Vec::with_capacity(size);
    
    for i in 0..size {
        let mut row = Vec::with_capacity(features);
        for j in 0..features {
            // Simple deterministic data generation
            row.push((i * j) as f64 % 10.0);
        }
        X.push(row);
        
        // For classification, alternate between two classes
        y.push((i % 2) as f64);
    }
    
    (X, y)
}

fn benchmark_classification_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("Classification Tree");
    
    // Small dataset benchmark
    let (X_small, y_small) = create_large_dataset(100, 5);
    group.bench_function("fit_small_dataset", |b| {
        b.iter(|| {
            let mut tree = ClassificationTree::default();
            tree.fit(black_box(X_small.clone()), black_box(y_small.clone()));
        });
    });
    
    // Medium dataset benchmark
    let (X_medium, y_medium) = create_large_dataset(1000, 10);
    group.bench_function("fit_medium_dataset", |b| {
        b.iter(|| {
            let mut tree = ClassificationTree::default();
            tree.fit(black_box(X_medium.clone()), black_box(y_medium.clone()));
        });
    });
    
    // Prediction benchmark
    let (X_train, y_train) = create_large_dataset(500, 10);
    let (X_test, _) = create_large_dataset(100, 10);
    
    let mut tree = ClassificationTree::default();
    tree.fit(X_train, y_train);
    
    group.bench_function("predict", |b| {
        b.iter(|| {
            tree.predict(black_box(X_test.clone()));
        });
    });
    
    group.finish();
}

fn benchmark_regression_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("Regression Tree");
    
    // Generate regression data (y = sum of features)
    let mut X = Vec::with_capacity(1000);
    let mut y = Vec::with_capacity(1000);
    
    for i in 0..1000 {
        let features = 10;
        let mut row = Vec::with_capacity(features);
        let mut sum = 0.0;
        
        for j in 0..features {
            let val = (i * j) as f64 % 10.0;
            row.push(val);
            sum += val;
        }
        
        X.push(row);
        y.push(sum);
    }
    
    // Training benchmark
    group.bench_function("fit", |b| {
        b.iter(|| {
            let mut tree = RegressionTree::default();
            tree.fit(black_box(X.clone()), black_box(y.clone()));
        });
    });
    
    // Prediction benchmark
    let mut tree = RegressionTree::default();
    tree.fit(X.clone(), y);
    
    let test_X = X[0..100].to_vec();
    
    group.bench_function("predict", |b| {
        b.iter(|| {
            tree.predict(black_box(test_X.clone()));
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_classification_tree,
    benchmark_regression_tree
);
criterion_main!(benches); 