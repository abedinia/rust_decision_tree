# Contributing to Rust Decision Tree

Thank you for your interest in contributing to the Rust Decision Tree library! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

1. **Prerequisites**
   - Rust 1.70 or later
   - Cargo
   - Clippy (for linting)
   - Rustfmt (for code formatting)

2. **Getting Started**
   ```bash
   git clone https://github.com/abedinia/rust_decision_tree.git
   cd rust_decision_tree
   cargo build
   ```

3. **Development Tools**
   - Install development dependencies:
     ```bash
     rustup component add clippy rustfmt
     cargo install cargo-audit cargo-tarpaulin
     ```

## Project Structure

```
rust_decision_tree/
├── src/
│   ├── decision_tree.rs      # Core decision tree implementation
│   ├── classification_tree.rs # Classification tree implementation
│   ├── regression_tree.rs    # Regression tree implementation
│   ├── decision_node.rs      # Tree node implementation
│   └── utils.rs              # Utility functions
├── tests/
│   └── tree_tests.rs         # Unit tests
├── examples/                 # Example applications
├── benches/                  # Benchmark tests
└── docs/                    # Documentation
```

## Coding Standards

1. **Code Style**
   - Follow Rust's official style guide
   - Use `cargo fmt` to format code
   - Run `cargo clippy` to check for common issues

2. **Naming Conventions**
   - Use snake_case for variables and functions
   - Use PascalCase for types and traits
   - Use SCREAMING_SNAKE_CASE for constants

3. **Documentation**
   - Document all public APIs
   - Include examples in doc comments
   - Use `#![deny(missing_docs)]` to enforce documentation

4. **Error Handling**
   - Use custom error types
   - Provide meaningful error messages
   - Use the `thiserror` crate for error definitions

## Testing Guidelines

1. **Unit Tests**
   - Write tests for all public functions
   - Use property-based testing where appropriate
   - Include edge cases and error conditions

2. **Integration Tests**
   - Test the library as a whole
   - Include real-world use cases
   - Test performance characteristics

3. **Benchmarks**
   - Include benchmarks for critical operations
   - Compare against other implementations
   - Document performance characteristics

4. **Code Coverage**
   - Maintain high test coverage
   - Use `cargo tarpaulin` for coverage reports
   - Aim for at least 90% coverage

## Documentation

1. **API Documentation**
   - Use `///` for public API documentation
   - Include examples in doc tests
   - Document all parameters and return values

2. **Examples**
   - Provide working examples for all features
   - Include both simple and advanced use cases
   - Document performance considerations

3. **Tutorials**
   - Create step-by-step tutorials
   - Include real-world use cases
   - Provide performance optimization guides

## Performance Considerations

1. **Optimization Guidelines**
   - Profile before optimizing
   - Use appropriate data structures
   - Minimize allocations
   - Leverage parallel processing

2. **Memory Management**
   - Use appropriate lifetimes
   - Minimize cloning
   - Use references where possible
   - Implement proper cleanup

3. **Concurrency**
   - Ensure thread safety
   - Use appropriate synchronization
   - Document thread safety guarantees

## Pull Request Process

1. **Before Submitting**
   - Run all tests and benchmarks
   - Update documentation
   - Ensure code style compliance
   - Check for performance regressions

2. **PR Requirements**
   - Clear description of changes
   - Related issue number
   - Tests for new features
   - Documentation updates
   - Performance impact analysis

3. **Review Process**
   - Code review by maintainers
   - CI checks must pass
   - Performance benchmarks must pass
   - Documentation must be complete

## Release Process

1. **Versioning**
   - Follow semantic versioning
   - Update CHANGELOG.md
   - Update version in Cargo.toml

2. **Release Checklist**
   - Run all tests and benchmarks
   - Update documentation
   - Create release notes
   - Tag the release
   - Publish to crates.io

## Getting Help

- Open an issue for questions
- Join our Discord channel
- Check the documentation
- Review existing issues

## Acknowledgments

Thank you to all contributors who have helped make this project better! 