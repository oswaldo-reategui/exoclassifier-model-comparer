# Machine Learning Model Evaluation

This repository contains a Python script for evaluating various machine learning models. It's part of a project for a Master's course in Machine Learning, focusing on model selection and performance analysis using a dataset on exoplanets.

## Overview

The script demonstrates the process of loading a dataset, preparing different machine learning models, and evaluating their performance using cross-validation. The models included are Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.

## Features

Data Loading: Function to load data from a CSV file into feature and target variables.
Model Preparation: Setting up different machine learning models for comparison.
Model Evaluation: Using K-Fold cross-validation to evaluate model accuracy.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn

## Usage

To use this script, ensure you have the required Python version and libraries installed. Place your dataset (CSV file) in the 'data' directory and specify its path in the load_data function. Run the script to see the evaluation results of the models.

## How to Run

`python model_evaluation.py`

This will print the average accuracy scores of the evaluated models to the console.

## Dataset

The dataset used is 'exoplanets_2018.csv', which should be structured with 37 feature columns and 1 target column.

## Contributing

Feel free to fork this repository and submit pull requests for enhancements. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
