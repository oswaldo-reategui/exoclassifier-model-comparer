# Import the appropriate libraries
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Function lo load the data
'''
Load the dataset from the provided csv file and return the features and 
target variables (X and Y)
'''
def load_data(filename):
  dataframe = pd.read_csv(filename)
  array = dataframe.values
  X = array[:,0:37]
  Y = array[:,37]
  return X, Y

# Function to prepare models 
'''
Prepare and return a list of models for evaluation
'''
def prep_models():
  models = []
  models.append(('LR', LogisticRegression(solver='liblinear', C=100, max_iter=2000)))
  models.append(('CART', DecisionTreeClassifier()))
  models.append(('RF', RandomForestClassifier(n_estimators=100, criterion='gini')))
  return models

# Function to evaluate the models
'''
Fit the models to the provided features and target variables using cross validation.
Return a list of the mean accuracy for each model
'''
def evaluate_models(models, X, Y):
  kfold = KFold(n_splits=10, random_state=7, shuffle=True)
  results = []
  for name, model in models:
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    results.append(cv_results.mean())
  return [round(result, 2) for result in results]

# Main function call
'''
Main function to load the data, set up the models, evaluate them and return the results.
'''
def main():
  X, Y = load_data('./data/exoplanets_2018.csv')
  models = prep_models()
  results =evaluate_models(models, X, Y)
  return results

if __name__=="__main__":
  print(main())
