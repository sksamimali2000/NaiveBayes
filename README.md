# 🌸 Naive Bayes Classifier from Scratch and Comparison with Sklearn Models

A simple implementation of a **Naive Bayes Classifier** from scratch applied on the **Iris dataset**,  
along with comparison to **GaussianNB** and **MultinomialNB** from scikit-learn.

---

## 🚀 Project Overview

This project demonstrates:
- Manual implementation of a Naive Bayes classifier  
- Discretization of continuous features into categorical bins  
- Calculation of probabilities using frequency counts and Laplace smoothing  
- Prediction of class labels for test data  
- Comparison of custom model with scikit-learn’s **GaussianNB** and **MultinomialNB** classifiers  
- Evaluation using accuracy, classification report, and confusion matrix

---

## ⚡ Usage

```python
import numpy as np
from sklearn import datasets, model_selection
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Data Loading
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Discretize continuous features
def makeLabelled(column):
    second_limit = column.mean()
    first_limit = 0.5 * second_limit
    third_limit = 1.5 * second_limit
    for i in range(len(column)):
        if column[i] < first_limit:
            column[i] = 0
        elif column[i] < second_limit:
            column[i] = 1
        elif column[i] < third_limit:
            column[i] = 2
        else:
            column[i] = 3
    return column

for i in range(X.shape[1]):
    X[:, i] = makeLabelled(X[:, i])

# Train/Test Split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)

# Custom Naive Bayes Implementation
def fit(X_train, Y_train):
    result = {}
    class_values = set(Y_train)
    for current_class in class_values:
        result[current_class] = {}
        result["total_data"] = len(Y_train)
        current_class_rows = (Y_train == current_class)
        X_train_current = X_train[current_class_rows]
        Y_train_current = Y_train[current_class_rows]
        num_features = X_train.shape[1]
        result[current_class]["total_count"] = len(Y_train_current)
        for j in range(1, num_features + 1):
            result[current_class][j] = {}
            all_possible_values = set(X_train[:, j - 1])
            for current_value in all_possible_values:
                result[current_class][j][current_value] = (X_train_current[:, j - 1] == current_value).sum()
    return result

def probability(dictionary, x, current_class):
    output = np.log(dictionary[current_class]["total_count"]) - np.log(dictionary["total_data"])
    num_features = len(dictionary[current_class].keys()) - 1
    for j in range(1, num_features + 1):
        xj = x[j - 1]
        count_current_class_with_value_xj = dictionary[current_class][j].get(xj, 0) + 1
        count_current_class = dictionary[current_class]["total_count"] + len(dictionary[current_class][j].keys())
        output += np.log(count_current_class_with_value_xj) - np.log(count_current_class)
    return output

def predictSinglePoint(dictionary, x):
    classes = dictionary.keys()
    best_p = -1e9
    best_class = -1
    for current_class in classes:
        if current_class == "total_data":
            continue
        p_current_class = probability(dictionary, x, current_class)
        if p_current_class > best_p:
            best_p = p_current_class
            best_class = current_class
    return best_class

def predict(dictionary, X_test):
    return [predictSinglePoint(dictionary, x) for x in X_test]

# Train and predict
dictionary = fit(X_train, Y_train)
Y_pred_custom = predict(dictionary, X_test)

# Evaluate custom model
print("Custom Naive Bayes Classifier Results:")
print(classification_report(Y_test, Y_pred_custom))
print(confusion_matrix(Y_test, Y_pred_custom))

# GaussianNB from sklearn
clf_gaussian = GaussianNB()
clf_gaussian.fit(X_train, Y_train)
Y_pred_gaussian = clf_gaussian.predict(X_test)

print("\nGaussianNB Results:")
print(classification_report(Y_test, Y_pred_gaussian))
print(confusion_matrix(Y_test, Y_pred_gaussian))

# MultinomialNB from sklearn
clf_multinomial = MultinomialNB()
clf_multinomial.fit(X_train, Y_train)
Y_pred_multinomial = clf_multinomial.predict(X_test)

print("\nMultinomialNB Results:")
print(classification_report(Y_test, Y_pred_multinomial))
print(confusion_matrix(Y_test, Y_pred_multinomial))
```


✅ Key Outcomes

Successful implementation of Naive Bayes from scratch

Categorical feature binning of continuous data

Laplace smoothing for probability estimation

Performance evaluation and comparison with scikit-learn models

⚙️ Requirements

Python >= 3.7

numpy

scikit-learn

Install dependencies using:

pip install numpy scikit-learn

📄 License

MIT License

Made with ❤️ by Sk Samim Ali
