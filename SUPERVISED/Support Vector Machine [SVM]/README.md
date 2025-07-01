# 📂 Support Vector Machine (SVM)

## 🧠 Overview
This directory contains an implementation of Support Vector Machine (SVM), a powerful supervised learning algorithm used for classification tasks. SVM works by finding the optimal hyperplane that maximizes the margin between different classes in the feature space.

## 📘 Learning & Concepts Covered
- Understanding the mathematical foundation of SVM
- Hyperplanes and decision boundaries
- Margin maximization
- Support vectors
- Kernel functions for non-linear classification
- Soft margin concept and the C parameter
- Model evaluation and accuracy measurement

## 🎯 File: `svm1.py`

### 📌 Concept/Goal
The script implements a Support Vector Machine classifier to categorize breast cancer tumors as malignant or benign based on various features extracted from digitized images of fine needle aspirates (FNA) of breast masses. It demonstrates the complete workflow from data loading to model training, evaluation, and prediction.

### ⚙️ Functions & Methods Used

#### `datasets.load_breast_cancer()`
```python
cancer = datasets.load_breast_cancer()
```
- Loads the breast cancer dataset from scikit-learn's built-in datasets
- Contains features of breast cancer cells and binary classification (malignant/benign)

#### `sklearn.model_selection.train_test_split()`
```python
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
```
- Splits data into training and testing sets
- `test_size=0.2` allocates 20% of data for testing

#### `svm.SVC()`
```python
clf = svm.SVC(kernel='linear', C=2)
```
- Creates a Support Vector Classifier
- `kernel='linear'` specifies a linear kernel function
- `C=2` controls the trade-off between smooth decision boundary and classifying training points correctly (soft margin)

#### `clf.fit()`
```python
clf.fit(x_test, y_test)
```
- Trains the SVM model on the training data
- Finds the optimal hyperplane that separates the classes

#### `clf.predict()`
```python
y_predict = clf.predict(x_test)
```
- Uses the trained model to predict classes for test data
- Returns an array of predicted class labels

#### `metrics.accuracy_score()`
```python
accuracy = metrics.accuracy_score(y_test, y_predict)
```
- Calculates the accuracy of the model
- Compares predicted labels with actual labels

### ▶️ How it Works (Step-by-step)
1. Load the breast cancer dataset from scikit-learn
2. Extract features (x) and target labels (y)
3. Split the data into training (80%) and testing (20%) sets
4. Create an SVM classifier with a linear kernel and C=2
5. Train the model on the training data
6. Use the trained model to predict classes for the test data
7. Calculate and display the accuracy of the model

### 📄 External References
- [Scikit-learn Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Scikit-learn train_test_split Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Scikit-learn SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

## 📊 Dataset
The breast cancer dataset is a built-in dataset in scikit-learn with the following characteristics:
- Features: 30 numerical features extracted from images of breast mass cells
- Target: Binary classification (0 for malignant, 1 for benign)
- Size: 569 instances

## ▶️ How to Run
```bash
# Navigate to the Support Vector Machine directory
cd "SUPERVISED/Support Vector Machine [SVM]"

# Run the script
python svm1.py
```

## 🔍 SVM Concepts Explained

### Hyperplane and Margin
SVM finds the optimal hyperplane that maximizes the margin between different classes. The margin is the distance between the hyperplane and the closest data points from each class (called support vectors).

### Kernels
When data is not linearly separable, SVM uses kernel functions to transform the data into a higher-dimensional space where it becomes linearly separable. Common kernels include:
- Linear: No transformation
- Polynomial: Maps data to a higher-dimensional polynomial space
- RBF (Radial Basis Function): Maps to an infinite-dimensional space
- Sigmoid: Similar to neural networks

### Soft Margin
The C parameter controls the trade-off between having a smooth decision boundary and classifying training points correctly:
- Small C: Prioritizes a smoother decision boundary, allowing some misclassifications
- Large C: Prioritizes classifying training points correctly, potentially leading to overfitting

## 😎 Fun Fact
SVMs were first introduced in the 1960s but gained popularity in the 1990s. Despite being overshadowed by neural networks in recent years, SVMs still excel in scenarios with limited training data and high-dimensional feature spaces. They're particularly effective in text classification and bioinformatics applications where the number of features far exceeds the number of samples!