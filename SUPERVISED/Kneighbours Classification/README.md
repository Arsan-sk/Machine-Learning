# 📂 K-Nearest Neighbors Classification

## 🧠 Overview
This directory contains an implementation of the K-Nearest Neighbors (KNN) classification algorithm. KNN is a simple, instance-based learning algorithm that classifies new data points based on the majority class of their k nearest neighbors in the feature space.

## 📘 Learning & Concepts Covered
- Understanding the KNN algorithm and its mathematical foundation
- Converting categorical data to numerical data using label encoding
- Training and evaluating a KNN classifier
- Visualizing and interpreting KNN results
- Exploring the impact of the 'k' parameter on classification
- Computing distances between data points using Euclidean distance

## 🎯 File: `2ndml.py`

### 📌 Concept/Goal
The script implements a K-Nearest Neighbors classifier to categorize cars into different classes (unacceptable, acceptable, good, very good) based on various attributes like buying price, maintenance cost, number of doors, capacity, luggage boot size, and safety rating. It demonstrates the complete workflow from data preprocessing to model evaluation and prediction visualization.

### ⚙️ Functions & Methods Used

#### `pd.read_csv()`
```python
data = pd.read_csv("car.data")
```
- Loads data from a CSV file into a pandas DataFrame
- Used to read the car evaluation dataset

#### `preprocessing.LabelEncoder()`
```python
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
```
- Creates a label encoder object
- Converts categorical data (like 'high', 'low') to numerical values
- Applied to each categorical feature in the dataset

#### `list(zip())`
```python
x = list(zip(buying, maint, doors, persons, lug_boot, safety))
```
- Combines multiple lists into a list of tuples
- Each tuple contains the feature values for one data point

#### `sklearn.model_selection.train_test_split()`
```python
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
```
- Splits data into training and testing sets
- `test_size=0.1` allocates 10% of data for testing

#### `KNeighborsClassifier()`
```python
model = KNeighborsClassifier(n_neighbors=5)
```
- Creates a KNN classifier with k=5 neighbors
- The parameter `n_neighbors` determines how many neighbors to consider for classification

#### `model.fit()`
```python
model.fit(x_train, y_train)
```
- Trains the KNN model on the training data
- For KNN, this essentially stores the training data for later use in predictions

#### `model.score()`
```python
accuracy = model.score(x_test, y_test)
```
- Evaluates model performance on test data
- Returns the accuracy (proportion of correctly classified instances)

#### `model.predict()`
```python
predict = model.predict(x_test)
```
- Uses the trained model to predict classes for test data
- Returns an array of predicted class labels

#### `model.kneighbors()`
```python
n = model.kneighbors([x_test[x]], 9, True)
```
- Finds the k nearest neighbors of a data point
- Returns distances and indices of the neighbors
- Used to visualize which neighbors influenced a particular prediction

### ▶️ How it Works (Step-by-step)
1. Load the car evaluation dataset from 'car.data'
2. Convert categorical features to numerical values using label encoding:
   - buying price (buying)
   - maintenance cost (maint)
   - number of doors (doors)
   - person capacity (persons)
   - luggage boot size (lug_boot)
   - safety rating (safety)
   - class label (cls)
3. Prepare the feature matrix (x) and target vector (y)
4. Split the data into training (90%) and testing (10%) sets
5. Create a KNN classifier with k=5 neighbors
6. Train the model on the training data
7. Evaluate the model's accuracy on the test data
8. Make predictions on the test data
9. For each test instance:
   - Display the predicted class, feature values, and actual class
   - Find and display the 9 nearest neighbors and their distances

### 📄 External References
- [Pandas read_csv Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
- [Scikit-learn LabelEncoder Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- [Python zip Documentation](https://docs.python.org/3/library/functions.html#zip)
- [Scikit-learn train_test_split Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Scikit-learn KNeighborsClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## 📊 Dataset
The `car.data` file contains car evaluation data with the following attributes:
- buying: buying price (v-high, high, med, low)
- maint: maintenance price (v-high, high, med, low)
- doors: number of doors (2, 3, 4, 5-more)
- persons: capacity in terms of persons (2, 4, more)
- lug_boot: luggage boot size (small, med, big)
- safety: estimated safety of the car (low, med, high)
- class: car acceptability (unacc, acc, good, vgood)

## ▶️ How to Run
```bash
# Navigate to the KNeighbours Classification directory
cd "SUPERVISED/Kneighbours Classification"

# Run the script
python 2ndml.py
```

## 😎 Fun Fact
The K-Nearest Neighbors algorithm is sometimes called a "lazy learner" because it doesn't actually "learn" anything during the training phase - it simply stores the training data and does all the real work during prediction time! This is in contrast to "eager learners" like decision trees or neural networks that build a generalized model during training.