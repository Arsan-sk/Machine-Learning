# 📂 Linear Regression

## 🧠 Overview
This directory contains an implementation of Linear Regression, a fundamental supervised learning algorithm used for predicting continuous values. The implementation uses the student performance dataset to predict final grades based on various features.

## 📘 Learning & Concepts Covered
- Understanding the mathematical foundation of Linear Regression (y = mx + c for 2D, y = ax₁ + bx₂ + cx₃ + ... + E for multi-dimensional)
- Data preprocessing and feature selection
- Training and testing model splitting
- Model evaluation and accuracy measurement
- Model persistence using pickle
- Data visualization with matplotlib

## 🎯 File: `1stml.py`

### 📌 Concept/Goal
The script implements a linear regression model to predict students' final grades (G3) based on their previous grades (G1, G2), study time, failures, and absences. It demonstrates the complete machine learning workflow from data loading to model evaluation and visualization.

### ⚙️ Functions & Methods Used

#### `pd.read_csv()`
```python
data = pd.read_csv('student-mat.csv', sep = ';')
```
- Loads data from a CSV file into a pandas DataFrame
- `sep = ';'` specifies the delimiter used in the CSV file

#### `np.array()`
```python
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
```
- Converts pandas DataFrame or Series to NumPy arrays
- Used to prepare data for scikit-learn models

#### `sklearn.model_selection.train_test_split()`
```python
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state=40)
```
- Splits data into training and testing sets
- `test_size=0.1` allocates 10% of data for testing
- `random_state=40` ensures reproducibility of the split

#### `linear_model.LinearRegression()`
```python
linear = linear_model.LinearRegression()
```
- Creates a linear regression model object
- Implements Ordinary Least Squares method

#### `linear.fit()`
```python
linear.fit(x_train, y_train)
```
- Trains the model on the training data
- Calculates the coefficients and intercept

#### `linear.score()`
```python
accuracy = linear.score(x_test, y_test)
```
- Evaluates model performance on test data
- Returns R² score (coefficient of determination)

#### `pickle.dump()` and `pickle.load()`
```python
with open("student-model.pickle", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("student-model.pickle", "rb")
linear = pickle.load(pickle_in)
```
- Saves and loads the trained model to/from a file
- Allows reusing the model without retraining

#### `plt.scatter()` and `plt.show()`
```python
plt.scatter(data[p], data['G3'])
plt.xlabel("G1")
plt.ylabel("Final Grade")
plt.show()
```
- Creates scatter plots to visualize relationships between variables
- Displays the plot on screen

### ▶️ How it Works (Step-by-step)
1. Load the student dataset from 'student-mat.csv'
2. Select relevant features (G1, G2, studytime, failures, absences) and the target variable (G3)
3. Split the data into features (x) and target (y)
4. Further split the data into training (90%) and testing (10%) sets
5. Create and train a linear regression model on the training data
6. Evaluate the model's accuracy on the test data
7. Save the trained model to a pickle file for future use
8. Load the model from the pickle file
9. Display the model's coefficients and intercept
10. Use the model to make predictions on the test data
11. Compare predicted values with actual values
12. Create a scatter plot to visualize the relationship between a selected feature and the target variable

### 📄 External References
- [Pandas read_csv Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
- [NumPy array Documentation](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
- [Scikit-learn train_test_split Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Scikit-learn LinearRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Python pickle Documentation](https://docs.python.org/3/library/pickle.html)
- [Matplotlib scatter Documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)

## 📊 Dataset
The `student-mat.csv` file contains student performance data with various attributes including:
- Past performance (G1, G2)
- Study time
- Number of failures
- Absences
- Final grade (G3) - the target variable

## ▶️ How to Run
```bash
# Navigate to the Linear Regression directory
cd "SUPERVISED/Linear Regression"

# Run the script
python 1stml.py
```

## 😎 Fun Fact
Linear regression is one of the oldest statistical techniques, dating back to the early 19th century when it was first used by Adrien-Marie Legendre and Carl Friedrich Gauss. Despite its age and simplicity, it remains a powerful tool in the modern data scientist's toolkit and is often the first algorithm taught in machine learning courses!