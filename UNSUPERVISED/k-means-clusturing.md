# 📂 K-Means Clustering

## 🧠 Overview
This file contains an implementation of K-Means Clustering, a fundamental unsupervised learning algorithm used for grouping similar data points into clusters. The implementation uses the digits dataset to demonstrate how K-means can identify patterns in unlabeled data.

## 📘 Learning & Concepts Covered
- Understanding the K-means algorithm and its mathematical foundation
- Data preprocessing and scaling
- Cluster initialization and centroid calculation
- Distance metrics and cluster assignment
- Evaluating clustering performance with various metrics
- Visualizing clustering results

## 🎯 File: `k-means-clusturing.py`

### 📌 Concept/Goal
The script implements K-means clustering to group handwritten digits from the digits dataset into clusters based on their feature similarity. It demonstrates the complete workflow from data loading and preprocessing to model training and evaluation using various clustering metrics.

### ⚙️ Functions & Methods Used

#### `load_digits()`
```python
digits = load_digits()
```
- Loads the digits dataset from scikit-learn's built-in datasets
- Contains images of handwritten digits (0-9) and their labels

#### `scale()`
```python
data = scale(digits.data)
```
- Standardizes features by removing the mean and scaling to unit variance
- Makes computation easier and improves algorithm performance
- As explained in the code comments, this reduces the scale of values (e.g., converting 1,2,3,4,5 to 0.1,0.2,0.3,0.4,0.5)

#### `KMeans()`
```python
clf = KMeans(n_clusters=k, init='random', n_init=10)
```
- Creates a K-means clustering model
- `n_clusters=k` sets the number of clusters (10 for digits 0-9)
- `init='random'` specifies random initialization of centroids
- `n_init=10` runs the algorithm 10 times with different initializations

#### `bench_k_means()`
```python
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'%
          (name, estimator.inertia_,
           metrics.homogeneity_score(y, estimator.labels_),
           metrics.completeness_score(y, estimator.labels_),
           metrics.v_measure_score(y, estimator.labels_),
           metrics.adjusted_rand_score(y,estimator.labels_),
           metrics.adjusted_mutual_info_score(y, estimator.labels_),
           metrics.silhouette_score(data, estimator.labels_,
                                    metric='euclidean')))
```
- Custom function to evaluate the K-means model using various metrics
- Trains the model and prints performance metrics

#### Evaluation Metrics
- **inertia_**: Sum of squared distances of samples to their closest cluster center
- **homogeneity_score**: Each cluster contains only members of a single class
- **completeness_score**: All members of a given class are assigned to the same cluster
- **v_measure_score**: Harmonic mean of homogeneity and completeness
- **adjusted_rand_score**: Similarity between true labels and cluster assignments
- **adjusted_mutual_info_score**: Information between true labels and cluster assignments
- **silhouette_score**: Measure of how similar an object is to its own cluster compared to other clusters

### ▶️ How it Works (Step-by-step)
1. Load the digits dataset from scikit-learn
2. Scale the data to standardize features
3. Extract the target labels (y) for evaluation purposes
4. Set the number of clusters (k=10) to match the number of digit classes
5. Create a K-means clustering model with random initialization
6. Train the model on the scaled data
7. Evaluate the clustering performance using various metrics
8. Display the evaluation results

### 📄 External References
- [Scikit-learn Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Scikit-learn Preprocessing Scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)
- [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-learn Clustering Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.cluster)

## 📊 Dataset
The digits dataset is a built-in dataset in scikit-learn with the following characteristics:
- Features: 64 numerical features (8x8 pixel images flattened)
- Target: 10 classes (digits 0-9)
- Size: 1797 instances

## ▶️ How to Run
```bash
# Navigate to the UNSUPERVISED directory
cd "UNSUPERVISED"

# Run the script
python k-means-clusturing.py
```

## 🔍 K-Means Algorithm Explained

### How K-Means Works
As explained in the code comments:

1. **Initialization**: Randomly select k centroids from the data points
2. **Assignment**: Calculate the distance of each data point from both centroids and assign points to the nearest centroid, forming clusters
3. **Update**: For each cluster, calculate the average of all points to find a new centroid
4. **Repeat**: Redraw the dividing line and reassign points that may have changed clusters
5. **Convergence**: Continue until no further separation of points is possible

### Real-World Example
As mentioned in the code comments, consider a real estate dataset with features like:
- Number of bedrooms
- Area
- Distance to nearest hospital
- Distance to nearest school
- Distance to nearest mall

K-means would group similar properties together, allowing for market segmentation or property type classification without predefined labels.

### Limitations
- Sensitive to initial centroid selection
- Assumes clusters are spherical and equally sized
- May converge to local optima
- Requires specifying the number of clusters in advance

## 😎 Fun Fact
K-means clustering was first proposed in 1957, making it one of the oldest algorithms still widely used in machine learning today! Despite its age, it remains popular due to its simplicity and efficiency. The algorithm is so versatile that it's used in fields ranging from market segmentation and document clustering to image compression and anomaly detection.