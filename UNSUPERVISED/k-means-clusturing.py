'''
K-Means-Clustring is a algorithm for unsupervised ML where unLike of supervised ML we dont give labels/target to predict 
but based on given data patterns its make clusters of same data sets as per features hence we get n cluster ie predictions can be make based on given data sets
How its work ... Note [When ever itssays data point make figure of some datasets forunsupervised learnig without label let take features such as bedrooms area nearest_Hospital nearest_school nearest_mall are features given and model makes cluster baed on dataset for different pattern in it]
sofirstly it make K centroid as given let 2 Randomly in for data point 
then make a straight line between them and make a perpendicular straight line from center of that line 
now calculate each datapoint distance from both centroid and divide them based on nearest centroid 
hence we got two clusters 
now for each cluster take average find center and make centeroids at that coordinate 
then repeat that line making process 
then again divide datapoints based on new line as may be there points go here and there 
repeat this process untill we can furthur seperate point for each centroid
and yup we got our clusters ready 
just think deep
we have our datapoint [features mentioned] with its nearest centroid (matching) point i.e all they are around same [as we do that clusturing based on distance and in coordinate syste, point are close if have same features value]
hence each cluster can be identify as a same values and we get our prediction model ready
if we pass a random fetures completely new [as mentioned above] its get settle in respective matching cluster 
and i.e what model predict for it that it fit for which range 
'''

import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data) # scale does as its calledjust lower the scale of values to make computations easy like we have 1,2,3,4,5 so may be scale like 0.1,0.2,0.3,0.4,0.5 its just example real data may be way more big instead 1,2,3...
y= digits.target

k = 10
samples, features = data.shape

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
    # go for documentation to know in detail about all score what they tell about
    
clf = KMeans(n_clusters=k, init='random', n_init=10)
bench_k_means(clf, "1", data)
