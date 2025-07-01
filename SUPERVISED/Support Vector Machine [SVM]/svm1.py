# SVM is Used for Performing Classifications  Algorithm we used for this called SVC(support vector classification)
'''
SVM Working :
Its simple classify or divide data using a hyperplane(margin) between them 
lets we have a red and green balls one side is all of red and another side all green balss so we draw a line(hyperplane) tp seperate them just that
now how does it work so we take a closest balls from each color tothe hyper line and the distance for both side balss will be same,but there may be tomany hyper plane can be formed so choose which one
so we choose the hyperplane which get the distance longest in all bcz the total distance is called margin and the longer or can say wider the margin the less in accuracy will be there 
how let we draw a line at passing through each point now these called support vector and there will be no other balls between both support vector and henceless accuracy
Kernals:
now may be few times our datasets are not well classified and overlap to each other sowe pass it into kernals and return a nD dataset to n+1D lets 2D to convert in 3D and make easier to classify as we get one extra dimention to measure and make plane there
how lets we have (x,y) as (2,3) 2^2+3^2 = 13 now we have (x,y,z) = (2,3,13) and if need more dimention then pass to kernal
Soft margin:
as we discusse there will be bo data point fall in between margin but some time letting fall a data point in margin result best or if we are not strict for reult then its called soft margine 
otherwise its hard margin
'''

import sklearn
from sklearn import datasets
from sklearn import svm
import sklearn.model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Willuse dataset of Breast Canser for Perfroming classification 
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names) # return all features i.e x for finding the y i.e which classification based on these x 
# print(cancer.target_names) # return all label/target to predict or cansay classify i.e y 

x = cancer.data # all x based on which it get classify
y = cancer.target # all y to classify in

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# print(x_train,y_train) # it gives that y i.e classes are le as 0 and 1 so lets make list accordingly

classes = ['malignant','bengin']

# can compare with Kneighbours
clf = svm.SVC(kernel='linear', C=2) # there is tomany params u can refer documentation for details by default kernal is rbf
# without any params 0.9473684210526315,  0.8947368421052632
# kernel = liner 0.9649122807017544
# C is like how much soft margin level more it more datapoint in between margin 0.9035087719298246
# linear + c=2   0.9912280701754386
clf.fit(x_test,y_test)

y_predict = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test,y_predict) # compare and give accuracy 
print(accuracy)