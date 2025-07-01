# KNeighbors is a one of the classification method where we find best cass in which the given test value fit let we have 3 class of red blue and gree color 
# and set k=3 so its see for 3 neigbours if its k=5 then its look for 5 neighbours [ one which coloosests as it about neighbouirs 😂 ]
# then we vote based on neigbours let we have 2red and 1 blue for k=3 so the test value get classify as red and for k=5 lets we get 3 green 1 red and 1 blue so it get classify as gree
# always choose k as odd as if not we may get tie instead of winner
# dont choose to big value for K as it may go for big search range for neigbours and we may get wrong value, as parents says dont take toffy from random one and if you play away from neigbourhood it may happen
# So trhats pretty much what happen in  KNeighbors offcourse there exist a mathematical calculations forfinding and voting but overall that all 😊

# little bit maths behind 
# as for calculating distance between neighbours to test point its uses a "Icarian distance" i.e distancebetween two location co ordinates given by:
# sqrt((x2-x1^2+(y2-y1)^2 + ...if any other dimention like z))


import sklearn
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

le = preprocessing.LabelEncoder() # just creat a onject of performing label encoding i.e. convert qulitative datain to quantitive like high low to 2and 1 yes no to 1 0 and so so that mathamatical processing can be done 

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# print(buying) #  [3 3 3 ... 1 1 1]

predict = "class"

x = list(zip(buying, maint, doors, persons, lug_boot, safety,)) # This will return a list of tupples where each tupple have a these 6 values

y = list(cls) # list of class

x_train, x_test, y_train, y_test =sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# print(x_train,y_test)

model = KNeighborsClassifier(n_neighbors=5) # value of k asparameter
model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)
# print(accuracy)

names = ["unacc", "acc", "good", "vgood"] # as classification we have le ofthere index to these 

predict = model.predict(x_test)

for x in range(len(x_test)):
    print(f"Predicted : {names[predict[x]]}, Data : {x_test[x]}, Actual : {names[y_test[x]]} ")
    # Predicted : good, Data : (np.int64(2), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(1)), Actual : good 
    # Predicted : good, Data : (np.int64(1), np.int64(1), np.int64(3), np.int64(2), np.int64(2), np.int64(1)), Actual : good 
    # Predicted : unacc, Data : (np.int64(2), np.int64(0), np.int64(1), np.int64(2), np.int64(1), np.int64(0)), Actual : unacc 
    # Predicted : unacc, Data : (np.int64(2), np.int64(2), np.int64(2), np.int64(1), np.int64(2), np.int64(0)), Actual : unacc 
   
    n = model.kneighbors([x_test[x]], 9, True) # take 2d array and we have 1D hence pass like [x_test[x]] instead just x_test[x] then number of neighbours
    # return deistancefrom point to each neighbour as a array and another array with neighbours index
    print(f'N : {n}')
    # Predicted : good, Data : (np.int64(0), np.int64(2), np.int64(0), np.int64(0), np.int64(1), np.int64(1)), Actual : good
    # N : (array([[1.        , 1.        , 1.        , 1.        , 1.        ,
    #     1.        , 1.        , 1.        , 1.41421356]]), array([[ 905,  489, 1089, 1118, 1484,  308, 1020, 1250,  134]])) distance and index
