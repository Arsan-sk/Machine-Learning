# import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
import pickle   #used to save model
from matplotlib import style


data = pd.read_csv('student-mat.csv', sep = ';')

# print(data.head())

# linear regression = mx+C for 2 dimentional set with just x and y 
# byt in real life we may have more x and y for one y to predict we have numbers of variable x so we have like follow:
# y = ax1+bx2+cx3+dx4+...+ E 
# the higher the variable x it effect the y more 

data = data[["G1","G2","G3","studytime","failures","absences"]]
# print(data.head())

predict = "G3" # its a label as we predict it

x = np.array(data.drop([predict], axis=1)) # give all column of data except predict as we drop it
y = np.array(data[predict])  # give array of only predict column 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state=40) # split data as train and test part as if we test it on samedata its know forit the result.so we say split 10% as test and get train on remainig 90%
# if we set the random state according to to random_state assign its choose same dta each time like for 42 let p train dta each timegiven to train that like respective for 42 and if assign 40 we gwt different data for training that will be same each time due to 40  other wise we get random accuracy as we run programe each timetrain data get change
#accuracy befor andafter setting random state
# 0.7998157060810287 without random state
# 0.8500249847617904 without random state
# 0.8947612020229178 without random state
# 0.8840114312044507 without random state
# 0.7477224871562675 at 42
# 0.7477224871562675 at 42
# 0.7477224871562675 at 42
# 0.8745669515393772 at 40
# 0.8745669515393772 at 40
# so we can try different value for getting best accuracy while commonly used is 43


# linear = linear_model.LinearRegression() # load linear regression model in linear  
# linear.fit(x_train,y_train)

# accuracy = linear.score(x_test,y_test) # this gives model accuracy based on the test data result
# print(accuracy)

# with open("student-model.pickle", "wb") as f:
#     pickle.dump(linear, f) # its take linear model and save it in the .pickle file


# we can loop through n times for finding best model instead of passing random_state and save .pckle file if its accuracy > desired


# # load model in line variable from .pickle file as we comment out the training model it fetch it from .pickle file and peform texting
pickle_in = open("student-model.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient : \n", linear.coef_) # as we have 5 varibale on which y depends we will get 5 coefficient as we disscusse at line 11-14
#  [ 0.14611419  0.9845577  -0.18626916 -0.26050632  0.03515658]
print("Intercept : \n", linear.intercept_) # intercept where best fit line cut on y axis
#  -1.4748231341482096


# use of model to predict

predictions = linear.predict(x_test)

# checking predicted and actual value fo test value
for x in range(len(predictions)):
    print(f"{predictions[x]} | {x_test[x]} | {y_test[x]}")


# plot making

p = 'studytime' # pass each variable and check there corelation with y
style.use("ggplot")
plt.scatter(data[p],data['G3']) # pass x and y where we can change x to check y correlation with each variable
plt.xlabel("G1")
plt.ylabel("Final Grade")
plt.show()