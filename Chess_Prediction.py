from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.decomposition import PCA
import time
from sklearn.preprocessing import MinMaxScaler
import math as m
import sys 


data_x = None
data_y = None
train_x = list()
train_y = list()
test_x = list()
test_y = list()

# Command line arguments as python Chess_Prediction.py [algorithm to use] [cross-validation] [train and predict]
# Usage specified in README

alg = sys.argv[1]
crossval = int(sys.argv[2])
trainpredict = int(sys.argv[3])


# This function will get data from a file in the format 'feature, feature, ..., classifier'
# and split it into a 2d array for training features, training classifiers, testing features
# and testing classifiers. At the moment there is a 50/50 ratio between training and testing
# data, but this can be modified by changing the training_ratio value (for example 0.8
# for 80% of data to go to training set and 20% to testing).

def getData(fileName):
    f = open(fileName, "r")
    data = f.readlines()
    f.close()
    np.random.shuffle(data)
    x = list()
    y = list()
    training_ratio = 0.5
    for i in data:
        temp = i.split(",")
        num_features = len(temp) - 1
        newlist = list()
        for i in range(num_features):
            newlist.append(ord(temp[i]))
        x.append(newlist[:num_features])
        y.append(temp[num_features].strip())
    scalar = MinMaxScaler()
    scalar.fit(x)
    x = scalar.transform(x)
    train_x = np.array(x[:int(training_ratio * len(x))])
    train_y = np.array(y[:int(training_ratio * len(x))])
    test_x = np.array(x[int(1 - training_ratio * len(x)):])
    test_y = np.array(y[int(1- training_ratio * len(x)):])
    return(train_x, train_y, test_x, test_y)

# This function will calculate a score based on the percent correctly identified in
# the testing set. This calculation is a simple correct / total.

def compare(actual, expected):
    length = len(actual)
    tot = 0
    for i in range(len(actual)):
        if actual[i] == expected[i]:
            tot += 1
    return float(tot) / length

# This function simplifies the dataset to be a strict draw vs win scenario. As is seen in the data,
# without this the data would be classified by moves taken for king-rook side to beat king side.
# This function is included for simplicity and understanding of scoring system, as the algorithm 
# outputting 3 when the answer was 4 being counted incorrect biases against the algorithm more than
# a win/draw situation

def simplifyY(y):
    for i in range(len(y)):
        if y[i] != 'draw':
            y[i] = "win"



train_x, train_y, test_x, test_y = getData("krkopt.data")

baggingParams = {"n_estimators": np.concatenate([np.arange(1,10)])} #these are dicts of params to thoroughly cross validate across. They can be modified for different results.
svmParams = {"kernel": ['rbf', 'poly', 'sigmoid'], 'degree': np.arange(2,10), 'C': [1, 5, 10, 50, 100, 500, 1000]}
nnParams = {"hidden_layer_sizes": [(2,2,2) ,(5,5,5), (10,10,10), (20,20,20), (200,200,200), (500,500,500), (2,2,2,2) ,(5,5,5,5), (10,10,10,10), (20,20,20,20) , (200,200,200,200) , (500,500,500,500) ]}

# Performs cross-validation on bagging forest classifier, prints results with each set of params and their mean test score.

def bagging_cross_val():
    bagging = BaggingClassifier(n_estimators=4)
    bagging.fit(train_x, train_y)
    gscv = GridSearchCV(bagging, baggingParams)
    gscv.fit(train_x, train_y)
    print(gscv.cv_results_['params'])
    print(gscv.cv_results_['mean_test_score'])

# Performs cross-validation on support vector machine classifier, prints results with each set of params and their mean test score.
    
def svm_cross_val():
    svm = svm.SVC()
    svm.fit(train_x, train_y)
    gscv = GridSearchCV(svm, svmParams)
    gscv.fit(train_x, train_y)
    print(gscv.cv_results_['params'])
    print(gscv.cv_results_['mean_test_score'])

# Performs cross-validation on neural network classifier, prints results with each set of params and their mean test score.
    
def nn_cross_val():
    nn = MLPClassifier(hidden_layer_sizes=(4,4), max_iter=2000)
    nn.fit(train_x, train_y)
    gscv = GridSearchCV(nn, nnParams)
    gscv.fit(train_x, train_y)
    print(gscv.cv_results_['params'])
    print(gscv.cv_results_['mean_test_score'])

# Will train and predict with neural network classifier - Value of the parameters in the constructor must be changed 
# manually to fit what works best from crossvalidation. These two functions, while can be run on the same run of the
# program will operate independently from each other.

def nn_performance(train_x, train_y, test_x, test_y):
    nn = MLPClassifier(hidden_layer_sizes=(500,500,500,500), max_iter=2000)
    nn.fit(train_x, train_y)
    perflist = list()
    for i in range(50):
        index = int(len(test_x) / 50) #using 50 for clt
        curr = nn.predict(test_x[i * index: (i + 1) *index])
        perflist.append(compare(curr, test_y[i * index: (i + 1) *index]))
    mean = sum(perflist) / len(perflist)
    standev = np.std(perflist)
    print(mean)
    print(standev)

# Will train and predict with bagging forest classifier - Value of the parameters in the constructor must be changed 
# manually to fit what works best from crossvalidation. These two functions, while can be run on the same run of the
# program will operate independently from each other.

    
def bagging_performance(train_x, train_y, test_x, test_y):
    bagging = BaggingClassifier(n_estimators=1000)
    bagging.fit(train_x, train_y)
    perflist = list()
    for i in range(50):
        index = int(len(test_x) / 50) #using 50 for clt
        curr = bagging.predict(test_x[i * index: (i + 1) *index])
        perflist.append(compare(curr, test_y[i * index: (i + 1) *index]))
    mean = sum(perflist) / len(perflist)
    standev = np.std(perflist)
    print(mean)
    print(standev)

# Will train and predict with support vector classifier - Value of the parameters in the constructor must be changed 
# manually to fit what works best from crossvalidation. These two functions, while can be run on the same run of the
# program will operate independently from each other.


def svm_performance(train_x, train_y, test_x, test_y):
    svmt = svm.SVC(C=100)
    svmt.fit(train_x, train_y)
    perflist = list()
    for i in range(50):
        index = int(len(test_x) / 50) #using 50 for clt
        curr = svmt.predict(test_x[i * index: (i + 1) *index])
        perflist.append(compare(curr, test_y[i * index: (i + 1) *index]))
    mean = sum(perflist) / len(perflist)
    standev = np.std(perflist)
    print(mean)
    print(standev)




if alg == "bagging":
    if crossval == 1:
        bagging_cross_val()
    if trainpredict == 1:
        bagging_performance(train_x, train_y, test_x, test_y)

if alg == "svm":
    if crossval == 1:
        svm_cross_val()
    if trainpredict == 1:
        svm_performance(train_x, train_y, test_x, test_y)

if alg == "nn":
    if crossval == 1:
        nn_cross_val()
    if trainpredict == 1:
        nn_performance(train_x, train_y, test_x, test_y)

