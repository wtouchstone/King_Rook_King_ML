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

alg = sys.argv[1]
crossval = int(sys.argv[2])
trainpredict = int(sys.argv[3])

def getData(fileName):
    f = open(fileName, "r")
    data = f.readlines()
    f.close()
    np.random.shuffle(data)
    x = list()
    y = list()
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
    train_x = np.array(x[:int(.5 * len(x))])
    train_y = np.array(y[:int(.5 * len(x))])
    test_x = np.array(x[int(.5 * len(x)):])
    test_y = np.array(y[int(.5 * len(x)):])
    return(train_x, train_y, test_x, test_y)

def compare(actual, expected):
    length = len(actual)
    tot = 0
    for i in range(len(actual)):
        if actual[i] == expected[i]:
            tot += 1
    return float(tot) / length

def simplifyY(y):
    for i in range(len(y)):
        if y[i] != 'draw':
            y[i] = "win"



train_x, train_y, test_x, test_y = getData("krkopt.data")

baggingParams = {"n_estimators": np.concatenate([np.arange(1,10)])}#, np.arange(0,10)*10+10, np.arange(0,10)*100+100])}
svmParams = {"kernel": ['rbf', 'poly', 'sigmoid'], 'degree': np.arange(2,10), 'C': [1, 5, 10, 50, 100, 500, 1000]}
#svmParams = {"kernel": ['rbf', 'poly', 'sigmoid'], 'C': [1, 5, 10, 50, 100, 500, 1000], 'gamma':['scale', 'auto']}
nnParams = {"hidden_layer_sizes": [(2,2,2) ,(5,5,5), (10,10,10), (20,20,20), (200,200,200), (500,500,500), (2,2,2,2) ,(5,5,5,5), (10,10,10,10), (20,20,20,20) , (200,200,200,200) , (500,500,500,500) ]}



def baggingCrossVal():
    bagging = BaggingClassifier(n_estimators=4)
    bagging.fit(train_x, train_y)
    gscv = GridSearchCV(bagging, baggingParams)
    gscv.fit(train_x, train_y)
    print(gscv.cv_results_['params'])
    print(gscv.cv_results_['mean_test_score'])

def svmCrossVal():
    svm = svm.SVC()
    svm.fit(train_x, train_y)
    gscv = GridSearchCV(svm, svmParams)
    gscv.fit(train_x, train_y)
    print(gscv.cv_results_['params'])
    print(gscv.cv_results_['mean_test_score'])

def nnCrossVal():
    nn = MLPClassifier(hidden_layer_sizes=(4,4), max_iter=2000)
    nn.fit(train_x, train_y)
    gscv = GridSearchCV(nn, nnParams)
    gscv.fit(train_x, train_y)
    print(gscv.cv_results_['params'])
    print(gscv.cv_results_['mean_test_score'])


def nnperformance(train_x, train_y, test_x, test_y):
    nn = MLPClassifier(hidden_layer_sizes=(500,500,500,500), max_iter=2000)
    nn.fit(train_x, train_y)
    perflist = list()
    for i in range(50):
        index = int(len(test_x) / 50) #using 50 for clt
        #print(i * index)
        #print((i+1)*index)
        curr = nn.predict(test_x[i * index: (i + 1) *index])
        perflist.append(compare(curr, test_y[i * index: (i + 1) *index]))
    mean = sum(perflist) / len(perflist)
    standev = np.std(perflist)
    print(mean)
    print(standev)
    #print(perflist)

def baggingperformance(train_x, train_y, test_x, test_y):
    bagging = BaggingClassifier(n_estimators=1000)
    bagging.fit(train_x, train_y)
    perflist = list()
    for i in range(50):
        index = int(len(test_x) / 50) #using 50 for clt
        #print(i * index)
        #print((i+1)*index)
        curr = bagging.predict(test_x[i * index: (i + 1) *index])
        perflist.append(compare(curr, test_y[i * index: (i + 1) *index]))
    mean = sum(perflist) / len(perflist)
    standev = np.std(perflist)
    print(mean)
    print(standev)
    #print(perflist)

def svmperformance(train_x, train_y, test_x, test_y):
    svmt = svm.SVC(C=100)
    svmt.fit(train_x, train_y)
    perflist = list()
    for i in range(50):
        index = int(len(test_x) / 50) #using 50 for clt
        #print(i * index)
        #print((i+1)*index)
        curr = svmt.predict(test_x[i * index: (i + 1) *index])
        perflist.append(compare(curr, test_y[i * index: (i + 1) *index]))
    mean = sum(perflist) / len(perflist)
    standev = np.std(perflist)
    print(mean)
    print(standev)
    #print(perflist)



if alg == "bagging":
    if crossval == 1:
        baggingCrossVal()
    if trainpredict == 1:
        baggingperformance(train_x, train_y, test_x, test_y)

if alg == "svm":
    if crossval == 1:
        svmCrossVal()
    if trainpredict == 1:
        svmperformance(train_x, train_y, test_x, test_y)

if alg == "nn":
    if crossval == 1:
        nnCrossVal()
    if trainpredict == 1:
        nnperformance(train_x, train_y, test_x, test_y)





#nn = MLPClassifier(hidden_layer_sizes=(500, 500, 500, 500), max_iter=2000)
#nn.fit(train_x, train_y)



#result = nn.predict(test_x)
#print(compare(result, test_y))

#gscv = GridSearchCV(svm, svmParams)
#gscv.fit(train_x, train_y)

#linearsvc = LinearSVC()
#linearsvc.fit(train_x, train_y)
#result = linearsvc.predict(test_x)
#print(compare (result, test_y))


#result = gscv.fit(train_x, train_y)
#print(gscv.cv_results_['params'])
#print(gscv.cv_results_['mean_test_score'])
#result = svm.predict(test_x)
#print(compare(result, test_y))
#print(tend-tstart)
#result = clf.predict(test_x)
#print(compare(result, test_y))
#['mean_fit_time', 'mean_score_time', 'mean_test_score', 'param_C', 'params', 'rank_test_score', 'split0_test_score', 
#'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'std_fit_time', 'std_score_time', 'std_test_score']