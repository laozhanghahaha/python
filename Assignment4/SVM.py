from svmutil import *
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss
from sklearn.metrics import make_scorer
from sklearn.svm import SVC


parkinsonsTestStatML = open("..\Assignment4\parkinsonsTestStatML.dt")
parkinsonsTrainStatML = open("..\Assignment4\parkinsonsTrainStatML.dt")

data_Test = np.loadtxt(parkinsonsTestStatML)
data_Test_Mat =  np.reshape(data_Test,(-1,23))

data_Train = np.loadtxt(parkinsonsTrainStatML)
data_Train_Mat = np.reshape(data_Train,(-1,23))

# another way to nomalization data
# x_train = np.array(data_Train_Mat[:,0:len(data_Train_Mat[0])-1])
# x_scale = preprocessing.scale(x_train)

def normalization(data):
    data = np.array(data[:,0:len(data[0])-1])
    data = np.rollaxis(data, axis=0)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    return data

data_Train_feature = normalization(data_Train_Mat)

# print(np.mean(data_Train_feature,axis=0))
# print(np.std(data_Train_feature,axis=0))

data_Train_label = data_Train_Mat[:,len(data_Train_Mat[0])-1]

data_Test_feature = normalization(data_Test_Mat)
data_Test_label = data_Test_Mat[:,len(data_Test_Mat[0])-1]

# print(np.mean(data_Test_feature,axis=0))
# print(np.std(data_Test_feature,axis=0))

tuned_parameters = {'kernel':['rbf'],'gamma':[1e+2,1e+1,1e-0,1e-1,1e-2,1e-3,1e-4],
                     'C':[0.001,0.01,0.1,1,10,100,1000]}

def sklearnSvm():
        svc = SVC()
        score = make_scorer(zero_one_loss,greater_is_better=False)
        clf = GridSearchCV(svc, tuned_parameters, scoring = score,cv = 5)

        clf.fit(data_Train_feature,data_Train_label)

        test_true, test_predict = data_Test_label, clf.predict(data_Test_feature)
        train_ture, train_predict = data_Train_label, clf.predict(data_Train_feature)

        err_Train = zero_one_loss(train_ture, train_predict)
        err_Test = zero_one_loss(test_true, test_predict)

        print(clf.best_params_)
        print(classification_report(test_true, test_predict))
        print(err_Train)
        print(err_Test)

c_parameters = ['-c 1 ','-c 10 ','-c 100 ','-c 1000 ']
g_parameters = ['-g 1e+1','-g 1e-0','-g 1e-1','-g 1e-2']
#
# def libSvm_find_best(data_label, data_feature):
#     y,x = data_label.tolist(), data_feature.tolist()
#     prob = svm_problem(y, x, isKernel=True)
#     # (ACC, MSE, SCC) = evaluations(data_Train_label, p_label)
#     accuracy = []
#     best_c = []
#     best_g = []
#     for cparam in c_parameters:
#         for gparam in g_parameters:
#             param = svm_parameter(cparam+gparam)
#             model = svm_train(prob, param, '-v 5')
#             p_label, p_acc, p_val = svm_predict(data_Train_label, data_Train_feature.tolist(), model)
#             accuracy.append(p_acc[0])
#             best_c.append(cparam)
#             best_g.append(gparam)
#     index = accuracy.index(max(accuracy))
#     return best_c[index]+best_g[index]
#
#
# best_params = libSvm_find_best(data_Train_label, data_Train_feature)
#
# def libSvm():
#     y, x = data_Train_label.tolist(), data_Train_feature.tolist()
#     prob = svm_problem(y, x, isKernel=True)
#     param = svm_parameter('-c 1 -g 0.1')
#     model = svm_train(prob, param, '-v 5')
#     p_label, p_acc, p_val = svm_predict(data_Train_label, data_Train_feature.tolist(), model)
#     test_label, test_acc, test_val = svm_predict(data_Test_label, data_Test_feature.tolist(), model)
#     err_Test = zero_one_loss(data_Test_label,test_label)
#     err_Train = zero_one_loss(data_Train_label, p_label)
#     print(err_Test,err_Train)

if __name__=='__main__':
    sklearnSvm()
