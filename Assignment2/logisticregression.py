import numpy as np
import matplotlib.pyplot as plt
import random

test_fr = open('..\LogisticRegression\IrisTestML.dt')
train_fr = open('..\LogisticRegression\IrisTrainML.dt')

# LOAD THE DATASET
def dataSet(fr):
    test_dataMat = []
    test_labelMat = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        if int(lineArr[-1]) == 0:
            lineArr[-1] = -1
        if int(lineArr[-1]) != 2:
            test_dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            test_labelMat.append(int(lineArr[-1]))
    return test_dataMat, test_labelMat

def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))

def gradAscend(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    alpha = 0.001
    maxCycle = 170000
    for i in range(maxCycle):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.T * error
    print(h)
    return weights

def improvement(dataMatrix, classLabels):
    max_iter = 30000
    m, n = np.shape(np.array(dataMatrix))
    weights = np.ones(n)
    for j in range(max_iter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1 + i + j) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * np.asarray(dataMatrix[randIndex]) * error
            del (dataIndex[randIndex])
    print(h)
    return weights

def gradDescend(dataMatIn, classLabels):
    weights = [0, 0, 0]
    data_size = len(dataMatIn)
    maxCycle = 70000
    alpha = 0.7
    for k in range(maxCycle):
        sums = [0, 0, 0]
        each = [0, 0, 0]
        for i in range(data_size):
            power = classLabels[i] * (np.dot(weights, dataMatIn[i]))
            theta = 1/(1 + np.exp(power))
            each = np.dot(dataMatIn[i], classLabels[i]) * theta
            sums = [sums[n] + each[n] for n in range(len(each))]
        g_t = np.dot(-1 / data_size, sums)
        v_t = -g_t
        weights = weights + alpha * v_t
    return weights

def plotBestFit(weights, dataMat, labelMat, train_data, train_label):
    n = np.shape(dataMat)[0]
    m = np.shape(train_data)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcord3 = []; ycord3 = []
    xcord4 = []; ycord4 = []
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    for j in range(m):
        if train_label[j] == 1:
            xcord3.append(train_data[j][1])
            ycord3.append(train_data[j][2])
        else:
            xcord4.append(train_data[j][1])
            ycord4.append(train_data[j][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', label='test_label = 1')
    ax.scatter(xcord2, ycord2, s=30, c='green', marker='d', label='test_label = 0')

    ax.scatter(xcord3, ycord3, s=30, c='yellow', marker='o', label='train_label = 1')
    ax.scatter(xcord4, ycord4, s=30, c='gray', label='train_label = 0')

    min_x = min(train_data[0:])[1]
    max_x = max(train_data[0:])[1]

    min_y = (-weights[0] - weights[1] * min_x) / weights[2]
    max_y = (-weights[0] - weights[1] * max_x) / weights[2]
    ax.plot([min_x, max_x], [min_y, max_y], '-g')

    plt.xlabel('lengths of the sepal')
    plt.ylabel('widths of the sepal')
    plt.legend()
    plt.show()

train_data, train_label = dataSet(train_fr)
test_data, test_label = dataSet(test_fr)
weight2 = gradDescend(train_data, train_label)
print('weight2', weight2)

def classifyVector(inX, trainWeights):
    prob = sigmoid(sum(inX * trainWeights))
    if prob > 0.5:
        return 1
    else:
        return 0

def errorrate(test_data, test_label):
    errorCount = 0;
    numTestVec = 0
    for i in range(len(test_data)):
        if (int(test_label[i]) == -1):
            test_label[i] = 0
    for i in range(len(test_data)):
        numTestVec += 1
        print(classifyVector(test_data[i], weight2), test_label[i])
        if classifyVector(test_data[i], weight2) != int(test_label[i]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is :%f' % errorRate)
    return errorRate

if '__main__' == __name__:
    errorrate(test_data, test_label)
    plotBestFit(weight2, test_data, test_label, train_data, train_label)