from svmutil import *

y, x = [1,-1], [[1, 2, -2], [2, -2, 2]]
y, x = [1,-1], [{0:1, 1:2, 2:-2}, {0:2, 1:-2, 2:2}]
prob  = svm_problem(y, x, isKernel=True)
param = svm_parameter('-t 4 -c 4 -b 1')
m = svm_train(prob, param)
