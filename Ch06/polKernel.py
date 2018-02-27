import numpy as np
import svmMLiA
import colorUtil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
# w [[ -6.9696585  -12.89935491  -2.40255723]]
# there are 13 support vectors
# the training error rate is 0.010000
# the test error rate is 0.060000
# linear kernel
#    if kTup[0]=='lin':
#       K = X * A.T   
k1 = 1.3
dataArr, labelArr = svmMLiA.loadDataSet('testSetRBF.txt')
# (x1, x2) -> (x1^2, x2^2, 2^0.5*x1*x2)
ndDataArr = np.array(dataArr)
athree = np.array([np.multiply(ndDataArr[:, 0], np.power(2, 0.5) * ndDataArr[:, 1])]).reshape((100, 1))
ndDataArr = np.append(ndDataArr, athree, axis=1)
for v in ndDataArr:
    v[0] = v[0] ** 2
    v[1] = v[1] ** 2
feature1 = np.array(ndDataArr)[:, 0]
feature2 = np.array(ndDataArr)[:, 1]
feature3 = np.array(ndDataArr)[:, 2]
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

colors = list(map(colorUtil.color, labelArr))

ax1.scatter(feature1, feature2, feature3, s=80, c=colors, marker='+')
# plt.show()
b, alphas = svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 100000, ('pol', k1))

# to calculate w
alphas_w = np.array(alphas).T
labels_w = np.array([labelArr])
w = np.dot(np.multiply(alphas_w, labels_w), ndDataArr)
print w
datMat = mat(dataArr)
labelMat = mat(labelArr).transpose()


svInd = nonzero(alphas.A>0)[0]
svs = datMat[svInd]
labelSV = labelMat[svInd]

print "there are %d support vectors" % shape(svs)[0]
m, n = shape(datMat)
errorCount = 0
predict = []
for i in range(m):
    kernelEval = svmMLiA.kernelTrans(svs, datMat[i, :], ('pol', k1))
    predict0 = np.array(kernelEval.T * multiply(labelSV, alphas[svInd]) + b)
    predict.append(predict0)

    if sign(predict0) != sign(labelArr[i]):
        errorCount += 1
print "the training error rate is %f" % (float(errorCount)/m)
dataArr, labelArr = svmMLiA.loadDataSet('testSetRBF2.txt')
errorCount = 0
datMat = mat(dataArr)
labelMat = mat(labelArr).transpose()
m, n = shape(datMat)
for i in range(m):
    kernelEval = svmMLiA.kernelTrans(svs, datMat[i, :], ('pol', k1))
    predict0 = np.array(kernelEval.T * multiply(labelSV, alphas[svInd]) + b)
    predict.append(predict0)

    if sign(predict0) != sign(labelArr[i]):
        errorCount += 1
print "the test error rate is %f" % (float(errorCount)/m)

alpha_index = np.where(alphas.getA() > 0)
support_vector_value = np.dot(w, ndDataArr[alpha_index[0]].T) + b

support_vector_x = ndDataArr[alpha_index[0]][:, 0]
support_vector_y = ndDataArr[alpha_index[0]][:, 1]
support_vector_z = ndDataArr[alpha_index[0]][:, 2]

support_vector_labels = np.array(labelArr)[alpha_index[0]]
support_vector_colors = list(map(colorUtil.color, support_vector_labels))
ax1.scatter(support_vector_x, support_vector_y, support_vector_z, s=100, facecolors='none', edgecolors=support_vector_colors)

# draw plane
x = np.arange(0, 1, 0.1)
y = np.arange(0, 1, 0.1)
x, y = np.meshgrid(x, y)
w0 = w[0][0]
w1 = w[0][1]
w2 = w[0][2]

z = -(w0 * x + w1 * y + b)/w2
ax1.plot_surface(x, y, z,cmap=plt.get_cmap('rainbow'))
plt.show()