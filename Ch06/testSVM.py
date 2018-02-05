import svmMLiA
import matplotlib.pyplot as plt
import numpy as np
import colorUtil
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')

feature1 = np.array(dataArr)[:, 0]
feature2 = np.array(dataArr)[:, 1]

fig = plt.figure()
ax1 = fig.add_subplot(111)

colors = list(map(colorUtil.color, labelArr))

ax1.scatter(feature1, feature2, s=80, c=colors, marker='+')
b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

# to calculate w
xxx = np.array(alphas).T
xxy = np.array([labelArr])
npDataArr = np.array(dataArr)
w = np.dot(np.multiply(xxx, xxy), npDataArr)
support_vetor0 = []
support_vetor1 = []
# support_vetor_value = []
support_vetor_color_value = []
alpha_index = np.where(alphas.getA() > 0)
ff = npDataArr[alpha_index[0]]
support_vetor0 = npDataArr[alpha_index[0]][:, 0]
support_vetor1 = npDataArr[alpha_index[0]][:, 1]
support_vetor_labels = np.array(labelArr)[alpha_index[0]]
support_vetor_colors = list(map(colorUtil.color, support_vetor_labels))
# support_vetor_value = np.dot(w, npDataArr[alpha_index[0]].T) + b support vector value should be 1 or -1
ax1.scatter(support_vetor0, support_vetor1, s=100, facecolors='none', edgecolors=support_vetor_colors)
# to draw line for test set
slope = -w[0, 0] / w[0, 1]
xx = np.linspace(0, 10)
gax = (b / w[0, 1])[0, 0]
yy = xx * slope - gax
plt.ylim(ymax=6)
plt.ylim(ymin=-8)
ax1.plot(xx, yy)
plt.show()