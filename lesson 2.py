#!/usr/python

from sklearn.datasets import load_iris

iris = load_iris()

print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]

for i in range(len(iris.target)):
	print "Example %d: label %s, feature %s" % (i, iris.target[i], iris.data[i])

# -----------------
# test and classify
# -----------------

import numpy as np
test_indexes = [0, 50, 100]

# training
training_target = np.delete(iris.target, test_indexes)
training_data = np.delete(iris.data, test_indexes, axis=0)

# testing
testing_target = iris.target[test_indexes]
testing_data = iris.data[test_indexes]

# classification
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_target)

# print both the expected labels and the predicted labels
print testing_target
print clf.predict(testing_data)

# copy-pasted visualisation of decision tree (as per tutorial)
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("iris.pdf")
graph.write_png("iris.png")
