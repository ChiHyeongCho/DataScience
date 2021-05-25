
from sklearn import tree
from sklearn.tree import export_graphviz
import numpy as np
import pydotplus
from IPython.display import Image
import os, sys
PATH = 'graphviz-2.38\\release\\bin'
os.environ["PATH"] += os.pathsep + PATH


data = np.array([[0, 0, 1, 0, 0],
                [1, 0, 2, 0, 0],
                [0, 1, 2, 0, 1],
                [2, 1, 0, 2, 1],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 2, 0],
                [1, 1, 0, 2, 0],
                [0, 0, 2, 1, 0]])

x = data[:,0:4]
y = data[:,4]
print(x, '\n')
print(y)

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
clf.fit(x,y)

clf.predict([[0, 0, 1, 0]])

dot_data = export_graphviz(clf)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())