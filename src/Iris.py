'''
  Loads the Iris dataset (multiclass classification task), one-hot encoding its
  categorical output labels and splitting it in two parts: train set (75%) and
  test set(25%).
  
  see also: Ronald A Fisher. “The use of multiple measurements in taxonomic problems”. In: Annals of human genetics 7.2 (1936), pp. 179–188.
'''

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import random

#~ random.seed(0)

fin = open ("../datasets/iris_data.txt")
lines =[ line.split(',') for line in fin.readlines() ]
random.shuffle (lines)

iris_train_set = [ [ float( line[i] ) for i in range(len(line)-1) ] for line in lines ]

ohe = OneHotEncoder ( categorical_features = "all", sparse=False )
lae = LabelEncoder ()
columns = [ line[-1] for line in lines ]
iris_train_labels = list (ohe.fit_transform ( [ [num] for num in lae.fit_transform (columns) ] ))

ohe = None
del (ohe)

fin = None
del (fin)

lines = None
del (lines)

columns = None
del (columns)
