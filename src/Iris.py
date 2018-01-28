
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import random

fin = open ("../datasets/iris_data.txt")
lines =[ line.split(',') for line in fin.readlines() ]
random.shuffle (lines)

iris_train_set = [ [ float( line[i] ) for i in range(len(line)-1) ] for line in lines ]

ohe = OneHotEncoder ( categorical_features = "all", sparse=False )
lae = LabelEncoder ()
columns = [ line[-1] for line in lines ]
iris_train_labels = ohe.fit_transform ( [ [num] for num in lae.fit_transform (columns) ] )

ohe = None
del (ohe)

fin = None
del (fin)

lines = None
del (lines)

columns = None
del (columns)
