
import OneHotEncoder

#interesting columns in the dataset, 1-based
columns = [ 1,2,3, 41,42,43, 81,82,83, 121, 161, 201, 214, 227, 253, ] #+ [555, 556, 557, 558, 559, 560, 561]

fin = open ("../datasets/HAR_X_train.txt")
lines = [ line.split() for line in fin.readlines() ]
HAR_train_set = [ [ float( line[i-1] ) for i in columns ] for line in lines ]

fin = open ("../datasets/HAR_y_train.txt")
lines = [ line.split() for line in fin.readlines() ]
HAR_train_labels = OneHotEncoder.encode_int_matrix ( [ [ int( line[0] ) ] for line in lines ] )

fin = open ("../datasets/HAR_X_test.txt")
lines = [ line.split() for line in fin.readlines() ]
HAR_test_set = [ [ float( line[i-1] ) for i in columns ] for line in lines ]

fin = open ("../datasets/HAR_y_test.txt")
lines = [ line.split() for line in fin.readlines() ]
HAR_test_labels = OneHotEncoder.encode_int_matrix ( [ [ int( line[0] ) ] for line in lines ] )

fin = None
del (fin)

columns = None
del (columns)

lines = None
del (lines)
