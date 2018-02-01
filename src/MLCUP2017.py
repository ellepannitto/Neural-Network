
import random
import numpy as np

random.seed(0)

def normalize_matrix_zscore ( matrix, avgs, stds ):
	normalized_matrix=[]
	for i in range (len(matrix)):
		row = []
		for j in range (len(matrix[i])):
			row.append ( (matrix[i][j] - avgs[j])/stds[j] )
		normalized_matrix.append(row)
	return normalized_matrix


fin = open ("../datasets/ML-CUP17-TR.csv")
train_dataset = [ [ float(el) for el in line.strip().split(',')] for line in fin.readlines() if line!="\n" and not line.startswith("#") ] 
random.shuffle (train_dataset)

mean_per_attribute_dataset = np.average( train_dataset, axis=0 )
std_per_attribute_dataset  = np.std( train_dataset, axis=0, ddof=1 )

#~ print ("mean per attribute (whole dataset): {}".format(mean_per_attribute_dataset))
#~ print ("std per attribute (whole dataset): {}".format(std_per_attribute_dataset))

train_dataset = normalize_matrix_zscore ( train_dataset, mean_per_attribute_dataset, std_per_attribute_dataset )

validation_len = int(len(train_dataset) * 25 / 100)

cup_train_set = [ row[1:11] for row in train_dataset[validation_len:] ]
cup_validation_set = [ row[1:11] for row in train_dataset[:validation_len] ]

cup_train_labels = [ row[11:13] for row in train_dataset[validation_len:] ]
cup_validation_labels = [ row[11:13] for row in train_dataset[:validation_len] ]


#~ print ("shape of train set: {}x{}".format(len(cup_train_set), len(cup_train_set[0])))
#~ print ("shape of valid set: {}x{}".format(len(cup_validation_set), len(cup_validation_set[0])))

#~ print ("shape of train labels: {}x{}".format(len(cup_train_labels), len(cup_train_labels[0])))
#~ print ("shape of valid labels: {}x{}".format(len(cup_validation_labels), len(cup_validation_labels[0])))

#~ print ("train labels {}".format(cup_train_labels))
#~ print ("valid labels {}".format(cup_validation_labels))

fin = open ("../datasets/ML-CUP17-TS.csv")
test_dataset = [ [ float(el) for el in line.strip().split(',')] for line in fin.readlines() if line!="\n" and not line.startswith("#") ]
test_dataset = normalize_matrix_zscore ( test_dataset, mean_per_attribute_dataset[:-2], std_per_attribute_dataset[:-2] )

cup_test_set = [ row[1:11] for row in test_dataset ]

#~ print ("shape of test set: {}x{}".format(len(cup_test_set), len(cup_test_set[0])))
#~ print ("test set: {}".format(cup_test_set))
