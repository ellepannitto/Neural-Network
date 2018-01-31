
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


fin = open ("../datasets/machine.txt")
lines =[ line.split(',') for line in fin.readlines() ]
random.shuffle (lines)

dataset = [ [float(el) for el in line[2:9]] for line in lines ]

mean_per_attribute = np.average( dataset, axis=0 )
std_per_attribute = np.std( dataset, axis=0, ddof=1 )

normalized_dataset = normalize_matrix_zscore ( dataset, mean_per_attribute, std_per_attribute )

test_size = int (len(lines)*25/100)

machine_train_set = [ [el for el in line[0:6]] for line in normalized_dataset[test_size:] ]
machine_train_labels = [ [line[6]] for line in normalized_dataset[test_size:] ]

#~ print ("shape of train set: {}x{}".format(len(machine_train_set), len(machine_train_set[0])))

#~ print ("train set: {}".format (machine_train_set))
#~ print ("train lab: {}".format (machine_train_labels))

machine_test_set    = [ [el for el in line[0:6]] for line in normalized_dataset[:test_size] ]
machine_test_labels = [ [line[6] ] for line in normalized_dataset[:test_size] ]

#~ print ("shape of test set: {}x{}".format(len(machine_test_set), len(machine_test_set[0])))

#~ print ("mean per attribute: {}".format(mean_per_attribute))
#~ print ("std per attribute: {}".format(std_per_attribute))
