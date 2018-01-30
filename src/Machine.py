
import random

random.seed(0)

fin = open ("../datasets/machine.txt")
lines =[ line.split(',') for line in fin.readlines() ]
random.shuffle (lines)

test_size = int (len(lines)*75/100)

machine_train_set = [ [float(el) for el in line[2:8]] for line in lines[test_size:] ]
machine_train_labels = [ [float(line[8])] for line in lines[test_size:] ]

machine_test_set = [ [float(el) for el in line[2:8]] for line in lines[:test_size] ]
machine_test_labels = [ [float(line[8])] for line in lines[:test_size] ]

#~ print ("train size: {}".format(len(machine_train_set)))
#~ print ("first 10 train elements: {}".format(machine_train_set[:10]))
#~ print ("first 10 train labels  : {}".format(machine_train_labels[:10]))


#~ print ("test size: {}".format(len(machine_test_set)))
#~ print ("first 10 test elements: {}".format(machine_test_set[:10]))
#~ print ("first 10 test labels  : {}".format(machine_test_labels[:10]))
