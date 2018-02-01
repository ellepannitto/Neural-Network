
#~ import Monk
#~ import Iris
import MLCUP2017
import Params
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import Statistics

if __name__ == "__main__":
	
	#only one MONK
	#~ train_sets   = [ Monk.monk3_training_set ]
	#~ train_labels = [ [ y[0] for y in Monk.monk3_training_labels ] ]
	#~ test_sets    = [ Monk.monk3_test_set ]
	#~ test_labels  = [ [ y[0] for y in Monk.monk3_test_labels ] ]
	
	
	# Iris
	#~ train_sets = [ Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] ] 
	#~ train_labels = [ Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):] ] 
	#~ test_sets = [ Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] ] 
	#~ test_labels = [ Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] ] 
	
	# MLCUP2017
	test_len = int(len(MLCUP2017.cup_train_set)/10)
	train_sets = [ MLCUP2017.cup_train_set[test_len:] ] 
	train_labels = [ MLCUP2017.cup_train_labels[test_len:] ]
	test_sets = [ MLCUP2017.cup_train_set[:test_len] ] 
	test_labels = [ MLCUP2017.cup_train_set[:test_len] ]

	
	for i, train_s, train_l, test_s, test_l in zip ( range(1,len(train_sets)+1), train_sets, train_labels, test_sets, test_labels ):
		
		print ("--- TEST {} ---".format(i))
		
		
		clf = MLPRegressor(activation='logistic', solver='lbfgs', learning_rate_init=Params.ETA,
		                    momentum=Params.ALPHA, alpha=Params.LAMBDA, hidden_layer_sizes=Params.LAYERS_SIZE)

		clf.fit (train_s, train_l)
		
		predicted_train_l = clf.predict (train_s)
		#~ a = Statistics.MulticlassificationAccuracy ()
		a = Statistics.MEELoss ()
		for o,y in zip(predicted_train_l, train_l):
			a.update (o, y)
			
		print ("Accuracy on train set {}".format ( a.get() ))

		#~ a = Statistics.MulticlassificationAccuracy ()
		a = Statistics.MEELoss ()
		predicted_test_l = clf.predict (test_s)
		for o,y in zip(predicted_test_l, test_l):
			a.update (o, y)
			
		print ("Accuracy on test set {}".format ( a.get() ))
		
