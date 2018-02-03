
import numpy as np
import Params
import copy
import Statistics
import MLCUP2017
import Plotting

class Baseline:
	
	def __init__ (self):
		self.validation_set = None
	
	def setInputDim (self, n):
		
		self.inputDim = n
	
	def setOutputDim(self, n):		
	
		self.outputDim = n
	
	
	def set_train ( self, train_set, train_labels ):
		self.train_set = copy.deepcopy (train_set)
		self.train_labels = copy.deepcopy (train_labels)
		self.setInputDim (len(train_set[0]))
		self.setOutputDim (len(train_labels[0]))

	def set_validation (self, validation_set, validation_labels):
		self.validation_set = copy.deepcopy (validation_set)
		self.validation_labels = copy.deepcopy (validation_labels)
	
	#LOL
	def learn ( self ):
		self.train_losses = []
		self.validation_losses = None if self.validation_set is None else []
		self.validation_accuracies = None if self.validation_set is None else []

		xtrain = self.train_set
		ytrain = self.train_labels
		
		for i in range (Params.MAX_EPOCH):
			
			loss = Statistics.MSELoss ()
			
			for i,x,y in zip (range(1,len(xtrain)+1), xtrain, ytrain):
				o = [ np.random.normal ( 0, 1 ) for _ in range(self.outputDim) ]
				loss.update(o, y)
			
			self.train_losses.append(loss.get())
			
			if self.validation_set is not None:
				loss = Statistics.MSELoss()
				accuracy = Statistics.MEELoss ()
				for x,y in zip (self.validation_set, self.validation_labels):
					o = [ np.random.normal ( 0, 1 ) for _ in range(self.outputDim) ]
					loss.update(o, y)
					accuracy.update (o, y)
				self.validation_losses.append (loss.get())
				self.validation_accuracies.append (accuracy.get())
	
	#ANCORA PIÙ LOL
	def predict (self, x):
		return [ np.random.normal ( 0, 1 ) for _ in range(self.outputDim) ]

if __name__=="__main__":
	
	#MONKs
	#~ train_sets   = [ Monk.monk1_training_set, Monk.monk2_training_set, Monk.monk3_training_set ]
	#~ train_labels = [ Monk.monk1_training_labels, Monk.monk2_training_labels, Monk.monk3_training_labels ]
	#~ test_sets    = [ Monk.monk1_test_set, Monk.monk2_test_set, Monk.monk3_test_set ]
	#~ test_labels  = [ Monk.monk1_test_labels, Monk.monk2_test_labels, Monk.monk3_test_labels ]
	
	#only one MONK
	#~ train_sets   = [ Monk.monk3_training_set ]
	#~ train_labels = [ Monk.monk3_training_labels ]
	#~ test_sets    = [ Monk.monk3_test_set ]
	#~ test_labels  = [ Monk.monk3_test_labels ]
	
	# XOR
	#~ train_sets = [ [[0, 0], [0,1], [1,0], [1,1]] ]
	#~ train_labels = [ [[0],  [1],   [1],  [0] ] ]
	#~ test_sets = [ [[1,1] ] ]
	#~ test_labels = [ [[0] ] ]
	
	# Iris
	#~ train_sets = [ Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] ] 
	#~ train_labels = [ Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):] ] 
	#~ test_sets = [ Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] ] 
	#~ test_labels = [ Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] ] 

	# Machine
	#~ train_sets = [ Machine.machine_train_set ] 
	#~ train_labels = [ Machine.machine_train_labels ] 
	#~ test_sets = [ Machine.machine_test_set ] 
	#~ test_labels = [ Machine.machine_test_labels ] 

	# MLCUP2017
	test_len = int(len(MLCUP2017.cup_train_set)/4)
	train_sets = [ MLCUP2017.cup_train_set[test_len:] ] 
	train_labels = [ MLCUP2017.cup_train_labels[test_len:] ]
	test_sets = [ MLCUP2017.cup_train_set[:test_len] ] 
	test_labels = [ MLCUP2017.cup_train_set[:test_len] ]

	for i, train_s, train_l, test_s, test_l in zip ( range(1,len(train_sets)+1), train_sets, train_labels, test_sets, test_labels ):
		
		print ("--- TEST {} ---".format(i))
		
		myNN = Baseline()
		
		myNN.set_train (train_s, train_l)
		myNN.set_validation (test_s, test_l)
		
		myNN.learn()
		
		Plotting.plot_loss_accuracy_per_epoch (myNN)
		
		#~ a = Statistics.MulticlassificationAccuracy ()
		#~ a = Statistics.Accuracy ()
		#~ a = Statistics.MSELoss ()
		a = Statistics.MEELoss ()
		for x,y in zip(train_s, train_l):
			o = myNN.predict (x)
			a.update (o, y)
			
		print ("Accuracy on train set {}".format ( a.get() ))

		#~ a = Statistics.MulticlassificationAccuracy ()
		#~ a = Statistics.Accuracy ()
		#~ a = Statistics.MSELoss ()
		a = Statistics.MEELoss ()
		for x,y in zip(test_s, test_l):
			o = myNN.predict (x)
			a.update (o, y)
			
		print ("Accuracy on test set {}".format ( a.get() ))

