'''
  this module contains an utility (the class EarluStopping) to determine the best number of epochs after which the neural network training must be stopped,
  looking at the learning curves.
'''

import numpy as np

import Params
#~ import Monk
#~ import Iris
#~ import Machine
import MLCUP2017
from NeuralNetwork import NeuralNetwork
import Plotting

class EarlyStopping:
	'''
	  trains a neural network given a train and validation, then determines the best number of epochs after which the neural network training must be stopped.
	'''
	
	def __init__ (self, train_set, train_labels, validation_set, validation_labels, layers_size=(5,2), params=Params):
		'''
		  creates an instance of EarlyStopping
		  :params:
		   train_set:         sequence of instance to use as train set
		   train_labels:      expected outputs for the instance of train_set. train_labels[i] is the expected output for the pattern train_set[i]
		   validation_set:    sequence of instance to use as validation set
		   validation_labels: expected outputs for the instance of validation_set. validation_labels[i] is the expected output for the pattern validation_set[i]
		   layers_size:       a tuple that represents the number and sizes of the neural network hidden layers.
		   params: params used by the CrossValidation, EarlyStopping and NeuralNetwork.
		           params can be an instance of the ConfigurableParams class, or the global default params, (those defined in the module Params).
		           requested Params are:
		            all the params required by NeuralNetwork (ETA, ETA_DECAY, ETA_DECREASING_PERIOD ETA_RANGE, LAMBDA, ALPHA, MINBATCH, MINIBATCH_SAMPLE, MAX_EPOCH)
		    		   
		'''
		self.train_set = train_set
		self.train_labels = train_labels
		self.validation_set = validation_set
		self.validation_labels = validation_labels
		
		#~ print ("[EarlyStopping] train_set dim: {}".format(len(train_set)))
		#~ print ("[EarlyStopping] validation_set dim: {}".format(len(validation_set)))	
		#~ input()
		
		self.layers_size = layers_size
		self.params = params

		
	def find_best_epoch ( self, losses ):
		'''
		  examines the learning curve (validation loss per epoch) to determine the point after which stop the training.
		  
		  :params:
		   losses: a list with, for every position i, the loss at epoch i
		  
		  Since the learning is typically unstable during the first few epochs, this function tries
		  to skip this ”spiky zone”.
		  The parameter WINDOW allows to compare the validation loss at every
		  epoch with the average of the losses in the previous epochs, limited by the
		  window size.
		  this function stops when an epoch such that the loss is
		  greater than the average of the losses in the previous epochs minus a certain
		  tolerance is found. The parameters WINDOW was empirically found and fixed, and the
		  tolerance value increases linearly with the epochs: it starts with a negative
		  MIN_TOL value (therefore allowing the error to increase in the first epochs)
		  and it ends with a positive MAX_TOL value. In this way, this function
		  stops if the error is not decreasing enough.
		  In symbols, EarlyStopping decides to stop learning at epoch e* such that
		  
                                        WINDOW
		                           1     ____
		  e* = min losses[e] > --------- \     losses[e-i] - tollerance(e) 
		        e                WINDOW  /___
		                                  i=1  
		  
		  where 
		                                 e 
		  tollerance(e) = MIN_TOL + ----------- ( MAX_TOL - MIN_TOL )
		                             MAX_EPOCH
		                             
		'''
		
		MIN_WINDOW = 5
		INITIAL_TOLLERANCE = -0.05
		FINAL_TOLLERANCE = 0.01
		prev_avg = 0
		SPIKY_ZONE = int (len(losses) / 10)
		for i in range (SPIKY_ZONE):
			prev_avg += losses[i]/SPIKY_ZONE
		
		#~ print ("[DEBUG] initial prev_avg: {}".format(prev_avg))
		for i in range (SPIKY_ZONE+MIN_WINDOW, len(losses)):
			tollerance = INITIAL_TOLLERANCE + (FINAL_TOLLERANCE - INITIAL_TOLLERANCE) * ( i / len(losses) )
			#~ print ("[DEBUG] i={} loss={} prev_avg-toll={}".format(i, losses[i], prev_avg-tollerance))
			if losses[i] > prev_avg-tollerance :
				#~ print ("[DEBUG] stopping")
				return SPIKY_ZONE + np.argmin (losses[SPIKY_ZONE:i])
			prev_avg -= losses[i-MIN_WINDOW]/MIN_WINDOW
			prev_avg += losses[i] / MIN_WINDOW
		
		#~ print ("[DEBUG] argmin of all the epochs") 
		return SPIKY_ZONE + np.argmin (losses[SPIKY_ZONE:])
		
		
	def perform (self, do_plots = False):
		'''
		  performs the search of the best number of epoch for the specified dataset.
		  
		  :params:
		   do_plots: if True, after each completed neural network train, a plot with the learning curves is shown to the user. Default: False
		'''
		
		stats = []
		for i in range (Params.NUM_TRIALS_PER_CONFIGURATION):
			myNN = NeuralNetwork( self.params )
			
			for size in self.layers_size:
				myNN.addLayer(size)
			
			myNN.set_train (self.train_set, self.train_labels)
			myNN.set_validation (self.validation_set, self.validation_labels)
			
			myNN.learn()
			
			accuracy_per_epoch = myNN.validation_accuracies
			loss_per_epoch = myNN.validation_losses
			
			when_to_stop = self.find_best_epoch ( loss_per_epoch )
			#~ print ("[EarlyStopping] trial {} : epochs {} accuracy {}".format (i, when_to_stop, accuracy_per_epoch[when_to_stop]))
			
			if do_plots:
				Plotting.plot_loss_accuracy_per_epoch (myNN, show=False)
				Plotting.plot_vertical_line ( when_to_stop )
				Plotting.show_plot()
			
			stats.append ((when_to_stop, accuracy_per_epoch[when_to_stop]))
		
		self.mean_epochs = np.average([e[0] for e in stats])
		self.var_epochs = np.std([e[0] for e in stats])
		self.mean_accuracy = np.average([e[1] for e in stats])
		self.var_accuracy = np.std([e[1] for e in stats])
		#~ print ("mean: epochs {:.0f} +/- {:.4f} accuracy {:.4f} +/- {:.4f}".format( self.mean_epochs, self.var_epochs, self.mean_accuracy, self.var_accuracy) ) 

# unit tests
if __name__ == "__main__":
	
	# Iris
	#~ train_s = Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] 
	#~ train_l = Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):]
	#~ test_s =  Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] 
	#~ test_l =  Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] 
	
	# Machine
	#~ train_s = Machine.machine_train_set
	#~ train_l = Machine.machine_train_labels 
	#~ test_s = Machine.machine_test_set
	#~ test_l = Machine.machine_test_labels 
	
	# MLCUP2017
	test_len = int(len(MLCUP2017.cup_train_set)/10)
	train_s = MLCUP2017.cup_train_set[test_len:]
	train_l = MLCUP2017.cup_train_labels[test_len:]
	test_s = MLCUP2017.cup_train_set[:test_len]  
	test_l = MLCUP2017.cup_train_set[:test_len] 
	
	es = EarlyStopping (train_s, train_l, test_s, test_l, layers_size=Params.LAYERS_SIZE)
	es.perform (do_plots=True)
