
import numpy as np

import Params
#~ import Monk
#~ import Iris
#~ import Machine
import MLCUP2017
from NeuralNetwork import NeuralNetwork
import Plotting

class EarlyStopping:
	
	def __init__ (self, train_set, train_labels, validation_set, validation_labels, layers_size=(5,2), params=Params):
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
