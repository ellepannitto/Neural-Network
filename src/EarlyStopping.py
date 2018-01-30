
import numpy as np

import Params
#~ import Monk
import Iris
from NeuralNetwork import NeuralNetwork
import Plotting

class EarlyStopping:
	
	def __init__ (self, train_set, train_labels, validation_set, validation_labels, layers_size=(5,2), params=Params):
		self.train_set = train_set
		self.train_labels = train_labels
		self.validation_set = validation_set
		self.validation_labels = validation_labels
		
		self.layers_size = layers_size
		self.params = params

		
	#TODO: make this more resistant to spikes
	def find_best_epoch ( self, losses ):
		MAX_WINDOW = 50
		MIN_WINDOW = 5
		MAX_TOLLERANCE = 0.05
		for i in range (MAX_WINDOW, len(losses)):
			tollerance = MAX_TOLLERANCE * ( 1- i / len(losses) )
			window = int ( (MAX_WINDOW - MIN_WINDOW)*(1-i/len(losses)) + MIN_WINDOW )
			if all ([ losses[j]+tollerance < losses[i] for j in range(i-window, i) ] ) :
				print ("[DEBUG] arg min :{}".format(i-window+1))
				return np.argmin (losses[:i-window+1])
		
		#~ print ("[DEBUG] argmin of all the epochs") 
		return np.argmin (losses)
		
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
			#~ print ("trial {} : epochs {} accuracy {}".format (i, when_to_stop, accuracy_per_epoch[when_to_stop]))
			
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
	train_s = Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] 
	train_l = Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):]
	test_s =  Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] 
	test_l =  Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] 
	
	es = EarlyStopping (train_s, train_l, test_s, test_l, layers_size=(6,))
	es.perform (do_plots=True)
