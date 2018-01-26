
import Params
import Monk
import numpy as np
import OneHotEncoder
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

class EarlyStopping:
	
	def __init__ (self, train_set, train_labels, validation_set, validation_labels, layers_size=(5,2), model_name="unnamed"):
		self.train_set = train_set
		self.train_labels = train_labels
		self.validation_set = validation_set
		self.validation_labels = validation_labels
		
		self.model_name = model_name
		
		self.layers_size = layers_size
		
	#TODO: maybe it can be otpimized :)
	def find_best_epoch ( self, losses ):
		WINDOW = 5
		TOLLERANCE = 0.001
		for i in range (WINDOW, Params.MAX_EPOCH):
			if all ([ losses[j] < losses[i] for j in range(i-WINDOW, i) ] ) or losses[i-WINDOW]<losses[i]+TOLLERANCE:
				return np.argmin (losses[:i-WINDOW])
		
		return np.argmin (losses)
		
	def perform (self, do_plots = False):
		
		print ("*** Model: {} ***".format (self.model_name))
		
		stats = []
		for i in range (Params.NUM_TRIALS_PER_CONFIGURATION):
			myNN = NeuralNetwork()	
			myNN.setInputDim (len(self.train_set[0]))
			myNN.setOutputDim (len(self.train_labels[0]))
			
			for size in self.layers_size:
				myNN.addLayer(size)
			
			myNN.set_train (self.train_set, self.train_labels)
			myNN.set_validation (self.validation_set, self.validation_labels)
			
			myNN.learn()
			
			accuracy_per_epoch = myNN.validation_accuracies
			loss_per_epoch = myNN.validation_losses
			
			when_to_stop = self.find_best_epoch ( loss_per_epoch )
			print ("trial {} : epochs {} accuracy {}".format (i, when_to_stop, accuracy_per_epoch[when_to_stop]))
			
			if do_plots:
				plt.plot(list(range(len(myNN.train_losses))), myNN.train_losses, 'r--', label='train error')
				plt.plot(list(range(len(myNN.validation_losses))), myNN.validation_losses, 'b-', label='validation error')
				plt.plot(list(range(len(myNN.validation_accuracies))), myNN.validation_accuracies, 'k-', label='validation Accuracy')
				plt.plot([when_to_stop,when_to_stop], [0,1],'g')
				plt.legend()
				plt.ylabel('Loss')
				plt.xlabel('epoch')
				axes = plt.gca()
				axes.set_xlim([0,Params.MAX_EPOCH])
				axes.set_ylim([0,1])
				plt.show()
			
			stats.append ((when_to_stop, accuracy_per_epoch[when_to_stop]))
		
		self.mean_epochs = np.average([e[0] for e in stats])
		self.var_epochs = np.std([e[0] for e in stats])
		self.mean_accuracy = np.average([e[1] for e in stats])
		self.var_accuracy = np.std([e[1] for e in stats])
		print ("mean: epochs {:.4f} +/- {:.4f} accuracy {:.4f} +/- {:.4f}".format( self.mean_epochs, self.var_epochs, self.mean_accuracy, self.var_accuracy) ) 
	
if __name__ == "__main__":
	
	train_s = Monk.monk3_training_set
	train_l = Monk.monk3_training_labels
	test_s  = Monk.monk3_test_set
	test_l  = Monk.monk3_test_labels
	
	encoded_train_s = OneHotEncoder.encode_int_matrix (train_s)
	encoded_test_s = OneHotEncoder.encode_int_matrix (test_s)
	
	es = EarlyStopping (encoded_train_s, train_l, encoded_test_s, test_l, layers_size=(2,), model_name="Monk 3")
	es.perform (do_plots=True)
