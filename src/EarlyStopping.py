
import Params
import numpy as np

class EarlyStopping:
	
	def __init__ (self, train_set, train_labels, validation_set, validation_labels, layers_size=(5,2), model_name="unnamed"):
		self.train_set = train_set
		self.train_labels = train_labels
		self.validation_set = validation_set
		self.validation_labels = validation_labels
		
		self.model_name = model_name
		
		self.layers_size = layers_size
		
	#TODO: moch: find a better implementation
	def find_best_epoch ( self, losses ):
		WINDOW = 5
		for i in range (WINDOW, Params.MAX_EPOCH):
			if all ([ losses[i]>losses[j] for j in range(i-WINDOW, i) ] ):
				return np.argmin (losses[:i-WINDOW])
		
		return np.argmin (losses)
		
	def perform (self, do_plots = False):
		
		print ("*** Model {}: ***".format (self.model_name))
		
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
		self.var_epochs = np.var([e[0] for e in stats])
		self.mean_accuracy = np.average([e[1] for e in stats])
		self.var_accuracy = np.var([e[1] for e in stats])
		print ("mean: epochs {} +/- {} accuracy {} +/- {}".format( self.mean_epochs, self.var_epochs, self.mean_accuracy, self.car_accuracy) ) 
	
if __name__ == "__main__":
	
