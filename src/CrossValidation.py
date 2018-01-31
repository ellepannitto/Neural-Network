
import random
import copy
import time
import numpy as np

import EarlyStopping
import OneHotEncoder
import Params

#~ import Iris
#~ import Monk
import Machine


class KFoldCrossValidation:
	
	def __init__ (self, dataset, labels, K, model_name, shuffle=True, params=Params ):
		
		self.dataset = copy.deepcopy (dataset)
		self.labels = copy.deepcopy (labels)
		self.K = K
		self.model_name = model_name
		self.params = params
		if shuffle:
			indices = random.sample (range(len(self.dataset)), len(self.dataset))
			self.dataset = [ self.dataset[i] for i in indices ]
			self.labels = [ self.labels[i] for i in indices ]
		
	def perform (self, do_plots=False):
				
		#~ print ("*** Model: {} ***".format (self.model_name))
		
		validation_len = int(len(self.dataset)/self.K)
		accuracies = []
		epochs = []
		
		for i in range (self.K):
			
			if i == self.K-1:
				validation_s = self.dataset[validation_len*i : ]
				validation_l = self.labels[validation_len*i : ]
			else: 
				validation_s = self.dataset[validation_len*i : validation_len*(i+1)] 
				validation_l = self.labels[validation_len*i : validation_len*(i+1)] 
			
			train_s = self.dataset[0 : validation_len*i] + self.dataset[validation_len*(i+1) : ]
			train_l = self.labels[0 : validation_len*i] + self.labels[validation_len*(i+1) : ]
			
			es = EarlyStopping.EarlyStopping ( train_s, train_l, validation_s, validation_l, layers_size=self.params.LAYERS_SIZE, params=self.params )
			es.perform ( do_plots=do_plots )
			
			accuracies.append (es.mean_accuracy)
			epochs.append (es.mean_epochs)
			
			print ("[KFoldCrossValidation] {} - Fold {}/{} accuracy {}".format(self.model_name, i+1,self.K, es.mean_accuracy))
			
				
		self.mean_accuracy = np.average(accuracies)
		self.var_accuracy = np.std(accuracies)
		self.mean_epochs = np.average(epochs)
		self.var_epochs = np.std(epochs)
	
	def dump ( self ):
		
		with open ( "../dumps/"+self.model_name, "w" ) as fout:
			fout.write ("test finished at {}\n\n".format (time.strftime('%d/%m/%Y at %H:%M')))
			fout.write ("PARAMETERS\n")
			fout.write ("LAYERS_SIZE={}\n".format(self.params.LAYERS_SIZE))
			fout.write ("ALPHA={}\n".format(self.params.ALPHA))
			fout.write ("LAMBDA={}\n".format(self.params.LAMBDA))
			fout.write ("ETA_DECAY={}\n".format(self.params.ETA_DECAY))
			if self.params.ETA_DECAY:
				fout.write ("ETA_RANGE={}\n".format(self.params.ETA_RANGE))
				fout.write ("ETA_DECREASING_PERIOD={}\n".format(self.params.ETA_DECREASING_PERIOD))
			else:
				fout.write ("ETA={}\n".format(self.params.ETA))
			fout.write ("MINIBATCH={}\n".format (self.params.MINIBATCH))
			if self.params.MINIBATCH:
				fout.write ("MINIBATCH_SAMPLE={}\n".format (self.params.MINIBATCH_SAMPLE))
			fout.write ("\nRESULTS\n")
			fout.write ("K (folds number) {}\n".format (self.K))
			fout.write ("mean accuracy: {} +/- {}\n".format (self.mean_accuracy, self.var_accuracy))
			fout.write ("mean epochs: {} +/- {}\n".format (self.mean_epochs, self.var_epochs))
			fout.write ("\n")
			


if __name__=="__main__":
	
	# Iris
	#~ train_s = Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] 
	#~ train_l = Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):]
	#~ test_s =  Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] 
	#~ test_l =  Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] 
	
	# Monk 3
	#~ train_s = Monk.monk3_training_set
	#~ train_l = Monk.monk3_training_labels
	#~ test_s  = Monk.monk3_test_set
	#~ test_l  = Monk.monk3_test_labels
	
	# Machine
	train_s = Machine.machine_train_set
	train_l = Machine.machine_train_labels
	test_s = Machine.machine_test_set
	test_l = Machine.machine_test_labels 

	kfcv = KFoldCrossValidation ( train_s+test_s, train_l+test_l, K=10, model_name="Machine", shuffle=True )
	kfcv.perform( do_plots=True )
	kfcv.dump ()
