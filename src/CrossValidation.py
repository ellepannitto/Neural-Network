'''
  this module contains an implementation of a k-fold cross validation to estimate the Mean Euclidean Error on a dataset, 
  that uses a neural network and EarlyStopping to find the best number of epoch to stop learning (the KFoldCrossValidation class).
'''

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
	'''
	  Implements a k-fold cross validation on a dataset.
	  The dataset is splitted in K equal parts, the K folds are run:
	   At every fold, a different fraction of the dataset of size 1/K is used as validation set, the remaining part of the sataset is used as train set.
	   EarlyStopping is used on the validation set to determine a good number of epoch to stop training.
	  
	  The result of this process is the mean number of epochs after which the training should be stopped, and the mean validation Mean Squared Error, obtained
	  by considering the average results on the validation set for each fold, and the standard deviation.
	  These results can be dumped on a file.
	'''
	
	def __init__ (self, dataset, labels, K, model_name, shuffle=True, params=Params ):
		'''
		  creates an instance of KFoldCrossValidation.
		  
		  :params:
		    dataset:    the input patterns on which the coss validation is performed.
		    labels :    the correct labels for dataset. labels[i] is the label for dataset[i].
		    K:          the number of fold.
		    model_name: a string that identifies this model.
		    shuffle:    if True, the dataset and labels will be shuffled before starting the cross validation. Default: True
		    params:     params used by the CrossValidation, EarlyStopping and NeuralNetwork.
		                params can be an instance of the ConfigurableParams class, or the global default params, (those defined in the module Params).
		                requested Params are:
		                 all the params required by NeuralNetwork (ETA, ETA_DECAY, ETA_DECREASING_PERIOD ETA_RANGE, LAMBDA, ALPHA, MINBATCH, MINIBATCH_SAMPLE, MAX_EPOCH)
		                 LAYERS_SIZE: a tuple that represents the number and sizes of the neural network hidden layers.
		 
		'''
		
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
		'''
		  performs the K-fold cross validation.
		  
		  :params:
		   do_plots: if True, after each completed neural network train, a plot with the learning curves is shown to the user. Default: False
		'''
			
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
		'''
		  writes the k-fold results on a file named with this model's name, specified as parameter to the __init__ function.
		  The content of the files includes: the parameters of this model, the mean Euclidean Error and the mean number of epochs
		'''
		
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
			
	def report_error ( self ):
		'''
		  writes that there was an error performing a kfold with this parameters on a file named with this model's name, specified as parameter to the __init__ function.
		  The content of the files includes: the parameters of this model, a string that explain that there was an error.
		'''
		
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
			fout.write ("\nTHERE WERE ERRORS WHILE EXECUTING KFOLD WITH THESE PARAMETERS\n")
			

#unit tests
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
