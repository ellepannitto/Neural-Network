
import random
import copy
import EarlyStopping
import OneHotEncoder

import Iris
import Monk
#~ import HAR


class KFoldCrossValidation:
	
	def __init__ (self, dataset, labels, K, model_name, shuffle=True ):
		
		self.dataset = copy.deepcopy (dataset)
		self.labels = copy.deepcopy (labels)
		self.K = K
		self.model_name = model_name
		if shuffle:
			indices = random.sample (range(len(self.dataset)), len(self.dataset))
			self.dataset = [ self.dataset[i] for i in indices ]
			self.labels = [ self.labels[i] for i in indices ]
		
	def perform (self, do_plots=False):
		
		validation_len = int(len(self.dataset)/self.K)
		
		print ("*** Model: {} ***".format (self.model_name))
		
		for i in range (self.K):
			
			if i == self.K-1:
				validation_s = self.dataset[validation_len*i : ]
				validation_l = self.labels[validation_len*i : ]
			else: 
				validation_s = self.dataset[validation_len*i : validation_len*(i+1)] 
				validation_l = self.labels[validation_len*i : validation_len*(i+1)] 
			
			train_s = self.dataset[0 : validation_len*i] + self.dataset[validation_len*(i+1) : ]
			train_l = self.labels[0 : validation_len*i] + self.labels[validation_len*(i+1) : ]
			
			es = EarlyStopping.EarlyStopping ( train_s, train_l, validation_s, validation_l, layers_size=(6,) )
			
			print ("Fold {}/{}".format(i+1,self.K))
			#~ print ("Train sample: {}".format(len(train_s)))
			#~ print ("Valid sample: {}".format(len(validation_s)))
			
			es.perform ( do_plots=do_plots )
					


if __name__=="__main__":
	
	# Iris
	train_s = Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] 
	train_l = Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):]
	test_s =  Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] 
	test_l =  Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] 
	
	# Monk 3
	#~ train_s = Monk.monk3_training_set
	#~ train_l = Monk.monk3_training_labels
	#~ test_s  = Monk.monk3_test_set
	#~ test_l  = Monk.monk3_test_labels
			
	kfcv = KFoldCrossValidation ( train_s, train_l, K=10, model_name="Iris", shuffle=True )
	kfcv.perform( do_plots=False )
