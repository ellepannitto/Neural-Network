
import random
import copy
import EarlyStopping
import Iris
import OneHotEncoder
import Params

from sklearn.neural_network import MLPClassifier


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
			print ("Train sample: {}".format(len(train_s)))
			print ("Valid sample: {}".format(len(validation_s)))
			
			es.perform ( do_plots=do_plots )
		
			#~ train_l = [ el[0] for el in train_l ]
			#~ validation_l = [ el[0] for el in validation_l ]
			
			#~ print ("SKLEARN")
			#~ clf = MLPClassifier(activation='logistic', solver='lbfgs', learning_rate_init=Params.ETA,
		                    #~ momentum=Params.ALPHA, alpha=Params.LAMBDA, hidden_layer_sizes=(6,), early_stopping=True)
			#~ clf.fit (train_s, train_l)
		
			#~ l=[]
			#~ predicted_train_l = clf.predict (train_s)
			#~ for o,y in zip(predicted_train_l, train_l):
				#~ l.append(1 if ((o>0.5 and y>0.5) or (o<=0.5 and y<=0.5)) else 0)
				
			#~ print ("Accuracy on train set {}".format ( (sum(l)*1.0/len(l))) )

			#~ l=[]
			#~ predicted_validation_l = clf.predict (validation_s)
			#~ for o,y in zip(predicted_validation_l, validation_l):
				#~ l.append(1 if ((o>0.5 and y>0.5) or (o<=0.5 and y<=0.5)) else 0)
				
			#~ print ("Accuracy on validation set {}".format ( (sum(l)*1.0/len(l))) )

				
			


if __name__=="__main__":
	
	# Iris
	train_s = Iris.iris_train_set[int(len(Iris.iris_train_set)/8):] 
	train_l = Iris.iris_train_labels[int(len(Iris.iris_train_set)/8):]
	test_s =  Iris.iris_train_set[:int(len(Iris.iris_train_set)/8)] 
	test_l =  Iris.iris_train_labels[:int(len(Iris.iris_train_set)/8)] 
	
	kfcv = KFoldCrossValidation ( train_s+test_s, train_l+test_l, K=8, model_name="Iris", shuffle=False )
	kfcv.perform( do_plots=True )
