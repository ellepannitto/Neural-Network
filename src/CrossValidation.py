
import random
import copy
import NeuralNetwork

class KFoldCrossValidation:
	
	def __init__ (self, dataset, labels, K, shuffle=True ):
		
		self.dataset = copy.deepcopy (dataset)
		self.labels = copy.deepcopy (labels)
		self.K = K
		if shuffle:
			indices = random.sample (range(len(self.dataset)), len(self.dataset))
			self.dataset = [ self.dataset[i] for i in indices ]
			self.labels = [ self.labels[i] for i in indices ]
			
		#~ print ("shuffled dataset: {}".format (self.dataset))
		#~ print ("shuffled labels: {}".format (self.labels))
	
	def perform (self):
		
		validation_len = int(len(self.dataset)/self.K)
		
		for i in range (self.K):
			
			if i == self.K-1:
				validation_s = self.dataset[validation_len*i : ]
				validation_l = self.labels[validation_len*i : ]
			else: 
				validation_s = self.dataset[validation_len*i : validation_len*(i+1)] 
				validation_l = self.dataset[validation_len*i : validation_len*(i+1)] 
			
			train_s = self.dataset[0 : validation_len*i] + self.dataset[validation_len*(i+1) : ]
			train_l = self.labels[0 : validation_len*i] + self.labels[validation_len*(i+1) : ]
			
			myNN = NeuralNetwork()
			
			myNN.addLayer (6)
			
			myNN.set_train (train_s, train_l)
			myNN.set_validation (validation_s, validation_l)
			
			myNN.learn()
		
			
			print ("fold: {} train_s: {} validation_s: {}".format(i, train_s, validation_s))



if __name__=="__main__":

	kfcv = KFoldCrossValidation ( list(range(20)), list(range(100, 120)), 3 )
	kfcv.perform()
